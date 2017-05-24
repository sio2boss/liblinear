#!/usr/bin/env python

import os, sys
sys.path = [os.path.dirname(os.path.abspath(__file__))] + sys.path
from ctypes import *
from ctypes.util import find_library
from os import path
import scipy
from scipy import sparse

if sys.version_info[0] < 3:
	range = xrange
	from itertools import izip as zip
	
__all__ = ['svm_read_problem', 'load_model', 'save_model', 'evaluations',
           'train', 'predict', 'liblinear', 'feature_node', 'gen_feature_nodearray', 'problem',
           'parameter', 'model', 'toPyModel', 'L2R_LR', 'L2R_L2LOSS_SVC_DUAL',
           'L2R_L2LOSS_SVC', 'L2R_L1LOSS_SVC_DUAL', 'MCSVM_CS',
           'L1R_L2LOSS_SVC', 'L1R_LR', 'L2R_LR_DUAL', 'L2R_L2LOSS_SVR',
           'L2R_L2LOSS_SVR_DUAL', 'L2R_L1LOSS_SVR_DUAL', 'print_null']

try:
	dirname = path.dirname(path.abspath(__file__))
	if sys.platform == 'win32':
		liblinear = CDLL(path.join(dirname, r'..\windows\liblinear.dll'))
	else:
		liblinear = CDLL(path.join(dirname, '../liblinear.so.3'))
except:
# For unix the prefix 'lib' is not considered.
	if find_library('linear'):
		liblinear = CDLL(find_library('linear'))
	elif find_library('liblinear'):
		liblinear = CDLL(find_library('liblinear'))
	else:
		raise Exception('LIBLINEAR library not found.')

L2R_LR = 0
L2R_L2LOSS_SVC_DUAL = 1
L2R_L2LOSS_SVC = 2
L2R_L1LOSS_SVC_DUAL = 3
MCSVM_CS = 4
L1R_L2LOSS_SVC = 5
L1R_LR = 6
L2R_LR_DUAL = 7
L2R_L2LOSS_SVR = 11
L2R_L2LOSS_SVR_DUAL = 12
L2R_L1LOSS_SVR_DUAL = 13

PRINT_STRING_FUN = CFUNCTYPE(None, c_char_p)
def print_null(s):
	return

def genFields(names, types):
	return list(zip(names, types))

def fillprototype(f, restype, argtypes):
	f.restype = restype
	f.argtypes = argtypes

class feature_node(Structure):
	_names = ["index", "value"]
	_types = [c_int, c_double]
	_fields_ = genFields(_names, _types)

	def __str__(self):
		return '%d:%g' % (self.index, self.value)

def gen_feature_nodearray(xi, feature_max=None):
	if feature_max:
		assert(isinstance(feature_max, int))

	xi_shift = 0 # ensure correct indices of xi
	if scipy and isinstance(xi, tuple) and len(xi) == 2\
			and isinstance(xi[0], scipy.ndarray) and isinstance(xi[1], scipy.ndarray): # for a sparse vector
		index_range = xi[0] + 1 # index starts from 1
		if feature_max:
			index_range = index_range[scipy.where(index_range <= feature_max)]
	elif scipy and isinstance(xi, scipy.ndarray):
		xi_shift = 1
		index_range = xi.nonzero()[0] + 1 # index starts from 1
		if feature_max:
			index_range = index_range[scipy.where(index_range <= feature_max)]
	elif isinstance(xi, (dict, list, tuple)):
		if isinstance(xi, dict):
			index_range = xi.keys()
		elif isinstance(xi, (list, tuple)):
			xi_shift = 1
			index_range = range(1, len(xi) + 1)
		index_range = filter(lambda j: xi[j-xi_shift] != 0, index_range)

		if feature_max:
			index_range = filter(lambda j: j <= feature_max, index_range)
		index_range = sorted(index_range)
	else:
		raise TypeError('xi should be a dictionary, list, tuple, 1-d numpy array, or tuple of (index, data)')

	ret = (feature_node*(len(index_range)+2))()
	ret[-1].index = -1 # for bias term
	ret[-2].index = -1

	if scipy and isinstance(xi, tuple) and len(xi) == 2\
			and isinstance(xi[0], scipy.ndarray) and isinstance(xi[1], scipy.ndarray): # for a sparse vector
		for idx, j in enumerate(index_range):
			ret[idx].index = j
			ret[idx].value = (xi[1])[idx]
	else:
		for idx, j in enumerate(index_range):
			ret[idx].index = j
			ret[idx].value = xi[j - xi_shift]

	max_idx = 0
	if len(index_range) > 0:
		max_idx = index_range[-1]
	return ret, max_idx

try:
	from numba import jit
	jit_enabled = True
except:
	jit = lambda x: x
	jit_enabled = False

@jit
def csr_to_problem_jit(l, x_val, x_ind, x_rowptr, prob_val, prob_ind, prob_rowptr):
	for i in range(l):
		b1,e1 = x_rowptr[i], x_rowptr[i+1]
		b2,e2 = prob_rowptr[i], prob_rowptr[i+1]-2
		for j in range(b1,e1):
			prob_ind[j-b1+b2] = x_ind[j]+1
			prob_val[j-b1+b2] = x_val[j]
def csr_to_problem_nojit(l, x_val, x_ind, x_rowptr, prob_val, prob_ind, prob_rowptr):
	for i in range(l):
		x_slice = slice(x_rowptr[i], x_rowptr[i+1])
		prob_slice = slice(prob_rowptr[i], prob_rowptr[i+1]-2)
		prob_ind[prob_slice] = x_ind[x_slice]+1
		prob_val[prob_slice] = x_val[x_slice]

def csr_to_problem(x, prob):
	# Extra space for termination node and (possibly) bias term
	x_space = prob.x_space = scipy.empty((x.nnz+x.shape[0]*2), dtype=feature_node)
	prob.rowptr = x.indptr.copy()
	prob.rowptr[1:] += 2*scipy.arange(1,x.shape[0]+1)
	prob_ind = x_space["index"]
	prob_val = x_space["value"]
	prob_ind[:] = -1
	if jit_enabled:
		csr_to_problem_jit(x.shape[0], x.data, x.indices, x.indptr, prob_val, prob_ind, prob.rowptr)
	else:
		csr_to_problem_nojit(x.shape[0], x.data, x.indices, x.indptr, prob_val, prob_ind, prob.rowptr)

class problem(Structure):
	_names = ["l", "n", "y", "x", "bias"]
	_types = [c_int, c_int, POINTER(c_double), POINTER(POINTER(feature_node)), c_double]
	_fields_ = genFields(_names, _types)

	def __init__(self, y, x, bias = -1):
		if (not isinstance(y, (list, tuple))) and (not (scipy and isinstance(y, scipy.ndarray))):
			raise TypeError("type of y: {0} is not supported!".format(type(y)))

		if isinstance(x, (list, tuple)):
			if len(y) != len(x):
				raise ValueError("len(y) != len(x)")
		elif scipy != None and isinstance(x, (scipy.ndarray, sparse.spmatrix)):
			if len(y) != x.shape[0]:
				raise ValueError("len(y) != len(x)")
			if isinstance(x, scipy.ndarray):
				x = scipy.ascontiguousarray(x) # enforce row-major
			if isinstance(x, sparse.spmatrix):
				x = x.tocsr()
				pass
		else:
			raise TypeError("type of x: {0} is not supported!".format(type(x)))
		self.l = l = len(y)
		self.bias = -1

		max_idx = 0
		x_space = self.x_space = []
		if scipy != None and isinstance(x, sparse.csr_matrix):
			csr_to_problem(x, self)
			max_idx = x.shape[1]
		else:
			for i, xi in enumerate(x):
				tmp_xi, tmp_idx = gen_feature_nodearray(xi)
				x_space += [tmp_xi]
				max_idx = max(max_idx, tmp_idx)
		self.n = max_idx

		self.y = (c_double * l)()
		if scipy != None and isinstance(y, scipy.ndarray):
			scipy.ctypeslib.as_array(self.y, (self.l,))[:] = y
		else:
			for i, yi in enumerate(y): self.y[i] = yi

		self.x = (POINTER(feature_node) * l)()
		if scipy != None and isinstance(x, sparse.csr_matrix):
			base = addressof(self.x_space.ctypes.data_as(POINTER(feature_node))[0])
			x_ptr = cast(self.x, POINTER(c_uint64))
			x_ptr = scipy.ctypeslib.as_array(x_ptr,(self.l,))
			x_ptr[:] = self.rowptr[:-1]*sizeof(feature_node)+base
		else:
			for i, xi in enumerate(self.x_space): self.x[i] = xi

		self.set_bias(bias)

	def set_bias(self, bias):
		if self.bias == bias:
			return
		if bias >= 0 and self.bias < 0:
			self.n += 1
			node = feature_node(self.n, bias)
		if bias < 0 and self.bias >= 0:
			self.n -= 1
			node = feature_node(-1, bias)

		if isinstance(self.x_space, list):
			for xi in self.x_space:
				xi[-2] = node
		else:
			self.x_space["index"][self.rowptr[1:]-2] = node.index
			self.x_space["value"][self.rowptr[1:]-2] = node.value

		self.bias = bias


class parameter(Structure):
	_names = ["solver_type", "eps", "C", "nr_weight", "weight_label", "weight", "p", "init_sol"]
	_types = [c_int, c_double, c_double, c_int, POINTER(c_int), POINTER(c_double), c_double, POINTER(c_double)]
	_fields_ = genFields(_names, _types)

	def __init__(self, options = None):
		if options == None:
			options = ''
		self.parse_options(options)

	def __str__(self):
		s = ''
		attrs = parameter._names + list(self.__dict__.keys())
		values = map(lambda attr: getattr(self, attr), attrs)
		for attr, val in zip(attrs, values):
			s += (' %s: %s\n' % (attr, val))
		s = s.strip()

		return s

	def set_to_default_values(self):
		self.solver_type = L2R_L2LOSS_SVC_DUAL
		self.eps = float('inf')
		self.C = 1
		self.p = 0.1
		self.nr_weight = 0
		self.weight_label = None
		self.weight = None
		self.init_sol = None
		self.bias = -1
		self.flag_cross_validation = False
		self.flag_C_specified = False
		self.flag_solver_specified = False
		self.flag_find_C = False
		self.nr_fold = 0
		self.print_func = cast(None, PRINT_STRING_FUN)

	def parse_options(self, options):
		if isinstance(options, list):
			argv = options
		elif isinstance(options, str):
			argv = options.split()
		else:
			raise TypeError("arg 1 should be a list or a str.")
		self.set_to_default_values()
		self.print_func = cast(None, PRINT_STRING_FUN)
		weight_label = []
		weight = []

		i = 0
		while i < len(argv) :
			if argv[i] == "-s":
				i = i + 1
				self.solver_type = int(argv[i])
				self.flag_solver_specified = True
			elif argv[i] == "-c":
				i = i + 1
				self.C = float(argv[i])
				self.flag_C_specified = True
			elif argv[i] == "-p":
				i = i + 1
				self.p = float(argv[i])
			elif argv[i] == "-e":
				i = i + 1
				self.eps = float(argv[i])
			elif argv[i] == "-B":
				i = i + 1
				self.bias = float(argv[i])
			elif argv[i] == "-v":
				i = i + 1
				self.flag_cross_validation = 1
				self.nr_fold = int(argv[i])
				if self.nr_fold < 2 :
					raise ValueError("n-fold cross validation: n must >= 2")
			elif argv[i].startswith("-w"):
				i = i + 1
				self.nr_weight += 1
				weight_label += [int(argv[i-1][2:])]
				weight += [float(argv[i])]
			elif argv[i] == "-q":
				self.print_func = PRINT_STRING_FUN(print_null)
			elif argv[i] == "-C":
				self.flag_find_C = True

			else:
				raise ValueError("Wrong options")
			i += 1

		liblinear.set_print_string_function(self.print_func)
		self.weight_label = (c_int*self.nr_weight)()
		self.weight = (c_double*self.nr_weight)()
		for i in range(self.nr_weight):
			self.weight[i] = weight[i]
			self.weight_label[i] = weight_label[i]

		# default solver for parameter selection is L2R_L2LOSS_SVC
		if self.flag_find_C:
			if not self.flag_cross_validation:
				self.nr_fold = 5
			if not self.flag_solver_specified:
				self.solver_type = L2R_L2LOSS_SVC
				self.flag_solver_specified = True
			elif self.solver_type not in [L2R_LR, L2R_L2LOSS_SVC]:
				raise ValueError("Warm-start parameter search only available for -s 0 and -s 2")

		if self.eps == float('inf'):
			if self.solver_type in [L2R_LR, L2R_L2LOSS_SVC]:
				self.eps = 0.01
			elif self.solver_type in [L2R_L2LOSS_SVR]:
				self.eps = 0.001
			elif self.solver_type in [L2R_L2LOSS_SVC_DUAL, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L2R_LR_DUAL]:
				self.eps = 0.1
			elif self.solver_type in [L1R_L2LOSS_SVC, L1R_LR]:
				self.eps = 0.01
			elif self.solver_type in [L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL]:
				self.eps = 0.1

class model(Structure):
	_names = ["param", "nr_class", "nr_feature", "w", "label", "bias"]
	_types = [parameter, c_int, c_int, POINTER(c_double), POINTER(c_int), c_double]
	_fields_ = genFields(_names, _types)

	def __init__(self):
		self.__createfrom__ = 'python'

	def __del__(self):
		# free memory created by C to avoid memory leak
		if hasattr(self, '__createfrom__') and self.__createfrom__ == 'C':
			liblinear.free_and_destroy_model(pointer(self))

	def get_nr_feature(self):
		return liblinear.get_nr_feature(self)

	def get_nr_class(self):
		return liblinear.get_nr_class(self)

	def get_labels(self):
		nr_class = self.get_nr_class()
		labels = (c_int * nr_class)()
		liblinear.get_labels(self, labels)
		return labels[:nr_class]

	def get_decfun_coef(self, feat_idx, label_idx=0):
		return liblinear.get_decfun_coef(self, feat_idx, label_idx)

	def get_decfun_bias(self, label_idx=0):
		return liblinear.get_decfun_bias(self, label_idx)

	def get_decfun(self, label_idx=0):
		w = [liblinear.get_decfun_coef(self, feat_idx, label_idx) for feat_idx in range(1, self.nr_feature+1)]
		b = liblinear.get_decfun_bias(self, label_idx)
		return (w, b)

	def is_probability_model(self):
		return (liblinear.check_probability_model(self) == 1)

	def is_regression_model(self):
		return (liblinear.check_regression_model(self) == 1)

def toPyModel(model_ptr):
	"""
	toPyModel(model_ptr) -> model

	Convert a ctypes POINTER(model) to a Python model
	"""
	if bool(model_ptr) == False:
		raise ValueError("Null pointer")
	m = model_ptr.contents
	m.__createfrom__ = 'C'
	return m

fillprototype(liblinear.train, POINTER(model), [POINTER(problem), POINTER(parameter)])
fillprototype(liblinear.find_parameter_C, None, [POINTER(problem), POINTER(parameter), c_int, c_double, c_double, POINTER(c_double), POINTER(c_double)])
fillprototype(liblinear.cross_validation, None, [POINTER(problem), POINTER(parameter), c_int, POINTER(c_double)])

fillprototype(liblinear.predict_values, c_double, [POINTER(model), POINTER(feature_node), POINTER(c_double)])
fillprototype(liblinear.predict, c_double, [POINTER(model), POINTER(feature_node)])
fillprototype(liblinear.predict_probability, c_double, [POINTER(model), POINTER(feature_node), POINTER(c_double)])

fillprototype(liblinear.save_model, c_int, [c_char_p, POINTER(model)])
fillprototype(liblinear.load_model, POINTER(model), [c_char_p])

fillprototype(liblinear.get_nr_feature, c_int, [POINTER(model)])
fillprototype(liblinear.get_nr_class, c_int, [POINTER(model)])
fillprototype(liblinear.get_labels, None, [POINTER(model), POINTER(c_int)])
fillprototype(liblinear.get_decfun_coef, c_double, [POINTER(model), c_int, c_int])
fillprototype(liblinear.get_decfun_bias, c_double, [POINTER(model), c_int])

fillprototype(liblinear.free_model_content, None, [POINTER(model)])
fillprototype(liblinear.free_and_destroy_model, None, [POINTER(POINTER(model))])
fillprototype(liblinear.destroy_param, None, [POINTER(parameter)])
fillprototype(liblinear.check_parameter, c_char_p, [POINTER(problem), POINTER(parameter)])
fillprototype(liblinear.check_probability_model, c_int, [POINTER(model)])
fillprototype(liblinear.check_regression_model, c_int, [POINTER(model)])
fillprototype(liblinear.set_print_string_function, None, [CFUNCTYPE(None, c_char_p)])


def svm_read_problem(data_file_name, return_scipy=False):
	"""
	svm_read_problem(data_file_name, return_scipy=False) -> [y, x], y: list, x: list of dictionary
	svm_read_problem(data_file_name, return_scipy=True)  -> [y, x], y: ndarray, x: csr_matrix

	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	prob_y = []
	prob_x = []
	row_ptr = [0]
	col_idx = []
	for i, line in enumerate(open(data_file_name)):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		prob_y += [float(label)]
		if scipy != None and return_scipy:
			nz = 0
			for e in features.split():
				ind, val = e.split(":")
				val = float(val)
				if val != 0:
					col_idx += [int(ind)-1]
					prob_x += [val]
					nz += 1
			row_ptr += [row_ptr[-1]+nz]
		else:
			xi = {}
			for e in features.split():
				ind, val = e.split(":")
				if val != 0:
					xi[int(ind)] = float(val)
			prob_x += [xi]
	if scipy != None and return_scipy:
		prob_y = scipy.array(prob_y)
		prob_x = scipy.array(prob_x)
		col_idx = scipy.array(col_idx)
		row_ptr = scipy.array(row_ptr)
		prob_x = sparse.csr_matrix((prob_x, col_idx, row_ptr))
	return (prob_y, prob_x)

def load_model(model_file_name):
	"""
	load_model(model_file_name) -> model

	Load a LIBLINEAR model from model_file_name and return.
	"""
	model = liblinear.load_model(model_file_name.encode())
	if not model:
		print("can't open model file %s" % model_file_name)
		return None
	model = toPyModel(model)
	return model

def save_model(model_file_name, model):
	"""
	save_model(model_file_name, model) -> None

	Save a LIBLINEAR model to the file model_file_name.
	"""
	liblinear.save_model(model_file_name.encode(), model)

def evaluations_scipy(ty, pv):
	"""
	evaluations_scipy(ty, pv) -> (ACC, MSE, SCC)
	ty, pv: ndarray

	Calculate accuracy, mean squared error and squared correlation coefficient
	using the true values (ty) and predicted values (pv).
	"""
	if not (scipy != None and isinstance(ty, scipy.ndarray) and isinstance(pv, scipy.ndarray)):
		raise TypeError("type of ty and pv must be ndarray")
	if len(ty) != len(pv):
		raise ValueError("len(ty) must be equal to len(pv)")
	ACC = 100.0*(ty == pv).mean()
	MSE = ((ty - pv)**2).mean()
	l = len(ty)
	sumv = pv.sum()
	sumy = ty.sum()
	sumvy = (pv*ty).sum()
	sumvv = (pv*pv).sum()
	sumyy = (ty*ty).sum()
	with scipy.errstate(all = 'raise'):
		try:
			SCC = ((l*sumvy-sumv*sumy)*(l*sumvy-sumv*sumy))/((l*sumvv-sumv*sumv)*(l*sumyy-sumy*sumy))
		except:
			SCC = float('nan')
	return (float(ACC), float(MSE), float(SCC))

def evaluations(ty, pv, useScipy = True):
	"""
	evaluations(ty, pv, useScipy) -> (ACC, MSE, SCC)
	ty, pv: list, tuple or ndarray
	useScipy: convert ty, pv to ndarray, and use scipy functions for the evaluation

	Calculate accuracy, mean squared error and squared correlation coefficient
	using the true values (ty) and predicted values (pv).
	"""
	if scipy != None and useScipy:
		return evaluations_scipy(scipy.asarray(ty), scipy.asarray(pv))
	if len(ty) != len(pv):
		raise ValueError("len(ty) must be equal to len(pv)")
	total_correct = total_error = 0
	sumv = sumy = sumvv = sumyy = sumvy = 0
	for v, y in zip(pv, ty):
		if y == v:
			total_correct += 1
		total_error += (v-y)*(v-y)
		sumv += v
		sumy += y
		sumvv += v*v
		sumyy += y*y
		sumvy += v*y
	l = len(ty)
	ACC = 100.0*total_correct/l
	MSE = total_error/l
	try:
		SCC = ((l*sumvy-sumv*sumy)*(l*sumvy-sumv*sumy))/((l*sumvv-sumv*sumv)*(l*sumyy-sumy*sumy))
	except:
		SCC = float('nan')
	return (float(ACC), float(MSE), float(SCC))

def train(arg1, arg2=None, arg3=None):
	"""
	train(y, x [, options]) -> model | ACC

	y: a list/tuple/ndarray of l true labels (type must be int/double).

	x: 1. a list/tuple of l training instances. Feature vector of
	      each training instance is a list/tuple or dictionary.

	   2. an l * n numpy ndarray or scipy spmatrix (n: number of features).

	train(prob [, options]) -> model | ACC
	train(prob, param) -> model | ACC

	Train a model from data (y, x) or a problem prob using
	'options' or a parameter param.

	If '-v' is specified in 'options' (i.e., cross validation)
	either accuracy (ACC) or mean-squared error (MSE) is returned.

	options:
		-s type : set type of solver (default 1)
		  for multi-class classification
			 0 -- L2-regularized logistic regression (primal)
			 1 -- L2-regularized L2-loss support vector classification (dual)
			 2 -- L2-regularized L2-loss support vector classification (primal)
			 3 -- L2-regularized L1-loss support vector classification (dual)
			 4 -- support vector classification by Crammer and Singer
			 5 -- L1-regularized L2-loss support vector classification
			 6 -- L1-regularized logistic regression
			 7 -- L2-regularized logistic regression (dual)
		  for regression
			11 -- L2-regularized L2-loss support vector regression (primal)
			12 -- L2-regularized L2-loss support vector regression (dual)
			13 -- L2-regularized L1-loss support vector regression (dual)
		-c cost : set the parameter C (default 1)
		-p epsilon : set the epsilon in loss function of SVR (default 0.1)
		-e epsilon : set tolerance of termination criterion
			-s 0 and 2
				|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
				where f is the primal function, (default 0.01)
			-s 11
				|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)
			-s 1, 3, 4, and 7
				Dual maximal violation <= eps; similar to liblinear (default 0.)
			-s 5 and 6
				|f'(w)|_inf <= eps*min(pos,neg)/l*|f'(w0)|_inf,
				where f is the primal function (default 0.01)
			-s 12 and 13
				|f'(alpha)|_1 <= eps |f'(alpha0)|,
				where f is the dual function (default 0.1)
		-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
		-wi weight: weights adjust the parameter C of different classes (see README for details)
		-v n: n-fold cross validation mode
		-q : quiet mode (no outputs)
	"""
	prob, param = None, None
	if isinstance(arg1, (list, tuple)) or (scipy and isinstance(arg1, scipy.ndarray)):
		assert isinstance(arg2, (list, tuple)) or (scipy and isinstance(arg2, (scipy.ndarray, sparse.spmatrix)))
		y, x, options = arg1, arg2, arg3
		prob = problem(y, x)
		param = parameter(options)
	elif isinstance(arg1, problem):
		prob = arg1
		if isinstance(arg2, parameter):
			param = arg2
		else:
			param = parameter(arg2)
	if prob == None or param == None :
		raise TypeError("Wrong types for the arguments")

	prob.set_bias(param.bias)
	liblinear.set_print_string_function(param.print_func)
	err_msg = liblinear.check_parameter(prob, param)
	if err_msg :
		raise ValueError('Error: %s' % err_msg)

	if param.flag_find_C:
		nr_fold = param.nr_fold
		best_C = c_double()
		best_rate = c_double()
		max_C = 1024
		if param.flag_C_specified:
			start_C = param.C
		else:
			start_C = -1.0
		liblinear.find_parameter_C(prob, param, nr_fold, start_C, max_C, best_C, best_rate)
		print("Best C = %lf  CV accuracy = %g%%\n"% (best_C.value, 100.0*best_rate.value))
		return best_C.value,best_rate.value


	elif param.flag_cross_validation:
		l, nr_fold = prob.l, param.nr_fold
		target = (c_double * l)()
		liblinear.cross_validation(prob, param, nr_fold, target)
		ACC, MSE, SCC = evaluations(prob.y[:l], target[:l])
		if param.solver_type in [L2R_L2LOSS_SVR, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL]:
			print("Cross Validation Mean squared error = %g" % MSE)
			print("Cross Validation Squared correlation coefficient = %g" % SCC)
			return MSE
		else:
			print("Cross Validation Accuracy = %g%%" % ACC)
			return ACC
	else:
		m = liblinear.train(prob, param)
		m = toPyModel(m)

		return m

def predict(y, x, m, options=""):
	"""
	predict(y, x, m [, options]) -> (p_labels, p_acc, p_vals)

	y: a list/tuple/ndarray of l true labels (type must be int/double).
	   It is used for calculating the accuracy. Use [] if true labels are
	   unavailable.

	x: 1. a list/tuple of l training instances. Feature vector of
	      each training instance is a list/tuple or dictionary.

	   2. an l * n numpy ndarray or scipy spmatrix (n: number of features).

	Predict data (y, x) with the SVM model m.
	options:
	    -b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only
	    -q quiet mode (no outputs)

	The return tuple contains
	p_labels: a list of predicted labels
	p_acc: a tuple including  accuracy (for classification), mean-squared
	       error, and squared correlation coefficient (for regression).
	p_vals: a list of decision values or probability estimates (if '-b 1'
	        is specified). If k is the number of classes, for decision values,
	        each element includes results of predicting k binary-class
	        SVMs. if k = 2 and solver is not MCSVM_CS, only one decision value
	        is returned. For probabilities, each element contains k values
	        indicating the probability that the testing instance is in each class.
	        Note that the order of classes here is the same as 'model.label'
	        field in the model structure.
	"""

	def info(s):
		print(s)

	if scipy and isinstance(x, scipy.ndarray):
		x = scipy.ascontiguousarray(x) # enforce row-major
	elif sparse and isinstance(x, sparse.spmatrix):
		x = x.tocsr()
	elif not isinstance(x, (list, tuple)):
		raise TypeError("type of x: {0} is not supported!".format(type(x)))

	if (not isinstance(y, (list, tuple))) and (not (scipy and isinstance(y, scipy.ndarray))):
		raise TypeError("type of y: {0} is not supported!".format(type(y)))

	predict_probability = 0
	argv = options.split()
	i = 0
	while i < len(argv):
		if argv[i] == '-b':
			i += 1
			predict_probability = int(argv[i])
		elif argv[i] == '-q':
			info = print_null
		else:
			raise ValueError("Wrong options")
		i+=1

	solver_type = m.param.solver_type
	nr_class = m.get_nr_class()
	nr_feature = m.get_nr_feature()
	is_prob_model = m.is_probability_model()
	bias = m.bias
	if bias >= 0:
		biasterm = feature_node(nr_feature+1, bias)
	else:
		biasterm = feature_node(-1, bias)
	pred_labels = []
	pred_values = []

	if scipy and isinstance(x, sparse.spmatrix):
		nr_instance = x.shape[0]
	else:
		nr_instance = len(x)

	if predict_probability:
		if not is_prob_model:
			raise TypeError('probability output is only supported for logistic regression')
		prob_estimates = (c_double * nr_class)()
		for i in range(nr_instance):
			if scipy and isinstance(x, sparse.spmatrix):
				indslice = slice(x.indptr[i], x.indptr[i+1])
				xi, idx = gen_feature_nodearray((x.indices[indslice], x.data[indslice]), feature_max=nr_feature)
			else:
				xi, idx = gen_feature_nodearray(x[i], feature_max=nr_feature)
			xi[-2] = biasterm
			label = liblinear.predict_probability(m, xi, prob_estimates)
			values = prob_estimates[:nr_class]
			pred_labels += [label]
			pred_values += [values]
	else:
		if nr_class <= 2:
			nr_classifier = 1
		else:
			nr_classifier = nr_class
		dec_values = (c_double * nr_classifier)()
		for i in range(nr_instance):
			if scipy and isinstance(x, sparse.spmatrix):
				indslice = slice(x.indptr[i], x.indptr[i+1])
				xi, idx = gen_feature_nodearray((x.indices[indslice], x.data[indslice]), feature_max=nr_feature)
			else:
				xi, idx = gen_feature_nodearray(x[i], feature_max=nr_feature)
			xi[-2] = biasterm
			label = liblinear.predict_values(m, xi, dec_values)
			values = dec_values[:nr_classifier]
			pred_labels += [label]
			pred_values += [values]

	if len(y) == 0:
		y = [0] * nr_instance
	ACC, MSE, SCC = evaluations(y, pred_labels)

	if m.is_regression_model():
		info("Mean squared error = %g (regression)" % MSE)
		info("Squared correlation coefficient = %g (regression)" % SCC)
	else:
		info("Accuracy = %g%% (%d/%d) (classification)" % (ACC, int(round(nr_instance*ACC/100)), nr_instance))

	return pred_labels, (ACC, MSE, SCC), pred_values
