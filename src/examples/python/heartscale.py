import scipy
from liblinear import *
# Read data in LIBSVM format
y, x = svm_read_problem('../heart_scale', return_scipy = True) # y: ndarray, x: csr_matrix
m = train(y[:200], x[:200, :], '-c 4')
p_label, p_acc, p_val = predict(y[200:], x[200:, :], m)

# Construct problem in Scipy format
# Dense data: numpy ndarray
y, x = scipy.asarray([1,-1]), scipy.asarray([[1,0,1], [-1,0,-1]])
# Sparse data: scipy csr_matrix((data, (row_ind, col_ind))
y, x = scipy.asarray([1,-1]), scipy.sparse.csr_matrix(([1, 1, -1, -1], ([0, 0, 1, 1], [0, 2, 0, 2])))
prob  = problem(y, x)
param = parameter('-s 0 -c 4 -B 1')
m = train(prob, param)

# Other utility functions
save_model('heart_scale.model', m)
m = load_model('heart_scale.model')
p_label, p_acc, p_val = predict(y, x, m, '-b 1')
ACC, MSE, SCC = evaluations(y, p_label)

# Getting online help
help(train)