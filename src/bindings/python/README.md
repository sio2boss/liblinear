liblinear
=========

Python bindings for liblinear.  You must have the liblinear shared library on your LD_LIBRARY_PATH

# Usage

```
import scipy
from liblinear import *

y, x = svm_read_problem('../heart_scale', return_scipy = True) # y: ndarray, x: csr_matrix

```