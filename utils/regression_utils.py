# helper functions

# y = Xb
# b = ((X'X)^-1)X'y
def get_regression_coef(X, y):
    import numpy as np

    XtX = np.matmul(X.transpose(), X)
    XtXinv = np.linalg.inv(XtX)
    XtXinvXt = np.matmul(XtXinv, X.transpose())

    b = np.matmul(XtXinvXt, y)

    return b
