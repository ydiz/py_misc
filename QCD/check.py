def checkUnitary(U):
    return np.sum(np.absolute(np.matmul(U, adj(U))- np.eye(U.shape[-1])))
def checkAntiHermitian(P):
    return np.sum(np.absolute(P + adj(P)))
