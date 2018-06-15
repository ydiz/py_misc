import numpy as np
import re


def adj(a):
    return np.conj(np.moveaxis(a, -2, -1)) # transpose doesn't work; so used moveaxis
def trace(a):
    return np.trace(a, axis1=-2, axis2=-1)
def Omega(U):
    return np.sum(trace(U))

def readHeader(filename):
    with open(filename, "rb") as f:
        header = b''
        for _ in range(26):
            header += f.readline()
        return header, len(header)

def readField(filename):
    header, offset = readHeader(filename)
    dims = [int(x) for x in re.findall(b'DIMENSION_\d = (\d+)', header)]
    print("dims: ", dims)
    with open(filename, "rb") as f:
        f.seek(offset) # offset can be seen by f.seek(0)  f.read(562)
        data = np.fromfile(f, dtype='>f8').reshape((-1,2)) # reshape to complex number
    data_cmpl = np.apply_along_axis(lambda x: complex(*x), 1, data)
    data_lat = data_cmpl.reshape(*dims, 4, 3, 3)
    data_lat = data_lat.swapaxes(0,3).swapaxes(1,2)  # [x1,x2,x3,x4] -> [x4, x3, x2, x1]
    return data_lat

# after peekLorentz
def plaq_cal(U):
    plaqs = 0
    dim = U.shape[1]
    for mu in range(1, 4):
        for nu in range(mu):
            plaqs += np.sum(trace(   np.matmul(np.matmul(np.matmul(U[mu], np.roll(U[nu], -1, mu)), \
                                adj(np.roll(U[mu], -1, nu))), adj(U[nu])) ))
    return plaqs.real / 6.0 / dim**4 /3.0

# after peekLorentz
def staple_cal_GF(U, g):
    staple = np.zeros_like(g, dtype=np.complex128)
    for mu in range(4):
        staple += np.matmul( U[mu], adj(np.roll(g, -1, mu)) ) \
                    + adj( np.matmul(np.roll(g, 1, mu), np.roll(U[mu],1,mu)))
    return staple

# after peekLorentz
def staple_cal_U(U, mu):
    staple = np.zeros(U.shape[1:], dtype=np.complex128)
    for nu in range(4):
        if nu != mu:
            staple += np.matmul(np.matmul(np.roll(U[nu], -1, mu), adj(np.roll(U[mu], -1, nu))), adj(U[nu]))
            staple += np.matmul(np.matmul(adj(np.roll(np.roll(U[nu],-1,mu), 1, nu)), adj(np.roll(U[mu], 1, nu))), np.roll(U[nu], 1, nu))
    return staple

#only for 3x3 matrix
def Ta(U):
    ret = (U - adj(U)) * 0.5
    ret = ret - trace(ret)[..., None, None] / 3.0 * np.eye(3)
    return ret

# after peekLorentz
def Omega_g(U, g):
    s = np.zeros_like(g)
    for mu in range(4):
        s += np.matmul(U[mu], adj(np.roll(g, -1, mu)))
    return np.sum(np.real(trace(  np.matmul(g, s) )))

def projectOnGroup(m):
    shape = m.shape[:-2]
    for c1 in range(3):
        inner = np.zeros(shape, dtype=np.complex128)
        for c2 in range(3):
            inner += np.conjugate(m[..., c1, c2]) * m[..., c1, c2]
        nrm = np.sqrt(inner)
        for c2 in range(3):
            m[..., c1, c2] /= nrm
        for b in range(c1 + 1, 3):
            pr = np.zeros(shape, dtype=np.complex128)
            for c in range(3):
                pr += np.conjugate(m[..., c1, c]) * m[..., b,c]
            for c in range(3):
                m[..., b,c] -= pr * m[...,c1,c]
    return m

def generator_su3():
    ta = np.empty((8,3,3), dtype=np.complex128)
    ta[0] = np.array([[0,1j,0],[-1j,0,0],[0,0,0]])
    ta[1] = np.array([[0,1,0],[1,0,0],[0,0,0]])
    ta[2] = np.array([[0,0,1j],[0,0,0],[-1j,0,0]])
    ta[3] = np.array([[0,0,1],[0,0,0],[1,0,0]])
    ta[4] = np.array([[0,0,0],[0,0,1j],[0,-1j,0]])
    ta[5] = np.array([[0,0,0],[0,0,1],[0,1,0]])
    ta[6] = np.array([[1,0,0],[0,-1,0],[0,0,0]])
    ta[7] = np.array([[1,0,0],[0,1,0],[0,0,-2]])
    ta = ta * 0.5
    ta[7] = ta[7] / np.sqrt(3)
    return ta

def generate_P(grid):
    ta = generator_su3() #FIXME ta can be defined as global
    ta = 1j * ta # P is actually 1j * P
    P = np.zeros((4,)+grid + (3,3), dtype=np.complex128)
    for i in range(8):
        P += np.random.normal(size=(4,)+grid)[..., None, None] * ta[i]
        # P += np.full((4,)+grid, 0.5)[..., None, None] * ta[i] # for test
    return P

def coldConfiguration(shape):
    U = np.zeros(shape, dtype=np.complex128)
    U[..., range(3), range(3)] = 1.0
    return U

# arg * alpha must be traceless and anti-hermitian
def exp_traceless_antiHermi(arg, alpha):# exp(arg * alphs)
    arg = arg * alpha
    iQ2 = np.matmul(arg, arg)
    iQ3 = np.matmul(arg, iQ2)
    c0 = - np.imag(trace(iQ3)) / 3.0
    c1 = - np.real(trace(iQ2)) / 2.0

    tmp = c1 / 3.0 # can remove variable c1
    c0max = 2.0 * tmp**1.5
    theta = np.arccos(c0 / c0max) / 3.0
    u = np.sqrt(tmp) * np.cos(theta)
    w = np.sqrt(c1) * np.sin(theta)

    ixi0 = np.sin(w) / w * 1j
    u2 = u * u
    w2 = w * w
    cosw = np.cos(w)

    emiu = np.cos(u) - 1j * np.sin(u)
    e2iu = np.cos(2.0 * u) + 1j * np.sin(2.0 * u) # cos 2u and sin2u can be calculated analytically
    h0 = e2iu * (u2 - w2) + emiu * (8.0 * u2 * cosw + 2.0 * u * (3.0 * u2 + w2) * ixi0)
    h1 = e2iu * (2.0 * u) - emiu * ((2.0 * u * cosw) - (3.0 * u2 - w2) * ixi0)
    h2 = e2iu - emiu * (cosw + 3.0 * u * ixi0)

    fden = 1.0 / (9.0 * u2 - w2)
    f0 = h0 * fden
    f1 = h1 * fden
    f2 = h2 * fden

    return f0[..., None, None] * np.eye(3) - 1j  * f1[..., None, None] * arg - f2[..., None, None] * iQ2
