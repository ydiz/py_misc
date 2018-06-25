import numpy as np
from .core import *

pauli1 = np.array([[0, -1], [1,0]], dtype=np.complex128) # 2i * generator
pauli2 = np.array([[0, 1j], [1j,0]], dtype=np.complex128)
pauli3 = np.array([[1j, 0], [0, -1j]], dtype=np.complex128)
# no difference
# pauli1 = np.array([[0, 1j], [1j,0]], dtype=np.complex128)
# pauli2 = np.array([[0, 1], [-1,0]], dtype=np.complex128)
# pauli3 = np.array([[1j, 0], [0, -1j]], dtype=np.complex128)

def get_masks(grid):
    a = range(grid[0])
    coor = np.array(np.meshgrid(a,a,a,a, indexing='ij'))
    coor = np.moveaxis(coor, 0, -1)
    mask0 = np.sum(coor, axis=-1) % 2 == 0
    mask1 = np.sum(coor, axis=-1) % 2 == 1
    return (mask0, mask1)

#    static void su2SubGroupIndex(int &i1, int &i2, int su2_index)
#  0 -> 0,1; 1-> 0,2; 2 -> 1,2
def su2SubGroupIndex(su2_index):
    spare = su2_index
    i1 = 0
    while spare >= 2 - i1:
        spare = spare - 2 + i1
        i1 += 1
    i2 = i1 + 1 + spare
    return i1, i2

#still need antisymmetrization
def su2Extract(U, i1, i2):
    index = np.ix_([i1,i2],[i1,i2])
    return U[..., index[0], index[1]]

def su2Insert(newSU2, U, i1, i2):
    U[..., range(3), range(3)] = 1.0
    index = np.ix_([i1,i2],[i1,i2])
    U[..., index[0], index[1]] = newSU2

def random_three_vector(mag, shape):
    phi = np.random.uniform(0, 2.0 * np.pi, shape)
    cos_theta = np.random.uniform(-1, 1, shape)
    # phi = np.full(shape, 0.5) # for test
    # cos_theta = np.full(shape, 0.5)

    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    x = mag * sin_theta * np.cos( phi )
    y = mag * sin_theta * np.sin( phi )
    z = mag * cos_theta
    return (x,y,z)

# need grid
def subGroupHeatBath(g, bare_staple, betaMM, su2_index, mask, nheatbath=5):

    grid = g.shape[:4]
    staple = bare_staple * betaMM
    V = np.matmul(g, staple)

    sigma = su2Extract(V, *su2SubGroupIndex(su2_index))
    sigmaH = adj(sigma)
    u = sigma - sigmaH + trace(sigmaH)[..., None, None] * np.eye(2)
    udet = np.linalg.det(u)
    u = np.where((np.absolute(np.real(udet)) > 1e-7)[...,None,None], u, np.eye(2)) #zyd: those two checks can be removed?
    udet = np.where(np.absolute(np.real(udet)) > 1e-7, udet, 1)

    xi = 0.5 * np.sqrt(udet) #FIXME: udet is real; here can be simplified
    u = 0.5 * u / xi[..., None, None]
    alpha = np.real(2 * xi)

    numSites = np.sum(mask)
    hit = 0
    d = np.zeros(grid)
    Accepted = np.zeros(grid, dtype=bool)

    while np.sum(Accepted) < numSites and hit < nheatbath:
        R0 = np.random.rand(*grid)
        R1 = -np.log(np.random.rand(*grid))/alpha
        R2 = -np.log(np.random.rand(*grid))/alpha
        R3 = np.cos(np.random.rand(*grid) * 2.0 * np.pi)
        # R0 = np.full(grid, 0.5)
        # R1 = -np.log(np.full(grid, 0.5))/alpha
        # R2 = -np.log(np.full(grid, 0.5))/alpha
        # R3 = np.cos(np.full(grid, 0.5) * 2.0 * np.pi)
        R3 = R3 * R3

        d = np.where(Accepted, d, R2 + R1 * R3)
        thresh = 1.0 - 0.5 * d

        newlyAccepted = R0 * R0 < thresh#where(R0 * R0 < thresh, 1, 0)
        Accepted = np.where(newlyAccepted, newlyAccepted, Accepted)
        Accepted = np.where(mask, Accepted, 0)
        hit += 1

    a0 = np.where(mask, 1.0 - d, 0)
    a123mag = np.sqrt(np.absolute(1 - a0*a0)) #?abs is not necessary
    a1, a2, a3 = random_three_vector(a123mag, grid)
    ua = a0[..., None, None] * np.eye(2) + a1[..., None, None] * pauli1 \
        + a2[..., None, None] * pauli2 + a3[..., None, None] * pauli3

    b = np.where(mask[..., None, None], np.matmul(adj(u), ua), np.eye(2))
    L = np.zeros_like(g)
    su2Insert(b, L, *su2SubGroupIndex(su2_index))

    g = np.where(Accepted[..., None, None], np.matmul(L,g), g)
    return g


def GF_heatbath(U, g, betaMM, nsweeps, multi_hit):
    masks = get_masks(g.shape[:4])
    for _ in range(nsweeps):
        # print(Omega_g(U, g))
        for cb in range(2):
            bare_staple = staple_cal_GF(U, g)
            for su2_index in range(3):
                g = subGroupHeatBath(g, bare_staple, betaMM, su2_index, masks[cb], multi_hit)
            g = projectOnGroup(g)
    return g
