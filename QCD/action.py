from .core import *
from .heatbath import *

def dOmegadU_g(U, g):
    ret = np.empty_like(U)
    for mu in range(4):
        ret[mu] = np.matmul(np.matmul(U[mu], adj(np.roll(g, -1, mu))), g)
    ret = Ta(ret)
    return ret

class GFAction:
    def __init__(self, beta, M, innerMC_N_force=100, innerMC_N_H=100, hb_offset=100, hb_multi_hit=10, log_file="log.txt"):
        self.beta = beta
        self.M = M
        self.betaMM = beta * M * M
        self.innerMC_N_force = innerMC_N_force
        self.innerMC_N_H = innerMC_N_H
        self.hb_offset = hb_offset
        self.hb_multi_hit = hb_multi_hit
        self.log_file = log_file
        self.name = "GFAction"
    def S(self, U):
        Sw = WilsonAction(self.beta).S(U)
        SGF1 = - self.betaMM * np.sum(np.real(trace(U)))
        return SGF1 + Sw
    def deriv(self, U):
        dSwdU = WilsonAction(self.beta).deriv(U)
        factor = 0.5 * self.betaMM
        dSGF1dU = factor * Ta(U)

        dSGF2dU = np.zeros_like(U)
        g = coldConfiguration(U.shape[1:])
        g = GF_heatbath(U, g, self.betaMM, self.hb_offset, self.hb_multi_hit)

        for _ in range(self.innerMC_N_force):
            dSGF2dU += dOmegadU_g(U, g)
            g = GF_heatbath(U, g, self.betaMM, 1, self.hb_multi_hit)
        dSGF2dU = factor / self.innerMC_N_force * dSGF2dU

        return dSwdU + dSGF1dU - dSGF2dU


class WilsonAction:
    def __init__(self, beta):
        self.beta = beta
        self.name = "WilsonAction"
    def S(self, U): # after peekLorentz
        return self.beta * (1 - plaq_cal(U)) * 6 * U.shape[1]**4
    def deriv(self, U): # after peekLorentz; return also after peekLorentz
        factor = 0.5 * self.beta / 3.0
        dSdU = np.empty_like(U)
        for mu in range(4):
            dSdU[mu] = factor * Ta(np.matmul(U[mu], staple_cal_U(U, mu)))
        return dSdU
