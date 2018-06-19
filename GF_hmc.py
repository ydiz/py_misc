import numpy as np
from QCD import *

def log_name(M, trajL, innerMC_N_force, innerMC_N_H):
    return "./data6.19/" + str(M) + "_" + str(trajL) + "_" + str(innerMC_N_force) + "_" + str(innerMC_N_H) + ".txt"


grid = (4,4,4,4)
# U = coldConfiguration((4,)+grid+(3,3))
Umu = readField("U_softly_fixed_4")
U = np.moveaxis(Umu, 4, 0)
trajN = 1
trajL = 0.01
MDsteps = 1
innerMC_N_force = 100
innerMC_N_H = 100
hb_offset = 100
hb_multi_hit = 10
M = 1.0
log_file = log_name(M, trajL, innerMC_N_force, innerMC_N_H)
action = GFAction(5.6, M, innerMC_N_force, innerMC_N_H, hb_offset, hb_multi_hit, log_file)
HMC(U, trajN, trajL, MDsteps, action)
