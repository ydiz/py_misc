import numpy as np
from QCD import *

grid = (4,4,4,4)
U = coldConfiguration((4,)+grid+(3,3))
# Umu = readField("U_softly_fixed_4")
# U = np.moveaxis(Umu, 4, 0)
trajN = 1
trajL = 0.03
MDsteps = 1
# innerMC_N = 1000
innerMC_N_force = 1000
innerMC_N_H = 1000
hb_offset = 1000
hb_multi_hit = 10
M = 1.0

action = GFAction(5.6, M, innerMC_N_force, innerMC_N_H, hb_offset, hb_multi_hit)
HMC(U, trajN, trajL, MDsteps, action)
