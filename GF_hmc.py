import numpy as np
from QCD import *

grid = (4,4,4,4)
U = coldConfiguration((4,)+grid+(3,3))
# Umu = readField("U_softly_fixed_4")
# U = np.moveaxis(Umu, 4, 0)
trajN = 3
trajL = 0.01
MDsteps = 1
innerMC_N = 10000
hb_offset = 100
hb_multi_hit = 10
M = 1.0
action = GFAction(5.6, M, innerMC_N, hb_offset, hb_multi_hit)
HMC(U, trajN, trajL, MDsteps, action)
