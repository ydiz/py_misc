# FIXME expm part is slow. 0.3s per time
from .core import *
from .heatbath import *
# from scipy.linalg import expm

def update_U(U, P, eps):
    # do not use np.exp (element-wise exp); use exp_traceless_antiHermi
    # np.array([expm(eps * x) for x in P.reshape((-1,3,3))]).reshape(U.shape)
    newU = np.matmul(exp_traceless_antiHermi(P, eps), U)
    newU = projectOnGroup(newU)
    return newU

def update_P(U, P, eps, action):
    newP = P - eps * action.deriv(U)
    return newP

def integrate_LeapFrog(U, P, trajL, MDsteps, action):
    eps = trajL / MDsteps
    for i in range(MDsteps):
        if i==0:
            P = update_P(U, P, eps/2.0, action)
            # print(P)
        U = update_U(U, P, eps)
        p_eps = eps if i!= MDsteps-1 else eps/2.0
        P = update_P(U, P, p_eps, action)
    return U, P

def cal_H(U, P, action, before_after, f=None):
    Hp = - np.sum(trace(np.matmul(P, P))).real
    Hs = action.S(U)
    H0 = Hp + Hs
    if before_after==0:
        # print("Before traj: Hp: ", Hp, "\tS: ", Hs, "\tTotal H: ", H0)
        f.write("Before traj: Hp: " + str(Hp) + "\tS: "+str(Hs) + "\tTotal H: "+str(H0)+'\n')
    else:
        # print("After traj: Hp: ", Hp, "\tS: ", Hs, "\tTotal H: ", H0)
        f.write("After traj: Hp: " + str(Hp) + "\tS: "+str(Hs) + "\tTotal H: "+str(H0)+'\n')
    return H0

def HMC(U, trajN, trajL, MDsteps, action):
    f = open(action.log_file, 'w+')
    for traj in range(trajN):
        # print("traj: ", traj)
        f.write("traj: "+str(traj) + '\n')
        P = generate_P(U.shape[1:5])

        # H0 = cal_H(U, P, action, 0)
        H0 = cal_H(U, P, action, 0, f)

        newU, newP = integrate_LeapFrog(U, P, trajL, MDsteps, action)

        # H1 = cal_H(newU, newP, action, 1)
        H1 = cal_H(newU, newP, action, 1, f)
        deltaH = H1 - H0
        # print("delta H: ", deltaH)
        f.write("delta H: "+str(deltaH)+ '\n')

        ave = 0

        if action.name == "GFAction":
            g = coldConfiguration(U.shape[1:])
            g = GF_heatbath(U, g, action.betaMM, action.hb_offset, action.hb_multi_hit)
            for i in range(action.innerMC_N_H):
                tt = deltaH + action.betaMM * (Omega_g(newU, g) - Omega_g(U, g))
                ave += np.exp(tt)
                if i%30==0:
                    # print("DeltaH tt: ", tt)
                    f.write("DeltaH tt: "+str(tt)+ '\n')
                g = GF_heatbath(U, g, action.betaMM, 1, action.hb_multi_hit)
            # print("average: ", np.log(ave/action.innerMC_N_H))
            f.write("average: "+str(np.log(ave/action.innerMC_N_H))+ '\n')

        # always accept
        U = newU

        # print("plaq: ", plaq_cal(newU))
        # print('-'*50)
        f.write("plaq: " + str(plaq_cal(newU))+ '\n')
        f.write('-'*50 + '\n')

        f.close()
