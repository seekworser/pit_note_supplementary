import numpy as np


# prepare rho and rho'
# rho = |HV><HV|
# rho' = (|HV><HV| + |VH><VH|)/2
rho = np.matrix([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])
rho_d = np.matrix([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
]) / 2

# prepare u (pi/4 rotated QWP)
u_qwp_single = np.matrix([
    [1, -1j],
    [-1j, 1],
]) / 2
u_qwp = np.kron(u_qwp_single, u_qwp_single)

# prepare P_i
P_1 = np.matrix([
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])
P_2 = np.matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
])
P_3 = np.matrix([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
])

# calculation
rho_after_qwp = u_qwp.H * rho * u_qwp
rho_d_after_qwp = u_qwp.H * rho_d * u_qwp
prob_P1_rho = np.trace(P_1 * rho_after_qwp)
prob_P2_rho = np.trace(P_2 * rho_after_qwp)
prob_P3_rho = np.trace(P_3 * rho_after_qwp)
prob_P1_rho_d = np.trace(P_1 * rho_d_after_qwp)
prob_P2_rho_d = np.trace(P_2 * rho_d_after_qwp)
prob_P3_rho_d = np.trace(P_3 * rho_d_after_qwp)

# output result
with open("result.txt", "w") as f:
    lines = ([
        "#"*50 + "\n",
        ("rho:"),
        (str(rho)),
        ("rho':"),
        (str(rho_d)),
        ("unitary matrix representing pi/4 QWP"),
        str(u_qwp),
        ("\n" + "#"*50 + "\n"),
        ("after unitary operation"),
        ("rho:"),
        (rho_after_qwp),
        ("rho_d:"),
        (rho_d_after_qwp),
        ("\n" + "#"*50 + "\n"),
        ("probability for projection measurement with P1 (HH)"),
        ("rho: {0.real:.5f}    rho':{1.real:.5f}".format(
            prob_P1_rho,
            prob_P1_rho_d
        )),
        ("\n" + "#"*50 + "\n"),
        ("probability for projection measurement with P2 (VV)"),
        ("rho: {0.real:.5f}    rho':{1.real:.5f}".format(
            prob_P2_rho,
            prob_P2_rho_d
        )),
        ("\n" + "#"*50 + "\n"),
        ("probability for projection measurement with P3 (HV and VH)"),
        ("rho: {0.real:.5f}    rho':{1.real:.5f}".format(
            prob_P3_rho,
            prob_P3_rho_d
        )),
        ("\n" + "#"*50 + "\n"),
    ])
    f.writelines(["{}\n".format(line) for line in lines])
