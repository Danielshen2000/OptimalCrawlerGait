import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad, fixed_quad
from scipy import interpolate, optimize
import sys
import os
from scipy import interpolate, optimize, integrate
import logging

sys.path.append(os.path.abspath('../describing_function'))
from DF_num import Crawler
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
my_path = os.path.abspath(__file__)
print("This is the name of the program:", sys.argv[0])
"""
This case is only for uncoupled, undamped case
"""
m = 1
k = 0.1 # 0.1 LS, 0.09 ZS
c = 0.1 # 0.1 LS, 0.11 ZS
l = 0.1

ratio = np.linspace(0.1, 10, 20)
# omega = float(sys.argv[1]) # 1 0.45
omega = np.sqrt(2*k/m)

print(omega)

T = 2 * np.pi / omega
print(T)

alpha = 3.3# 6 3.3
beta = 0.05 #0.05 0.05
u_bar = 0.01
fric_bar = 0.01
crawler = Crawler(m, l, k, c, u_bar, fric_bar, 0, 0)
crawler.nondimensionalize()
pi_u = crawler.pi_u
pi_f = crawler.pi_f
zeta = crawler.zeta
print(pi_u)
print(pi_f)
print(zeta)


class Optimization:
    def __init__(self, t, u, gamma=1, ratio=1.2):
        self.u = u
        self.t = t
        self.f = None
        self.z3 = None
        self.z4 = None
        self.f_dfdz3 = None
        self.f_dfdz4 = None
        # Friction Parameters
        self.ratio = ratio
        self.gamma = gamma

    def fric_smo(self, x_dot):
        return 0.5 * self.gamma * ((1.2 + 1) / (1 + pyo.exp(-(-1 * x_dot - 0.2303) / 0.1)) + 1 - 1.2)

    def fric_smo_dot(self, x_dot):
        return -11 * pyo.exp(10 * x_dot + 2.303) / (pyo.exp(10 * x_dot + 2.303) + 1) ** 2

    def define_state_model(self, f):
        model = m = pyo.ConcreteModel()
        m.t = ContinuousSet(bounds=(0, T))

        # Define the independent Variables
        m.z1 = pyo.Var(m.t)
        m.z2 = pyo.Var(m.t)
        m.z3 = pyo.Var(m.t)
        m.z4 = pyo.Var(m.t)
        # Define the Derivatives
        m.dz1 = DerivativeVar(m.z1, wrt=m.t)
        m.dz2 = DerivativeVar(m.z2, wrt=m.t)
        m.dz3 = DerivativeVar(m.z3, wrt=m.t, bounds=(-20, 20))
        m.dz4 = DerivativeVar(m.z4, wrt=m.t, bounds=(-20, 20))

        # Define the differential equations
        def _dz1(m, t):
            return m.dz1[t] == m.z3[t] - m.z4[t]

        m.z1_dot = pyo.Constraint(m.t, rule=_dz1)

        def _dz2(m, t):
            return m.dz2[t] == m.z3[t] + m.z4[t]

        m.z2_dot = pyo.Constraint(m.t, rule=_dz2)

        def _dz3(m, t):
            return m.dz3[t] == pi_u * f(t) + pi_f * self.fric_smo(m.z3[t]) - 0.5 * m.z1[t] - zeta * (m.z3[t] - m.z4[t])

        m.z3_dot = pyo.Constraint(m.t, rule=_dz3)

        def _dz4(m, t):
            return m.dz4[t] == -pi_u * f(t) + pi_f * self.fric_smo(m.z4[t]) + 0.5 * m.z1[t] + zeta * (m.z3[t] - m.z4[t])

        m.z4_dot = pyo.Constraint(m.t, rule=_dz4)

        # Define the boundary conditions
        m.z1_bc = pyo.Constraint(expr=m.z1[m.t.first()] == m.z1[m.t.last()])
        m.z2_bc = pyo.Constraint(expr=m.z2[0] == 0)
        m.z3_bc = pyo.Constraint(expr=m.z3[m.t.first()] == m.z3[m.t.last()])
        m.z4_bc = pyo.Constraint(expr=m.z4[m.t.first()] == m.z4[m.t.last()])

        m.obj = pyo.Objective(expr=1)

        discretizer = pyo.TransformationFactory('dae.collocation')
        discretizer.apply_to(m, nfe=30, ncp=10, scheme='LAGRANGE-RADAU')
        return m

    def solve_model(self, m, solve_state = True, tee = False):
        solver = pyo.SolverFactory('ipopt')
        solver.options['max_iter'] = 500
        solver.options['halt_on_ampl_error'] = 'yes'
        results = solver.solve(m, tee=tee)
        if solve_state:
            z1 = [pyo.value(m.z1[tn]) for tn in m.t]
            z2 = [pyo.value(m.z2[tn]) for tn in m.t]
            z3 = [pyo.value(m.z3[tn]) for tn in m.t]
            z4 = [pyo.value(m.z4[tn]) for tn in m.t]
            z = [z1, z2, z3, z4]
        else:
            lam1 = [pyo.value(m.lam1[tn]) for tn in m.t]
            lam2 = [pyo.value(m.lam2[tn]) for tn in m.t]
            lam3 = [pyo.value(m.lam3[tn]) for tn in m.t]
            lam4 = [pyo.value(m.lam4[tn]) for tn in m.t]
            z = [lam1, lam2, lam3, lam4]
        return z

    def position(self, m):
        # t_step = m.t[2] - m.t[1]
        # z3 = np.array([pyo.value(m.z3[tn]) for tn in m.t]) * t_step
        # # z4 = np.array([pyo.value(m.z4[tn]) for tn in m.t]) * t_step
        # return np.cumsum(z3), np.cumsum(z4)
        t = list(m.t)
        z3 = np.array([pyo.value(m.z3[tn]) for tn in m.t])
        z4 = np.array([pyo.value(m.z4[tn]) for tn in m.t])
        p1 = integrate.cumtrapz(z3, t, initial = 0)
        p2 = integrate.cumtrapz(z4, t, initial = 0)
        return p1, p2

    def define_costate_model(self, m_state):
        model = m = pyo.ConcreteModel()
        m.t = ContinuousSet(bounds=(0, T))
        # Define the independent Variables
        m.lam1 = pyo.Var(m.t)
        m.lam2 = pyo.Var(m.t)
        m.lam3 = pyo.Var(m.t)
        m.lam4 = pyo.Var(m.t)

        # Define the Derivatives
        m.dlam1 = DerivativeVar(m.lam1, wrt=m.t)
        m.dlam2 = DerivativeVar(m.lam2, wrt=m.t)
        m.dlam3 = DerivativeVar(m.lam3, wrt=m.t)
        m.dlam4 = DerivativeVar(m.lam4, wrt=m.t)

        # Define the differential equations
        def _lam1dot(m, t):
            return m.dlam1[t] == 0.5 * m.lam3[t] - 0.5 * m.lam4[t] + 2 * beta * m_state.z1[t]

        m.lam1dot = pyo.Constraint(m.t, rule=_lam1dot)

        def _lam2dot(m, t):
            return m.dlam2[t] == 0

        m.lam2dot = pyo.Constraint(m.t, rule=_lam2dot)

        def _lam3dot(m, t):
            return m.dlam3[t] == -m.lam1[t] - m.lam2[t] - (pi_f * self.fric_smo_dot(m_state.z3[t]) - zeta) * m.lam3[
                t] - zeta * m.lam4[t]

        m.lam3dot = pyo.Constraint(m.t, rule=_lam3dot)

        def _lam4dot(m, t):
            return m.dlam4[t] == m.lam1[t] - m.lam2[t] - zeta * m.lam3[t] - (
                        pi_f * self.fric_smo_dot(m_state.z4[t]) - zeta) * m.lam4[t]

        m.lam4dot = pyo.Constraint(m.t, rule=_lam4dot)

        # Define the boundary conditions
        m.lam1_bc = pyo.Constraint(expr = m.lam1[m.t.first()] == m.lam1[m.t.last()])
        m.lam2_bc = pyo.Constraint(expr = m.lam2[m.t.first()] == 1)
        # m.lam2_bc = pyo.Constraint(expr=m.lam2[m.t.last()] == 2*m_state.z2[m.t.last()])
        m.lam3_bc = pyo.Constraint(expr = m.lam3[m.t.first()] == m.lam3[m.t.last()])
        m.lam4_bc = pyo.Constraint(expr = m.lam4[m.t.first()] == m.lam4[m.t.last()])

        m.obj = pyo.Objective(expr = 1)

        discretizer = pyo.TransformationFactory('dae.collocation')
        discretizer.apply_to(m, nfe=30, ncp=10, scheme='LAGRANGE-RADAU')
        logging.debug("Costate Model Defined")
        return m

    def optimization_onestep(self, eps = 0.05):
        self.f = interpolate.interp1d(self.t, self.u, fill_value = "extrpolate")
        m_state = self.define_state_model(self.f)
        z = self.solve_model(m_state)
        p1, p2 = self.position(m_state)
        m_costate = self.define_costate_model(m_state)
        lam = self.solve_model(m_costate, False)
        self.t = list(m_state.t)
        delta_u = pi_u*np.array(lam[2]) - pi_u*np.array(lam[3]) - 2 * alpha * self.f(self.t)
        self.u = self.f(self.t) + eps * delta_u
        return self.u, z[2], z[3], p1, p2

    def optimization_main(self, eps = 0.02, max_iter = 1000):
        max_cost = -1*np.inf
        u_opt = None
        z3_opt = None
        z4_opt = None
        p1_opt = None
        p2_opt = None
        for i in range(max_iter):
            print("---------------Iteration {}----------------".format(i))
            u, z3, z4, p1, p2= self.optimization_onestep(eps)
            cost = eval_cost(self.t, self.f(self.t), p1 - p2, p1 + p2)
            print(cost)
            if i>50 and cost[0] > max_cost:
                max_cost = cost[0]
                u_opt = u
                z3_opt = z3
                z4_opt = z4
                p1_opt = p1
                p2_opt = p2
            print(max_cost)
        return u_opt, z3_opt, z4_opt, p1_opt, p2_opt, max_cost



def eval_cost(t, u, z1, z2):
    J_dist = z2[-1]
    # J_u = np.sum(np.array(u)**2*(t[1]-t[0]))
    # print(len(u))
    # print(len(t))
    J_u = alpha*integrate.trapz(np.array(u)**2, t)
    J_strain = beta*integrate.trapz(np.array(z1)**2, t)
    return J_dist-J_u-J_strain, J_dist, J_u, J_strain




#
lam1 = 2
t = np.linspace(0, T, int(T * 100000))
u = 1*np.sin(omega * t)
# u = np.load("result_f.npy")
# t = np.linspace(0, T, len(u))
plt.plot(t,u)
f = interpolate.interp1d(t, u, fill_value="extrapolate")
print(t)
print(f(t))
my_opt = Optimization(t, u, 1)


u, z3, z4, p1, p2, cost = my_opt.optimization_main(0.02, 1000)
# # u, z3, z4, p1, p2 = my_opt.optimization_main(0.004, 100)
# # u, z3, z4, p1, p2 = my_opt.optimization_main(0.001, 100)
# plt.plot(my_opt.t, u)
# plt.plot(my_opt.t, z3)
# plt.plot(my_opt.t, z4)
# plt.plot(my_opt.t, p1)
# plt.plot(my_opt.t, p2)
# plt.savefig("test4.png")
# np.savetxt("u_opt.txt",my_opt.u)
# np.savetxt("t.txt", my_opt.t)

# u = np.loadtxt("u_opt.txt")
# my_opt.t = np.loadtxt("t.txt")
# my_opt = Optimization(my_opt.t, u, 1)
# u, z3, z4, p1, p2 = my_opt.optimization_main(0.02, 1000)

f  = interpolate.interp1d(my_opt.t, u, fill_value = "extrapolate")



m_state = my_opt.define_state_model(f)
z = my_opt.solve_model(m_state)
p1, p2 = my_opt.position(m_state)
plt.plot(m_state.t, z[0])
plt.plot(m_state.t, z[1])
plt.plot(m_state.t, z[2])
plt.plot(m_state.t, z[3])
# plt.plot(m_state.t, np.array(z[2])+np.array(z[3]))
plt.plot(m_state.t, p1)
plt.plot(m_state.t, p2)
plt.plot(m_state.t, (p1+p2)/2)
plt.savefig(f"./result/test omega {omega} k {k} alpha{alpha}.png")
plt.clf()

print(cost/T)
cost_file = open("./result/cost.txt", "a")
cost_file.write(f"{omega} {k} {cost/T} {alpha}\n")
cost_file.close()

fric_3 = [my_opt.fric_smo((pyo.value(m_state.z3[tn]))) for tn in m_state.t]
fric_4 = [my_opt.fric_smo((pyo.value(m_state.z4[tn]))) for tn in m_state.t]
dfric_3 = [my_opt.fric_smo_dot(pyo.value(m_state.z3[tn])) for tn in m_state.t]
dfric_4 = [my_opt.fric_smo_dot(pyo.value(m_state.z4[tn])) for tn in m_state.t]
plt.plot(m_state.t, fric_3)
plt.plot(m_state.t, fric_4)
plt.plot(m_state.t, dfric_3)
plt.plot(m_state.t, dfric_4)
plt.savefig(f"./result/friction omega{omega} k{k} alpha{alpha}.png")

m_costate = my_opt.define_costate_model(m_state)
t = list(m_costate.t)
lam = my_opt.solve_model(m_costate, False)
plt.clf();
plt.plot(m_costate.t, lam[0], label = "costate 1")
plt.plot(m_costate.t, lam[1], label = "costate 2")
plt.plot(m_costate.t, lam[2], label = "costate 3")
plt.plot(m_costate.t, lam[3], label = "costate 4")
# plt.plot(m_costate.t, pi_u*np.array(lam[2])-pi_u*np.array(lam[3]), label = "update (only energy term)")
# plt.plot(m_costate.t, pi_u*np.array(lam[2])-pi_u*np.array(lam[3])-2*alpha*f(t), label = "update (w/o strain)")
plt.legend()
plt.savefig(f"./result/FINALandcostate omega {omega} k {k} alpha {alpha}.png")
# print(list(m_costate.t))
# print(f(t))
plt.clf()
plt.plot(t, f(t))
print(eval_cost(t, f(t), p1 - p2, p1 + p2))

print(cost/T)
np.save('result_z.npy', z)
np.save('result_lam.npy', lam)
np.save('result_p1', p1)
np.save('result_p2', p2)
np.save('result_t', t)
np.save('result_f', f(t))
np.save('result_fric3', fric_3)
np.save('result_fric4', fric_4)
np.save('result_dfric3', dfric_3)
np.save('result_dfric4', dfric_4)



