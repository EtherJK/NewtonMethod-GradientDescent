import numpy as np
import math

class RPA_system:
    #from free_energy import RPA_free_energy_derivative, RPA_free_energy

    def __init__(self, chis, charges, Ns):
        self.chis = chis
        self.charges = charges
        self.Ns = Ns
        self.component = len(Ns)

    def constrain_phi_s(self, phi_s1):
        self.phi_s1 = phi_s1

    def set_initial_A_B(self, phi_a, phi_s):
        self.phi_a = phi_a
        self.phi_s = phi_s

    def set_initial_AB_C(self, phi_a, phi_b, phi_s):
        self.phi_a = phi_a
        self.phi_b = phi_b
        self.phi_s = phi_s


    def free_energy(self, phis):
        return (RPA_free_energy(phis, self.chis, self.charges, self.Ns))
    
    def free_energy_derivative(self, phis):
        return (RPA_free_energy_derivative(phis, self.chis, self.charges, self.Ns))

    def free_energy_intercept(self, phis):
        f = self.free_energy(phis)
        f_derivative = self.free_energy_derivative(phis)
        intercept = f - np.sum(f_derivative * phis)
        return intercept

    def equations_A_B_type(self, x, free_energy = None):
        # the equations=0 would like to solve
        # x include: phi_a1, phi_b1, phi_a2, phi_b2, phis2, xi
        y = np.zeros(6, dtype = float)
        phis1 = np.append(x[0:2], np.array([self.phi_s1, self.phi_s1]))
        phis2 = np.append(x[2:5], x[4])

        # chemical
        derivative = self.free_energy_derivative(phis1) - self.free_energy_derivative(phis2) - x[5] * charges
        y[0:3] = derivative[0:3]

        # intercept
        y[3] = self.free_energy_intercept(phis1) - self.free_energy_intercept(phis2)

        # electroneutral
        def electroneutral(phis, charges):
            q = phis * charges
            return (q[0] + q[2] - q[1] - q[3])
        y[4] = electroneutral(phis1, self.charges)
        y[5] = electroneutral(phis2, self.charges)

        if free_energy:
            print('mu diviate: ', derivative)
            print('intercept: ', y[3])
            print('Free Energy 1: ', self.free_energy(phis1))
            print('Free Energy 2: ', self.free_energy(phis2))
        return y

    def free_energy_A_B_type(self, phis_0, phis_1, phi_a2):
        # phis_1 include: phi_a1, phi_b1, phi_s1
        phis1 = np.append(phis_1[:], phis_1[2])
        v1 = (phis_0[0] - phi_a2) / (phis_1[0] - phi_a2)
        v2 = 1 - v1
        phis_2 = (phis_0 - v1 * phis_1) / v2
        phis2 = np.append(phis_2[:], phis_2[2])
        f1 = self.free_energy(phis1)
        f2 = self.free_energy(phis2)
        F = f1 * v1 + f2 * v2
        return F

    def equations_f_A_B_type(self, x):
        # x: phi_a1, phi_s1, phi_a2
        phis1 = np.array([x[0], x[0], x[1], x[1]])
        v2 = (self.phi_a - phis1[0]) / (x[2] - phis1[0])
        v1 = 1 - v2
        phi_b2 = x[2]
        phi_s2 = (self.phi_s - v1 * phis1[2]) / v2
        phis2 = np.array([x[2], phi_b2, phi_s2, phi_s2])
        f1 = self.free_energy(phis1)
        f2 = self.free_energy(phis2)
        F = f1 * v1 + f2 * v2
        return F

    def equations_f_AB_C_type(self, x):
        # x: 0-phi_a1, 1-phi_a2, 2-phi_a3, 3-phi_b1, 4-phi_b2, 5-phi_b3, 6-phi_s1, 7-phi_s2
        v1 = (self.phi_a * x[4] - self.phi_b * x[1] - self.phi_a * x[5] + self.phi_b * x[2] + x[1] * x[5] - x[2] * x[4]) \
           / (x[0] * x[4] - x[3] * x[1] - x[0] * x[5] + x[3] * x[2] + x[1] * x[5] - x[2] * x[4])
        v2 = -(self.phi_a * x[3] - self.phi_b * x[0] - self.phi_a * x[5] + self.phi_b * x[2] + x[0] * x[5] - x[3] * x[2]) \
            / (x[0] * x[4] - x[3] * x[1] - x[0] * x[5] + x[3] * x[2] + x[1] * x[5] - x[2] * x[4])
        v3 = 1 - v1 - v2
        phi_s3 = (self.phi_s - v1 * x[6] - v2 * x[7]) / v3
        phics = (x[0:3] * self.charges[0] + x[3:6] * self.charges[1] ) / self.charges[2]
        phis1 = np.array([x[0], x[3], phics[0], x[6], x[6]])
        phis2 = np.array([x[1], x[4], phics[1], x[7], x[7]])
        phis3 = np.array([x[2], x[5], phics[2], phi_s3, phi_s3])
        f1 = self.free_energy(phis1)
        f2 = self.free_energy(phis2)
        f3 = self.free_energy(phis3)
        F = f1 * v1 + f2 * v2 + f3 * v3
        return F

        

        



# calculate free energy
def RPA_free_energy(phis, chis, charges, Ns):
    # !!FOR ASYMMETRIC CASES: PHI SHOULD INCLUDE CATION & INION
    # calculate homogenous free energy under the VO theory
    # phis: phi of different components, not including water
    # chis: chi between different components, is a matrix
    # charges: average charge for each monomers
    # Ns: chain length for each mol, excluding solvent
    num_compo = len(phis)
    if (np.array([num_compo+1, num_compo+1]) != chis.shape).any():
        raise IndexError('Size of chis mismatch')
    if len(Ns) != num_compo:
        raise IndexError('Size of Ns mismatch')
    if len(charges) != num_compo:
        raise IndexError('Size of charges mismatch')

    # parameters
    l = 0.85e-9
    T = 300
    
    e = 1.6022e-19
    epsilon = 78
    epsilon0 = 8.8542e-12
    kB = 1.380649e-23
    lb = e**2 / 4 / math.pi / epsilon / epsilon0 / kB / T
    b = l

    # electrostatic interation of RPA system
    # f = 1/2 * l**3 * int(dq* (ln(det(I+GU) - rho*U)))
    phisV = phis / l**3
    steps = 3000 # steps to calculate energy

    q = [1.01**i for i in range(0, steps)]
    q = np.array(q, dtype = float) * 0.1 - 0.1 + 1e-10
    q = q.reshape(len(q), 1)
    gs = phisV * (1 + Ns / (1 + q**2 * b**2 * Ns / 12))
    for i in range(0, len(Ns)):
        if Ns[i] == 1:
            gs[:,i] = phisV[i]
    U = 4 * math.pi * lb / (q**2)
    sum1 = np.sum(gs * charges**2 * U, axis=1)
    sum2 = np.sum(phisV * U * charges**2, axis=1)
    dif = high_res_log_plus_1(sum1) - sum2
    for i in range(0, len(dif)):
        if sum2[i] != 0 and abs(dif[i] / sum2[i]) < 1e-7:
            dif[i] = 0
    q = q.T[0, :]
    inte = q**2 * dif / 4 / math.pi**2
    elec_interact = np.trapz(inte, q) * l**3

    # entropy
    phis2 = phis[:]
    phis2 = np.append(phis2, [1 - sum(phis)])
    entropy = 0
    for i in range(0, len(phis)):
        entropy += phi_log_phi(phis[i]) / Ns[i]
    entropy += phi_log_phi(1 - sum(phis))

    # Flory-Huggins free energy
    fh_interac = 0.5 * np.dot(np.dot(phis2, chis), phis2)
    f_tot = elec_interact + entropy + fh_interac
    #print(f_tot)
    return f_tot


def RPA_free_energy_derivative(phis, chis, charges, Ns):
    # calculate the derivative of free energy under RPA theory
    # for asymmetric cases
    # aim: which component would like to calculate

    # parameters
    l = 0.85e-9
    T = 300
    
    e = 1.6022e-19
    epsilon = 78
    epsilon0 = 8.8542e-12
    kB = 1.380649e-23
    lb = e**2 / 4 / math.pi / epsilon / epsilon0 / kB / T
    b = l

    # electrostatic interation of RPA system
    # f = 1/2 * int(dq* (gU/(sum(gU)+1) - U))
    phisV = phis / l**3
    steps = 3000 # steps to calculate energy

    q = [1.01**i for i in range(0, steps)]
    q = np.array(q, dtype = float) * 0.1 - 0.1 + 1e-10
    q = q.reshape(len(q), 1)
    pure_gs = (1 + Ns / (1 + q**2 * b**2 * Ns / 12))
    gs = phisV * pure_gs
    for i in range(0, len(Ns)):
        if Ns[i] == 1:
            gs[:,i] = phisV[i]
            pure_gs[:,i] = 1
    U = 4 * math.pi * lb / (q**2)
    denominator = np.sum(gs * charges**2 * U, axis=1, keepdims=True) + 1
    numerator = pure_gs * U * charges**2
    term2 = U * charges**2
    dif = numerator / denominator - term2
    inte = q**2 * dif / 4 / math.pi**2
    elec_interact = np.trapz(inte, q, axis = 0)
    
    # entropy
    phis2 = phis[:]
    phis2 = np.append(phis2, [1 - sum(phis)])
    entropy = np.log(phis) / Ns - np.log(1 - np.sum(phis)) - 1 + 1 / Ns

    # Flory-Huggins interaction
    chis0 = chis[0:-1, 0:-1]
    fh_interac = np.dot(phis, chis0) + np.diagonal(chis0) * phis - np.sum(chis[-1, 0:-1] * phis) + chis[-1, 0:-1] * (1 - sum(phis))
    derivative = elec_interact + entropy + fh_interac
    return derivative




def high_res_log_plus_1(number):
    result = np.zeros(np.size(number))
    for i in range(0, len(number)):
        if number[i] > 0 and number[i] < 1e-10:
            result[i] = number[i]
        else:
            result[i] = math.log(number[i] + 1)
    return result

def phi_log_phi(phi):
    if phi > 0:
        return phi * np.log(phi)
    elif phi == 0:
        return 0





if __name__ == '__main__':
    phis = np.array([0.5, 0.1])
    chis = np.array([[0, 1, 2,], [1, 0, 3,], [2, 3, 0,]])
    charges = np.array([1, 0.5])
    Ns = np.array([10, 1])
    print(RPA_free_energy_derivative(phis, chis, charges, Ns))
    print((RPA_free_energy(phis, chis, charges, Ns) - RPA_free_energy(np.array([0.5-1e-5, 0.1]), chis, charges, Ns))/1e-5)
    
