import numpy as np
import matplotlib.pyplot as plt
# from threading import Thread


class Trajectory(object):
    """Trajectory contains temporal dependencies of energies (energy),
     transition self.pabilities(p) and distance(r) and all functions
     for working with trajectories"""

    def __init__(self, dirname):
        """Loading of files with self.pabilities, energies and
        geometries by name of directory with these files"""

        p_file = open(dirname + "\\prob.out", "r")
        p_s = [line.split() for line in p_file.readlines()]

        geom_file = open(dirname + "\\Geo.out", "r")
        geo_s = [line.split() for line in geom_file.readlines()]

        energy_file = open(dirname + "\\energy.out", "r")
        e_s = [line.split() for line in energy_file.readlines()]

        p_s = p_s[3:]
        geo_s = geo_s[2:]
        e_s = e_s[3:]
        try:
            r = np.zeros(len(geo_s))
            t = np.zeros(len(p_s))
            p = np.zeros((len(p_s), len(p_s[0]) - 2))
            energy = np.zeros(len(e_s))
            for i in range(len(geo_s)):
                r[i] = float(geo_s[i][1])
                t[i] = float(p_s[i][0])
                energy[i] = float(e_s[i][2])
                for j in range(2, len(p_s[0])):
                    p[i][j - 2] = float(p_s[i][j])
                    if (i > 140) and (p[i, j - 2] > 0.0000001):
                        p[i][j - 2] = 0
            self.t = t
            self.r = r
            self.p = p
            self.energy = energy
        except IndexError:
            pass

    def intprob(self):
        """Calculate integral of probabilities over t"""
        prob = np.zeros((len(self.p), len(self.p[0])))
        for i in range(len(self.p) - 1):
            T = self.t[len(self.p) - 1] - self.t[i]
            for j in range(len(self.p[0])):
                prob[i][j] = np.trapz(
                    self.p[i:len(self.p), j], self.t[i:len(self.p)]) / T
        self.prob = prob

    def impact_squared(self, v0):
        """Calculate square of impact parameter"""
        b2 = np.zeros(len(self.r))
        for i in range(len(self.r)):
            dU = self.energy[i] - self.energy[len(self.r) - 1]
            b2[i] = self.r[i] ** 2 * (1 - 5.3E11 * dU / (v0 ** 2))
            if b2[i] < 0:
                b2[i] = 0
        return b2

    def sigma(self, v):
        """Calculate cross sections for external velocity,
         temperature and impact parameter and write them
         in file"""
        S = np.zeros(len(self.p[0]))
        b2 = self.impact_squared(v)
        for i in range(len(self.p[0])):
            S[i] = np.trapz(self.prob[:, i], b2)
        self.S = np.pi * S * 1E-16


def maxwell(v, T):
    """Calculate Maxwell distribution"""
    ex = np.exp(-2.189E-8 * float(v) ** 2 / float(T))
    A = 7.31E-12 / np.sqrt(float(T) ** 3)
    return A * ex * float(v) ** 2


def rate_const(sigma, v, T):
    """Calculate rate constants for external temperature"""
    S = np.zeros(len(sigma[0]))
    p = [maxwell(v[i], T) for i in range(len(v))]
    for j in range(len(sigma[0])):
        integrand = [sigma[i, j] * v[i] * p[i] for i in range(len(v))]
        S[j] = np.trapz(integrand, v)
    return S


track = Trajectory('Triplet4\\TRAJ_00013')
track.intprob()
# for j in range(len(track.prob[0])):
#     plt.plot(track.r, track.prob[:, j])
# plt.show()

v = np.arange(0.1, 1.1E6, 5E3)
s = np.zeros((len(v), len(track.prob[0])))
for i in range(len(v)):
    track.sigma(v[i])
    s[i] = track.S

T = np.arange(100, 1100, 100)
# T = [100]

out_file = open("rate_const_2p10.txt", "w")
for temp in T:
    # track.boltzmann(temp)
    # for vel in v:
        # track.impact_squared(vel)
        # track.print_sigmas(vel, temp)
    temp_str = str(temp)
    k = rate_const(s, v, temp)
    for rate in k:
        temp_str += " " + str(rate)
    temp_str += "\n"
    out_file.write(temp_str)
