from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


class Fitting:

    max_fitting_order = 240
    min_fittig_order = 100
    fitting_step = 2

    def __init__(self, path):
        if path.endswith('.mat'):
            self.data = loadmat(path)
            self.fr = self.data['fr'][0]
            self.fr_dimless = self.fr / np.amax(self.fr)
            self.ReH = self.data['rey'][0]
            self.ImH = self.data['imy'][0]
            self.H = np.vectorize(complex)(self.ReH, self.ImH)
            self.n = self.max_fitting_order + 1
            self.m = len(self.fr)
            self.poly_complex = np.multiply(1j, np.zeros((self.n, self.m)))
            self.D = np.zeros((self.n))
            self.v = np.zeros((self.n - 2))
            self.b = []
            self.omega_n = []

    def forsythe_poly(self):
        poly_real = np.zeros((self.n, self.m))

        s = np.ones((1, self.m))
        self.D[0] = np.sqrt(2*np.sum(np.multiply(s, s)))
        poly_real[0, :] = np.divide(s, self.D[0])
        self.poly_complex[0, :] = np.multiply((1j**0), poly_real[0, :])

        s = np.multiply(self.fr_dimless, poly_real[0, :])
        self.D[1] = np.sqrt(2*np.sum(np.multiply(s, s)))
        poly_real[1, :] = np.divide(s, self.D[1])
        self.poly_complex[1, :] = np.multiply((1j**1), poly_real[1, :])

        for i in range(2, self.n):
            self.v[i - 2] = 2*np.sum(np.multiply(np.multiply(self.fr_dimless, poly_real[i - 2, :]), poly_real[i-1, :]))
            s = np.multiply(self.fr_dimless, poly_real[i - 1, :]) - self.v[i - 2]*poly_real[i - 2, :]
            self.D[i] = np.sqrt(2*np.sum(np.multiply(s, s)))
            poly_real[i, :] = np.divide(s, self.D[i])
            self.poly_complex[i, :] = np.multiply((1j**i), poly_real[i, :])
        self.poly_complex = np.transpose(self.poly_complex)

    def calc_coeff(self, fitting_order):
        P = self.poly_complex[:, :fitting_order]
        T = np.multiply(self.poly_complex[:, :fitting_order], np.transpose(np.tile(self.H, (fitting_order, 1))))
        w = np.multiply(self.poly_complex[:, fitting_order], self.H)

        Y = 2*np.real(np.dot(np.transpose(np.conjugate(P)), P))
        X = -2*np.real(np.dot(np.transpose(np.conjugate(P)), T))
        Z = 2*np.real(np.dot(np.transpose(np.conjugate(T)), T))
        g = 2*np.real(np.dot(np.transpose(np.conjugate(P)), w))
        f = -2*np.real(np.dot(np.transpose(np.conjugate(T)), w))

        self.b = np.dot(np.linalg.pinv(np.dot(np.dot(np.transpose(X), Y), X)-Z),
                        np.dot(np.dot(np.transpose(X), Y), f) - g)

    def comrade_poles(self, fitting_order):
        C = np.zeros((fitting_order, fitting_order))

        C[fitting_order - 1, :] = np.multiply(-1, self.b) #
        C = C + np.diag(self.D[1:fitting_order], 1) + np.diag(np.multiply(-1, self.v[:(fitting_order - 1)]), -1)
        np.multiply(C[fitting_order - 1, :], self.D[fitting_order])
        poles, w = np.linalg.eig(C)
        poles = poles[(np.divide(-1*np.real(poles), np.abs(poles))) > 0]

        omega = np.unique(np.multiply(np.absolute(poles), np.amax(self.fr)))
        omega = omega[(omega <= np.amax(self.fr))]
        self.omega_n.append(omega)

    def plot_results(self):
        fig, ax1 = plt.subplots()
        color = 'tab:red'

        ax1.set_xlabel('freq. [Hz]')
        ax1.set_ylabel('|H($\omega$)|', color=color)
        ax1.plot(self.fr, np.abs(self.H), color=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Order [-]', color=color)
        for i in range(len(self.omega_n)):
            ax2.plot(self.omega_n[i], np.multiply(self.min_fittig_order + i*self.fitting_step,
                                                  np.ones(len(self.omega_n[i]))), 'bo', markersize=3)
        fig.tight_layout()
        plt.show()

    def run_rfp(self):
        self.forsythe_poly()
        for i in range(self.min_fittig_order, self.max_fitting_order + self.fitting_step, self.fitting_step):
            self.calc_coeff(i)
            self.comrade_poles(i)
        self.plot_results()

    @classmethod
    def get_fitting_order(cls):
        return cls.max_fitting_order

    @classmethod
    def set_fitting_order(cls, order):
        if int(order) % 2 != 0 or not isinstance(order, int):
            print("Wrong fitting order number! Setted to the closet lower even integer number.")
            cls.max_fitting_order = int(order) - 1
        else:
            cls.max_fitting_order = order
