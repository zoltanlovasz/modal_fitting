from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


class Fitting:

    max_fitting_order = 320
    min_fittig_order = 200
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
            self.poly_complex_nom = np.multiply(1j, np.zeros((self.n, self.m)))
            self.poly_complex_denom = np.multiply(1j, np.zeros((self.n, self.m)))
            self.D = np.zeros((self.n))
            self.v = np.zeros((self.n - 2))
            self.b = []
            self.omega_n = []

    def forsythe_nom(self):
        poly_real = np.zeros((self.n, self.m))

        s = np.ones((1, self.m))
        D = np.sqrt(2*np.sum(np.multiply(s, s)))
        poly_real[0, :] = np.divide(s, D)
        self.poly_complex_nom[0, :] = np.multiply((1j**0), poly_real[0, :])

        s = np.multiply(self.fr_dimless, poly_real[0, :])
        D = np.sqrt(2*np.sum(np.multiply(s, s)))
        poly_real[1, :] = np.divide(s, D)
        self.poly_complex_nom[1, :] = np.multiply((1j**1), poly_real[1, :])

        for i in range(2, self.n):
            v = 2*np.sum(np.multiply(np.multiply(self.fr_dimless, poly_real[i - 2, :]), poly_real[i-1, :]))
            s = np.multiply(self.fr_dimless, poly_real[i - 1, :]) - v*poly_real[i - 2, :]
            D = np.sqrt(2*np.sum(np.multiply(s, s)))
            poly_real[i, :] = np.divide(s, D)
            self.poly_complex_nom[i, :] = np.multiply((1j**i), poly_real[i, :])
        self.poly_complex_nom = np.transpose(self.poly_complex_nom)

    def forsythe_denom(self):
        poly_real = np.zeros((self.n, self.m))
        q = np.square(np.abs(self.H))

        s = np.ones((1, self.m))
        self.D[0] = np.sqrt(2*np.sum(np.multiply(np.multiply(s, s), q)))
        poly_real[0, :] = np.divide(s, self.D[0])
        self.poly_complex_denom[0, :] = np.multiply((1j**0), poly_real[0, :])

        s = np.multiply(self.fr_dimless, poly_real[0, :])
        self.D[1] = np.sqrt(2*np.sum(np.multiply(np.multiply(s, s), q)))
        poly_real[1, :] = np.divide(s, self.D[1])
        self.poly_complex_denom[1, :] = np.multiply((1j**1), poly_real[1, :])

        for i in range(2, self.n):
            self.v[i - 2] = 2 * np.sum(np.multiply(np.multiply(self.fr_dimless, poly_real[i - 1, :]),
                                                   np.multiply(poly_real[i - 2, :], q)))
            s = np.multiply(self.fr_dimless, poly_real[i - 1, :]) - np.multiply(self.v[i - 2], poly_real[i - 2, :])
            self.D[i] = np.sqrt(2*np.sum(np.multiply(np.multiply(s, s), q)))
            poly_real[i, :] = np.divide(s, self.D[i])
            self.poly_complex_denom[i, :] = np.multiply((1j**i), poly_real[i, :])
        self.poly_complex_denom = np.transpose(self.poly_complex_denom)

    def calc_denom_coeff(self, fitting_order):
        P = self.poly_complex_nom[:, :fitting_order]
        T = np.multiply(self.poly_complex_denom[:, :fitting_order], np.transpose(np.tile(self.H, (fitting_order, 1))))
        w = np.multiply(self.poly_complex_denom[:, fitting_order], self.H)

        Y = 2*np.real(np.dot(np.transpose(np.conjugate(P)), P))
        X = -2*np.real(np.dot(np.transpose(np.conjugate(P)), T))
        Z = 2*np.real(np.dot(np.transpose(np.conjugate(T)), T))
        g = 2*np.real(np.dot(np.transpose(np.conjugate(P)), w))
        f = -2*np.real(np.dot(np.transpose(np.conjugate(T)), w))

        self.b = np.dot(np.linalg.pinv(np.dot(np.dot(np.transpose(X), Y), X)-Z), np.dot(np.dot(np.transpose(X), Y), f) - g)

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
            ax2.plot(self.omega_n[i], np.multiply(self.min_fittig_order + i*self.fitting_step, np.ones(len(self.omega_n[i]))), 'ro')
        fig.tight_layout()
        plt.show()

    def run_rpf(self):
        self.forsythe_nom()
        self.forsythe_denom()
        for i in range(self.min_fittig_order, self.max_fitting_order + self.fitting_step, self.fitting_step):
            self.calc_denom_coeff(i)
            self.comrade_poles(i)
        self.plot_results()

    def plot_nyquist(self):
        fig = plt.figure(0)
        plt.plot(self.ReH, self.ImH)
        plt.title("Real - Imaginary diagram")
        plt.xlabel("Re(H($\omega$))")
        plt.ylabel("Im(H($\omega$))")
        fig.canvas.set_window_title('Nyquist diagram')
        plt.show()

    def plot_bode(self):
        fig = plt.figure(0)
        plt.plot(self.fr, np.abs(self.H))
        plt.title("Magnitude - Frequency diagram")
        plt.xlabel("Frequency")
        plt.ylabel("|H($\omega$)|")
        fig.canvas.set_window_title('Magnitude diagram')
        plt.show()

    def plot_imag(self):
        fig = plt.figure(0)
        plt.plot(self.fr, self.ImH)
        plt.title("Imaginary - Frequency diagram")
        plt.xlabel("Frequency")
        plt.ylabel("Im(H($\omega$))")
        fig.canvas.set_window_title('Im(H($\omega$)) diagram')
        plt.show()

    def plot_real(self):
        fig = plt.figure(0)
        plt.plot(self.fr, self.ReH)
        plt.title("Real - Frequency diagram")
        plt.xlabel("Frequency")
        plt.ylabel("Re(H($\omega$))")
        fig.canvas.set_window_title('Re(H($\omega$)) diagram')
        plt.show()

    @classmethod
    def get_fitting_order(cls):
        return cls.fitting_order

    @classmethod
    def set_fitting_order(cls, order):
        if int(order) % 2 != 0 or not isinstance(order, int):
            print("Wrong fitting order number! Setted to the closet lower even integer number.")
            cls.fitting_order = int(order) - 1
        else:
            cls.fitting_order = order


measurement1 = Fitting('theoretical_frf.mat')
measurement1.run_rpf()

