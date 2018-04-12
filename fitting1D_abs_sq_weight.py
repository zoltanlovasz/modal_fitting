from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


class Fitting:

    fitting_order = 200
    min_fittig_order = 100

    def __init__(self, path):
        if path.endswith('.mat'):
            self.data = loadmat(path)
            self.fr = self.data['fr'][0]
            self.fr_dimless = self.fr / np.amax(self.fr)
            self.ReH = self.data['rey'][0]
            self.ImH = self.data['imy'][0]
            self.H = np.vectorize(complex)(self.ReH, self.ImH)
            self.n = self.fitting_order + 1
            self.m = len(self.fr)
            self.poly_complex_denom = 1j*np.zeros((self.n, self.m))
            self.poly_complex_nom = 1j*np.zeros((self.n, self.m))
            self.v = np.zeros((self.fitting_order - 1))
            self.D = np.zeros((self.fitting_order + 1))
            self.b = np.zeros((self.fitting_order + 1))
            self.omega_n = []
            print(np.amax(self.fr))

    def forsythe_nom(self):
        poly_real_nom = np.zeros((self.n, self.m))

        poly_real_nom[0, :] = 1
        D_nom = 2 * np.dot(poly_real_nom[0, :], poly_real_nom[0, :])
        poly_real_nom[0, :] = np.divide(poly_real_nom[0, :], np.sqrt(D_nom))
        self.poly_complex_nom[0, :] = (1j**0)*poly_real_nom[0, :]

        poly_real_nom[1, :] = np.multiply(self.fr_dimless, poly_real_nom[0, :])
        D_nom = 2 * np.dot(poly_real_nom[1, :], poly_real_nom[1, :])
        poly_real_nom[1, :] = np.divide(poly_real_nom[1, :], np.sqrt(D_nom))
        self.poly_complex_nom[1, :] = (1j**1)*poly_real_nom[0, :]

        for i in range(2, self.n):
            v_nom = 2 * np.dot(np.multiply(self.fr_dimless, poly_real_nom[i - 1, :]),
                               poly_real_nom[i - 2, :])
            poly_real_nom[i, :] = np.multiply(self.fr_dimless, poly_real_nom[i - 1, :]) \
                                  - v_nom * poly_real_nom[i - 2, :]
            D_nom = 2 * np.dot(poly_real_nom[i, :], poly_real_nom[i, :])
            poly_real_nom[i, :] = np.divide(poly_real_nom[i, :], np.sqrt(D_nom))
            self.poly_complex_nom[i, :] = (1j ** i) * poly_real_nom[i, :]
        self.poly_complex_nom = np.transpose(self.poly_complex_nom)

    def forsythe_denom(self):
        poly_real_denom = np.zeros((self.n, self.m))
        q = np.square(np.abs(self.H))

        poly_real_denom[0, :] = 1
        self.D[0] = 2 * np.dot(np.multiply(q, poly_real_denom[0, :]), poly_real_denom[0, :])
        poly_real_denom[0, :] = poly_real_denom[0, :] / np.sqrt(self.D[0])
        self.poly_complex_denom[0, :] = (1j**0)*poly_real_denom[0, :]

        poly_real_denom[1, :] = np.multiply(self.fr_dimless, poly_real_denom[0, :])
        self.D[1] = 2 * np.dot(np.multiply(q, poly_real_denom[1, :]), poly_real_denom[1, :])
        poly_real_denom[1, :] = poly_real_denom[1, :] / np.sqrt(self.D[1])
        self.poly_complex_denom[1, :] = (1j**1)*poly_real_denom[1, :]

        for i in range(2, self.n):
            self.v[i - 2] = 2 * np.dot(np.multiply(self.fr_dimless, poly_real_denom[i - 1, :]),
                                       np.multiply(poly_real_denom[i - 2, :], q))
            poly_real_denom[i, :] = np.multiply(self.fr_dimless, poly_real_denom[i - 1, :]) \
                                    - self.v[i - 2] * poly_real_denom[i - 2, :]
            self.D[i] = 2 * np.dot(np.multiply(q, poly_real_denom[i, :]), poly_real_denom[i, :])
            poly_real_denom[i, :] = poly_real_denom[i, :] / np.sqrt(self.D[i])
            self.poly_complex_denom[i, :] = (1j**i)*poly_real_denom[i, :]
        self.poly_complex_denom = np.transpose(self.poly_complex_denom)

    def calc_denom_coeff(self, order):
        P = self.poly_complex_nom[:, :order]
        T = np.multiply(self.poly_complex_denom[:, :order],
                        np.transpose(np.tile(self.H, (order, 1))))
        w = np.multiply(self.poly_complex_denom[:, order], self.H)

        X = -2 * np.real(np.dot(np.transpose(np.conjugate(P)), T))
        f = -2 * np.real(np.dot(np.transpose(np.conjugate(T)), w))

        self.b = np.dot(np.linalg.pinv(np.dot(np.transpose(X), X) - np.eye(order)),
                        np.dot(np.transpose(X), f))
        self.b = np.append(self.b, 1)

    def poles_comrade(self, order):
        C = np.zeros((order, order))

        C[order - 1, :] = np.divide(self.b[:order], self.D[order])
        C = C + np.diag(self.D[1:order], 1) + np.diag(self.v[:(order-1)], -1)
        poles, w = np.linalg.eig(C)

        self.omega_n.append(np.sort(np.multiply(np.absolute(poles), np.amax(self.fr))))

    def run_rpf(self):
        self.forsythe_nom()
        self.forsythe_denom()
        for i in range(self.min_fittig_order, self.fitting_order + 2, 2):
            self.calc_denom_coeff(i)
            self.poles_comrade(i)
        self.plot_rpf_result()

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

    def plot_rpf_result(self):
        plt.plot(self.fr, np.abs(self.H))
        for i in range(len(self.omega_n)):
            print(i)
            plt.plot(self.omega_n[i], np.multiply(i, np.ones(len(self.omega_n[i]))), 'ro')
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


#measurement1 = Fitting('theoretical_frf.mat')
measurement2 = Fitting('ford_meas.mat')
#measurement1.run_rpf()
measurement2.run_rpf()
