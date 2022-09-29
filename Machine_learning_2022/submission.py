import numpy as np
import scipy.misc
from typing import Callable, Tuple, Union, List

class f1:
    def __call__(self, x: float):
        a = x ** 2
        return a

    def grad(self, x: float):
        """
        Args :
            x : float
        Returns :
            float
        """

        g = 2*x
        return g

    def hess ( self , x : float ) :
        """
        Args :
            x : float
        Returns :
            float
        """
        h = 2
        return h


class f2:
    def __call__(self, x: float):
        a = np.sin(3 * np.sqrt(x ** 3) + 2) + x ** 2
        return a

    def grad(self, x: float):
        """
        Args :
            x : float
        Returns :
            float
        """
        # d = 0.5 * x * (9 * x / np.sqrt(x ** 3) + 4) * np.cos(3 * np.sqrt(x ** 3) + x ** 2 + 6)
        # return d
        def f(x):
            return np.sin(3 * np.sqrt(x ** 3) + 2) + x ** 2

        return scipy.misc.derivative(f, x, dx=1e-6)

    def hess(self, x: float):
        """
        Args :
            x : float
        Returns :
            float
        """
        def f(x):
            return np.sin(3 * np.sqrt(x ** 3) + 2) + x ** 2

        if x < 2:
            return scipy.misc.derivative(f, x, n = 2, dx=1e-6)
        else:
            return scipy.misc.derivative(f, x, n = 2, dx=1e-5)

class f3:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            float
        """
        f_3 = (x[0] - 3.3) ** 2 / 4 + (x[1] + 1.7) ** 2 / 15
        return f_3

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            np.ndarray of shape (2,)
        """

        def partial_derivative(func, var=0, point=[]):
            args = point[:]

            def wraps(x):
                args[var] = x
                return func(*args)

            return scipy.misc.derivative(wraps, point[var], dx=1e-5)

        def f(x, y):
            return (x - 3.3) ** 2 / 4 + (y + 1.7) ** 2 / 15

        return np.array([partial_derivative(f, 0, [x[0], x[1]]), partial_derivative(f, 1, [x[0], x[1]])])

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            np.ndarray of shape (2, 2)
        """

        def partial_derivative(func, var=0, point=[]):
            args = point[:]

            def wraps(x):
                args[var] = x
                return func(*args)

            return scipy.misc.derivative(wraps, point[var], n = 2, dx=1e-5)

        def f(x, y):
            return (x - 3.3) ** 2 / 4 + (y + 1.7) ** 2 / 15

        return np.array([[partial_derivative(f, 0, [x[0], x[1]]),0], [0,partial_derivative(f, 1, [x[0], x[1]])]])

class SquaredL2Norm:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            float
        """
        a = np.linalg.norm(x)**2
        return a

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n,)
        """

        def f(x):
            return np.linalg.norm(x) ** 2


        final = np.array([])
        for i in range(len(x)):
            final = np.append(final, scipy.misc.derivative(f, np.array(x[i]), n=1, dx=1e-6))
        return final

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n, n)
        """
        def f(x):
            return np.linalg.norm(x) ** 2

        final = np.array([])
        for i in range(len(x)):
            final = np.append(final, scipy.misc.derivative(f, np.array(x[i]), n=2, dx=1e-6))
        return 2 * np.eye(final.shape[0])


class Himmelblau:
    def __call__(self, x: np.ndarray):
        a = (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
        return a
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            float
        """


    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            numpy array of shape (2,)
        """
        def partial_derivative(func, var=0, point=[]):
            args = point[:]

            def wraps(x):
                args[var] = x
                return func(*args)

            return scipy.misc.derivative(wraps, point[var], dx=1e-5)

        def f(x, y):
            return (x**2+y-11)**2+(x+y**2-7)**2

        return np.array([partial_derivative(f, 0, [x[0], x[1]]), partial_derivative(f, 1, [x[0], x[1]])])

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            numpy array of shape (2, 2)
        """
        pass

class Rosenbrok:
    def __call__(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            float
        """
        assert x.shape[0] >= 2, "x.shape[0] должен быть >= 2"
        sum = 0
        for i in range(len(x) - 1):
            sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2

        return sum

    def grad(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            numpy array of shape (n,)
        """

        len_x = x.shape[0]

        assert len_x >= 2, "x.shape[0] должен быть >= 2"

        gr = []
        gr.append(100 * 2 * (x[1] - x[0] ** 2) * -2 * x[0] - 2 * (1 - x[0]))
        for i in np.array(range(len_x - 2)) + 1:
            gr_i = 100 * 2 * (x[i] - x[i - 1] ** 2) + 100 * 2 * (x[i + 1] - x[i] ** 2) * -2 * x[i] - 2 * (1 - x[i])
            gr.append(gr_i)

        gr.append(100 * 2 * (x[-1] - x[-2] ** 2))
        return np.array(gr)

    def hess(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            numpy array of shape (n, n)
        """

        len_x = x.shape[0]

        assert len_x >= 2, "x.shape[0] должен быть >= 2"

        fin = []
        el11 = -400 * (x[1] - x[0] ** 2) + 800 * x[0] ** 2 + 2
        fin.append(el11)
        for i in np.array(range(len_x - 2)) + 1:
            elii = 2 * 100 - 4 * 100 * (x[i + 1] - x[i] ** 2) + 8 * 100 * x[i] ** 2 + 2
            hes.append(elii)

        elnn = 200
        fin.append(elnn)

        hes_matrix = np.diag(fin) + np.diag(-400 * x[:-1], k=1) + np.diag(-400 * x[:-1], k=-1)
        return hes_matrix