import numpy as np
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
        d = 0.5 * x * (9 * x / np.sqrt(x ** 3) + 4) * np.cos(3 * np.sqrt(x ** 3) + x ** 2 + 6)
        return d

    # def hess(self, x: float):
    #     """
    #     Args :
    #         x : float
    #     Returns :
    #         float
    #     """
    #     ...
    #
    #     return
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
        pass

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            np.ndarray of shape (2, 2)
        """
        pass
class SquaredL2Norm:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            float
        """
        a = np.linalg.norm((np.array([-0.83, 2.2])))**2
        return a

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n,)
        """
        pass

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n, n)
        """
        pass


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
        pass

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            numpy array of shape (2,)
        """
        pass

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            numpy array of shape (2, 2)
        """
        pass