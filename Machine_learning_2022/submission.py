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
        return -np.sin(3 * x ** (3 / 2) + 2) * 20.25 * x + np.cos(3 * x ** (3 / 2) + 2) * 2.25 / (x ** 0.5) + 2

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
        return np.array(
            [[12 * x[0] ** 2 + 4 * x[1] - 42, 4 * (x[0] + x[1])], [4 * (x[0] + x[1]), 4 * x[0] + 12 * x[1] ** 2 - 26]])


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
        #
        # fin = []
        # el11 = -400 * (x[1] - x[0] ** 2) + 800 * x[0] ** 2 + 2
        # fin.append(el11)
        # for i in np.array(range(len_x - 2)) + 1:
        #     elii = 2 * 100 - 4 * 100 * (x[i + 1] - x[i] ** 2) + 8 * 100 * x[i] ** 2 + 2
        #     hes.append(elii)
        #
        # elnn = 200
        # fin.append(elnn)
        #
        # hes_matrix = np.diag(fin) + np.diag(-400 * x[:-1], k=1) + np.diag(-400 * x[:-1], k=-1)
        # return hes_matrix

        hes = []
        x_11 = -4 * 100 * (x[1] - x[0] ** 2) + 8 * 100 * x[0] ** 2 + 2
        hes.append(x_11)
        for i in np.array(range(len_x - 2)) + 1:
            x_ii = 2 * 100 - 4 * 100 * (x[i + 1] - x[i] ** 2) + 8 * 100 * x[i] ** 2 + 2
            hes.append(x_ii)

        x_nn = 200
        hes.append(x_nn)

        hes_matrix = np.diag(hes) + np.diag(-400 * x[:-1], k=1) + np.diag(-400 * x[:-1], k=-1)
        return hes_matrix
#
# def minimize(
#         func: Callable,
#         x_init: np.ndarray,
#         learning_rate: Callable = lambda x: 0.1,
#         method: str = 'gd',
#         max_iter: int = 10_000,
#         stopping_criteria: str = 'function',
#         tolerance: float = 1e-2,
# ) -> Tuple:
#     """
#     Args:
#         func: функция, у которой необходимо найти минимум (объект класса, который только что написали)
#             (у него должны быть методы: __call__, grad, hess)
#         x_init: начальная точка
#         learning_rate: коэффициент перед направлением спуска
#         method:
#             "gd" - Градиентный спуск
#             "newtone" - Метод Ньютона
#         max_iter: максимально возможное число итераций для алгоритма
#         stopping_criteria: когда останавливать алгоритм
#             'points' - остановка по норме разности точек на соседних итерациях
#             'function' - остановка по норме разности значений функции на соседних итерациях
#             'gradient' - остановка по норме градиента функции
#         tolerance: c какой точностью искать решение (участвует в критерии остановки)
#     Returns:
#         x_opt: найденная точка локального минимума
#         list_try: (list) список с историей точек
#         list_f: (list) список с историей значений функции
#         list_grad: (list) список с исторей значений градиентов функции
#     """
#
#     assert max_iter > 0, 'max_iter должен быть > 0'
#     assert method in ['gd', 'newtone'], 'method can be "gd" or "newtone"!'
#     assert stopping_criteria in ['points', 'function', 'gradient'], \
#         'stopping_criteria can be "points", "function" or "gradient"!'
#
#     f = func
#     list_try = [x_init]
#     list_f = [f(x_init)]
#     list_grad = [f.grad(x_init)]
#
#     for i in range(max_iter):
#         if method == 'gd':
#             x_new = x_init - learning_rate(i) * f.grad(x_init)
#
#             if stopping_criteria == 'points' and np.linalg.norm(x_new - x_init) < tolerance:
#                 break
#             elif stopping_criteria == 'function' and np.linalg.norm(f(x_new) - f(x_init)) < tolerance:
#                 break
#             elif stopping_criteria == 'gradient' and np.linalg.norm(f.grad(x_new)) < tolerance:
#                 break
#             x_init = x_new
#             list_try.append(x_init)
#             list_grad.append(f.grad(x_init))
#             list_f.append(f(x_init))
#






def minimize(
        func: Callable,
        x_init: np.ndarray,
        learning_rate: Callable = lambda x: 0.1,
        method: str = 'gd',
        max_iter: int = 10_000,
        stopping_criteria: str = 'function',
        tolerance: float = 1e-2,
) -> Tuple:
    """
    Args:
        func: функция, у которой необходимо найти минимум (объект класса, который только что написали)
            (у него должны быть методы: __call__, grad, hess)
        x_init: начальная точка
        learning_rate: коэффициент перед направлением спуска
        method:
            "gd" - Градиентный спуск
            "newtone" - Метод Ньютона
        max_iter: максимально возможное число итераций для алгоритма
        stopping_criteria: когда останавливать алгоритм
            'points' - остановка по норме разности точек на соседних итерациях
            'function' - остановка по норме разности значений функции на соседних итерациях
            'gradient' - остановка по норме градиента функции
        tolerance: c какой точностью искать решение (участвует в критерии остановки)
    Returns:
        x_opt: найденная точка локального минимума
        points_history_list: (list) список с историей точек
        functions_history_list: (list) список с историей значений функции
        grad_history_list: (list) список с исторей значений градиентов функции
    """

    assert max_iter > 0, 'max_iter должен быть > 0'
    assert method in ['gd', 'newtone'], 'method can be "gd" or "newtone"!'
    assert stopping_criteria in ['points', 'function', 'gradient'], \
        'stopping_criteria can be "points", "function" or "gradient"!'

    f = func
    x_old = x_init
    points_history_list = [x_old]
    functions_history_list = [f(x_old)]
    grad_history_list = [f.grad(x_old)]

    for i in range(max_iter):
        if method == 'gd':
            x_new = x_old - learning_rate(i) * f.grad(x_old)

            if stopping_criteria == 'points' and np.linalg.norm(x_new - x_old) < tolerance:
                break
            elif stopping_criteria == 'function' and np.linalg.norm(f(x_new) - f(x_old)) < tolerance:
                break
            elif stopping_criteria == 'gradient' and np.linalg.norm(f.grad(x_new)) < tolerance:
                break
            x_old = x_new
            points_history_list.append(x_old)
            grad_history_list.append(f.grad(x_old))
            functions_history_list.append(f(x_old))
