import numpy as np
from typing import Callable, Tuple, Union, List


class f1:
    def __call__(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return x ** 2

    def grad(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return 2 * x

    def hess(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return 2


class f2:
    def __call__(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return np.sin(3 * x ** (3 / 2) + 2) + x ** 2

    def grad(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return np.cos(3 * x ** (3 / 2) + 2) * 4.5 * x ** 0.5 + 2 * x

    def hess(self, x: float):
        """
        Args:
            x: float
        Returns:
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
        return 0.25 * (x[0] - 3.3) ** 2 + (1 / 15) * (x[1] + 1.7) ** 2

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            np.ndarray of shape (2,)
        """
        return np.array([0.5 * x[0] - 1.65, (2 / 15) * x[1] + (1.7 * 2 / 15)])

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            np.ndarray of shape (2, 2)
        """
        return np.diag([0.5, 2 / 15])


class SquaredL2Norm:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            float
        """
        n = np.linalg.norm(x)
        return n ** 2

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n,)
        """
        return 2 * x

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n, n)
        """
        diag = np.repeat(2, len(x))
        return np.diag(diag)


class Himmelblau:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            float
        """
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            numpy array of shape (2,)
        """
        return np.array([2 * (x[0] ** 2 + x[1] - 11) * 2 * x[0] + 2 * (x[0] + x[1] ** 2 - 7),
                         2 * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7) * 2 * x[1]])

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

        summ = 0
        for i in range(len(x) - 1):
            summ_i = 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
            summ = summ + summ_i

        return summ

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

        elif method == 'newtone':
            if type(x_init) == float or type(x_init) == int:
                x_new = x_old - learning_rate(i) * (1 / (f.hess(x_old) + 1e-40)) * f.grad(x_old)
                x_old = x_new
                points_history_list.append(x_old)
                grad_history_list.append(f.grad(x_old))
                functions_history_list.append(f(x_old))
                if stopping_criteria == 'points' and np.linalg.norm(x_new - x_old) < tolerance:
                    # x_old = x_new
                    # points_history_list.append(x_old)
                    # grad_history_list.append(f.grad(x_old))
                    break
                elif stopping_criteria == 'function' and np.linalg.norm(f(x_new) - func(x_old)) < tolerance:
                    # x_old = x_new
                    # points_history_list.append(x_old)
                    # grad_history_list.append(f.grad(x_old))
                    break
                elif stopping_criteria == 'gradient' and np.linalg.norm(f.grad(x_new)) < tolerance:
                    # x_old = x_new
                    # points_history_list.append(x_old)
                    # grad_history_list.append(f.grad(x_old))
                    break


            elif type(x_init) == np.ndarray:
                x_new = x_old - learning_rate(i) * np.dot(np.linalg.inv(f.hess(x_old)), f.grad(x_old))
                x_old = x_new
                points_history_list.append(x_old)
                grad_history_list.append(f.grad(x_old))
                functions_history_list.append(f(x_old))
                if stopping_criteria == 'points' and np.linalg.norm(x_new - x_old) < tolerance:
                    # x_old = x_new
                    # points_history_list.append(x_old)
                    # grad_history_list.append(f.grad(x_old))
                    break
                elif stopping_criteria == 'function' and np.linalg.norm(f(x_new) - func(x_old)) < tolerance:
                    # x_old = x_new
                    # points_history_list.append(x_old)
                    # grad_history_list.append(f.grad(x_old))
                    break
                elif stopping_criteria == 'gradient' and np.linalg.norm(f.grad(x_new)) < tolerance:
                    # x_old = x_new
                    # points_history_list.append(x_old)
                    # grad_history_list.append(f.grad(x_old))
                    break

    x_opt = points_history_list[-1]

    return (x_opt, points_history_list, functions_history_list, grad_history_list)