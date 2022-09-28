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
        a = numpy.sin(3 * numpy.sqrt(x ** 3) + 2) + x ** 2
        return a

    def grad(self, x: float):
        """
        Args :
            x : float
        Returns :
            float
        """
        d = 0.5 * x * (9 * x / numpy.sqrt(x ** 3) + 4) * numpy.cos(3 * numpy.sqrt(x ** 3) + x ** 2 + 6)
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