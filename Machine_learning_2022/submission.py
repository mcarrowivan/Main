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