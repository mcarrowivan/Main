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

        g = lambda x: (x ** 2)
        grad1 = nd.Gradient(g)([x])
        return grad1

#     def hess ( self , x : float ) :
#     """
#     Args :
#         x : float
#     Returns :
#         float
#     """
#     ...
#     return