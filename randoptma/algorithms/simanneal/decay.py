"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

class ArithmeticGeometric:
    def __init__(self, init_T: float = 1000, a: float = 0.8, b: float = 0.01) -> None:
        self.T = init_T
        self.a = a
        self.b = b

    def next_T(self):
        self.T = self.a * self.T + self.b
        return self.T


class Linear:
    def __init__(self, init_T: float = 1000, step: float = 10) -> None:
        self.T = init_T
        self.step = step

    def next_T(self):
        self.T = self.T - self.step
        return self.T
