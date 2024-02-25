"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import math


class Logarithmic:
    def __init__(self, init_T: float = 1e3, min_val: float = 1e-5) -> None:
        self.t = 0
        self.min_val = min_val
        self.c = init_T * math.log(2)

    def next_T(self):
        self.t += 1
        return math.max(self.c / math.log(1 + self.t), self.min_val)


class Geometric:
    def __init__(
        self, init_T: float = 1e3, a: float = 0.8, min_val: float = 1e-5
    ) -> None:
        self.min_val = min_val
        self.T = init_T
        self.a = a

    def next_T(self):
        self.T = self.a * self.T
        return math.max(self.T, self.min_val)


class Linear:
    def __init__(
        self, init_T: float = 1000, step: float = 10, min_val: float = 1e-5
    ) -> None:
        self.T = init_T
        self.step = step

    def next_T(self):
        self.T = self.T - self.step
        return math.max(self.T, self.min_val)


class ArithmeticGeometric:
    def __init__(self, init_T: float = 1e3, a: float = 0.8, b: float = 1e-5) -> None:
        self.T = init_T
        self.a = a
        self.b = b

    def next_T(self):
        self.T = self.a * self.T + self.b
        return self.T
