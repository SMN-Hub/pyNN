from enum import Enum
from numpy import random as rd


class HyperParameter:
    pass


class LearningRate(HyperParameter):
    def __init__(self, distribution):
        self.distribution = distribution


class HyperParameterTuner:
    def __init__(self):
        pass

    def fit(self, X, Y):
