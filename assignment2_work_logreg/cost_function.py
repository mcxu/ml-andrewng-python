import numpy as np
from plot_data import DatasetReader


class CostFunction:
    def __init__(self) -> None:
        self.dr = DatasetReader()
        self.ds1 = self.dr.read_dataset_1()
        self.ds2 = self.dr.read_dataset_2()

    def h_theta_of_x(self, x, theta):
        denominator = 1 + np.exp(-1 * theta.transpose().dot(x))
        return 1/denominator

    def cost_function_J_theta(self, X, y, theta):
        m = len(y)
        J = 0
        grad = np.zeros(len(theta))

    def test_h_theta_of_x(self):
        result = self.h_theta_of_x(x, theta)
