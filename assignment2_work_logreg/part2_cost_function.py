from tkinter.tix import X_REGION
import numpy as np
import pandas as pd
from part1_plot_data import DatasetReader

# ============ Part 2: Compute Cost and Gradient ============

dr = DatasetReader()
df1 = dr.get_dataset_1()
df2 = dr.get_dataset_2()


def h_theta(z):
    # hypothesis function for logreg (sigmoid)
    denominator = 1 + np.exp(-z)
    return 1/denominator

def test_h_theta_of_x():
    x = df1[['exam1score','exam2score']]
    print("x:\n", x)
    print("x shape:\n", x.shape)

    test_thetas = np.matrix([
        [.5, .5]
    ])

    z = x.dot(np.transpose(test_thetas))

    result = h_theta(z)
    print("result:\n", result)

def costFunction(theta, X, y):
    # log reg cost function J(theta)

    m = len(y)  # number of training examples
    print("number of training examples: ", m)

    # You need to return the following variables correctly
    J = 0  # cost
    grad = np.zeros(theta.size)
    print("grad: ", grad)

    '''
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta
    %
    % Note: grad should have the same dimensions as theta
    '''
    #------ computing cost function J(theta) -------
    
    X_dot_theta = X.dot(theta)
    #print("X dot theta:\n", X_dot_theta)

    h_theta_result = h_theta(X_dot_theta)
    #print("h_theta_result:\n", h_theta_result)
    
    #print("y function:\n", y)

    log_h_theta_result = np.log(h_theta_result)
    #print("log_h_theta_result:\n ", log_h_theta_result)

    summation_1st_clause = -y * np.log(h_theta_result)
    #print("summation_1st_clause:\n", summation_1st_clause)

    summation_2nd_clause = (1 - y) * np.log(1 - h_theta_result)
    #print("summation_2nd_clause:\n", summation_2nd_clause)

    J_sum_component = summation_1st_clause - summation_2nd_clause
    #print("J_sum_component:\n", J_sum_component)
    J_sum_component_after_sum = np.sum(J_sum_component)
    #print("J_sum_component_after_sum:\n", J_sum_component_after_sum)
    J_after_avg = (1/m) * J_sum_component_after_sum
    #print("J_after_avg:\n", J_after_avg)

    # ---------- computing gradient ------------------------

    grad_diff = h_theta_result - y
    print("grad_diff:\n", grad_diff)

    for j in range(len(grad)):
        x_j = X[[j]]
        #print("X[{}]:\n{}".format(j, x_j))
        x_j = x_j.rename(columns={j: 0}) # rename column to 0 for each partial derivative
        grad_diff_xj = grad_diff * x_j
        # print("grad_diff_xj:\n", grad_diff_xj)
        summed_component = np.sum(grad_diff_xj)
        grad[j] = (1/m) * summed_component

    return [J_after_avg, grad]


def part2_compute_cost_and_gradient():
    # print("df1.info:\n", df1.info)

    # Setup the data matrix appropriately, and add ones for the intercept term
    X = df1[['exam1score', 'exam2score']]
    # print("X:\n", X[:5])
    # print("X.shape: ", X.shape)
    y = df1[['admitted']]

    # Setup the data matrix appropriately
    [m, n] = X.shape  # m rows, n cols
    # print("m: {}, n: {}".format(m, n))

    # Add intercept term (column of ones) to X
    X.insert(0, 0, np.ones((m, 1)))  # insert in place, a col of 1's
    X = X.rename(columns={'exam1score': 1, 'exam2score': 2})
    print("X after inserting intercept:\n", X)
    print("X shape after inserting intercept: ", X.shape)

    y = y.rename(columns={'admitted': 0})
    print("y dataset:\n", y)

    initial_theta = np.zeros((n+1, 1))
    print("initial_theta\n", initial_theta)
    print("initial theta dimensions: ", initial_theta.shape)

    [cost, grad] = costFunction(initial_theta, X, y)


# test_h_theta_of_x()
part2_compute_cost_and_gradient()
