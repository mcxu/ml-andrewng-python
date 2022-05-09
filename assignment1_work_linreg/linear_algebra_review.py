import numpy as np


def test1():
    m = np.matrix([[1], [4], [2]])
    print(3*m)

# test1()


def test1():
    x = np.matrix([
        [1, 2, 3],
        [4, 5, 6]
    ])

    # this is already transposed because it is a vertical matrix
    theta = np.ones((3, 1))
    print("theta:\n", theta)

    result = np.dot(x, theta)
    print("result: ", result)


def test2():
    # matrix matrix multiplication
    a = np.matrix([
        [1, 3],
        [2, 5]
    ])

    b = np.matrix([
        [0, 1],
        [3, 2]
    ])

    prod = a*b
    print("matrix product: ", prod)


def test3():
    X = np.matrix([
        [1, 2104],
        [1, 1416],
        [1, 1574],
        [1, 852]
    ])

    theta = np.matrix([
        [-40, 0.25],
        [200, 0.1],
        [-150, 0.4]
    ])

    theta_transpose = np.transpose(theta)
    print("theta transpose:\n", theta_transpose)

    h_theta = X*theta_transpose
    print("h_theta:\n", h_theta)

    """
    [[486.  410.4 691.6]
    [314.  341.6 416.4]
    [353.5 357.4 479.6]
    [173.  285.2 190.8]]
    """


def matrix_multiplication_properties():
    A = np.matrix([
        [1, 2],
        [4, 5]
    ])
    print("A:\n", A)

    B = np.matrix([
        [1, 1],
        [0, 2]
    ])
    print("B:\n", B)

    I = np.eye(2)
    print("I:\n", I)

    IA = I*A
    print("IA:\n", IA)

    AI = A*I
    print("AI:\n", AI)

    AB = A*B
    print("AB:\n", AB)

    BA = B*A
    print("BA:\n", BA)


def inverse_and_transpose():
    A = np.matrix([
        [1, 2, 0],
        [0, 5, 6],
        [7, 0, 9]
    ])
    A_ij = np.transpose(A)
    print("A_ij\n", A_ij)

# test1()
# test2()
# test3()
# matrix_multiplication_properties()
# inverse_and_transpose()
