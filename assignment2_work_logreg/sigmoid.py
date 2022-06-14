import numpy as np
from part1_plot_data import DatasetReader


def sigmoid(z):
    # sigmoid
    denominator = 1 + np.exp(-1 * z)
    return 1/denominator


def test1():
    z = np.zeros((10, 1))
    result = sigmoid(z)
    print("result: ", result)


def test2():
    dataset_reader = DatasetReader()
    df = dataset_reader.get_dataset_2()
    print("df: ", df)
    df_col0 = df['exam1score']
    results = sigmoid(df_col0)
    print("results: ", results)


# test2()
test1()
