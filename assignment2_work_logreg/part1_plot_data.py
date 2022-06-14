from ast import AsyncFunctionDef
import numpy as np
import pandas as pd
import plotly.express as px


class DatasetReader:
    def __init__(self) -> None:
        pass

    def get_dataset_1(self):
        df = pd.read_csv('assignment2_work_logreg/ex2data1.txt',
                         names=['exam1score', 'exam2score', 'admitted'])
        return df

    def get_dataset_2(self):
        return pd.read_csv('assignment2_work_logreg/ex2data2.txt',
                           names=['exam1score', 'exam2score', 'admitted'])

    def plot_dataset_1(self):
        df1 = self.get_dataset_1()
        # df.info()
        # print("head----------------")
        # print(df.head())
        fig = px.scatter(df1,
                         x='exam1score',
                         y='exam2score',
                         color='admitted',
                         color_continuous_scale=['red', 'blue'])
        fig.update_layout(title_text='Exam 2 scores vs exam 1 scores')
        fig.show()

    def plot_dataset_2(self):
        df2 = self.get_dataset_2()
        fig = px.scatter(df2,
                         x='exam1score', y='exam2score', color='admitted',
                         color_continuous_scale=['red', 'blue'])
        fig.update_layout(title_text='Exam 2 scores vs exam 1 scores')
        fig.show()


if __name__ == '__main__':
    print("in main")

    dr = DatasetReader()
    # read_dataset_1()
    # read_dataset_2()
    # plot_dataset_1()
    dr.plot_dataset_2()
