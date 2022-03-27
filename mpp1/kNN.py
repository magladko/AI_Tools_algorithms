import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    train_df = pd.read_csv('mpp_files/iristrain.csv')
    # print(train_df.head())

    test_df = pd.read_csv('mpp_files/iristest.csv')
    # test_df.head()

    train_plots = sns.pairplot(data=train_df.drop('nr', axis=1), hue='Species')
    train_plots.fig.suptitle('Pair Plot of training dataset', y=1.08)
    plt.show()


if __name__ == '__main__':
    main()
