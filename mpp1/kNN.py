import pandas as pd


def main():
    training_data_file = pd.read_csv('mpp_files/iristrain.csv')
    print(training_data_file.head())


if __name__ == '__main__':
    main()
