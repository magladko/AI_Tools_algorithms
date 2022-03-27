import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns


# create Perceptron class with the following attributes: weights, alpha, theta, and epochs
class Perceptron:
    def __init__(self, alpha: float, beta: float, theta: float, output_possibilities: [str]):
        self.errors = []
        self.data_size = 0
        self.weights = []
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        if len(output_possibilities) != 2:
            raise ValueError("Output possibilities must be a list of length 2")
        self.output_possibilities = output_possibilities

    # create a method to train the perceptron
    def train(self, inputs: [[int]], targets: [str], epochs: int):
        # print(inputs)
        # print(targets)
        # print()

        self.weights = np.random.rand(len(inputs[0]))
        print(perceptron.weights)
        for i in range(epochs):
            for j in range(len(inputs)):
                x = inputs[j]

                # numpy find index of value targets[j] in self.output_possibilities
                y = np.where(self.output_possibilities == targets[j])[0][0]

                prediction = self.predict(x)

                self.weights = normalize(self.weights + self.alpha * (prediction - y) * x)

                self.theta = self.theta - (prediction - y) * self.beta

        return self.weights, self.theta

    # create a method to test the perceptron
    def test(self, inputs: [[int]], targets: [str] = None):
        print(inputs)
        print(targets)
        print()
        self.data_size = len(inputs)
        predictions = []
        for i in range(len(inputs)):
            x = inputs[i]
            prediction = self.predict(x)

            if targets is not None and prediction != targets[i]:
                print("Prediction: {}, Actual: {}".format(self.output_possibilities[prediction], targets[i]))
                self.errors.append(inputs[i])

            predictions.append(self.predict(x))

        print(predictions)
        return predictions

    # create a method to predict the output of the perceptron
    def predict(self, inputs):
        return 0 if scalar_multiply(inputs, self.weights) < self.theta else 1

    # calculate the error rate of the perceptron
    def error_rate(self):
        return len(self.errors) / self.data_size

    # create a method to plot the error rate of the perceptron
    def plot_error_rate(self):
        plt.plot(range(len(self.errors)), self.errors)
        plt.xlabel("Epoch")
        plt.ylabel("Error Rate")
        plt.show()

    # plot separation line from weights with data points
    def plot_separation_line(self, inputs: [[int]], targets: [str]):
        # plot data points
        for i in range(len(inputs)):
            x = inputs[i]
            y = targets[i]
            # convert y to color
            color = "red" if y == self.output_possibilities[0] else "blue"
            plt.scatter(x[0], x[1], c=color)

        # plot separation line
        x = np.linspace(0, 1, 100)
        y = -(self.weights[0] * x + self.theta) / self.weights[1]
        plt.plot(x, y)
        plt.show()

    # # create a method to plot weights and data
    # def plot_weights(self, inputs: [[int]], targets: [str]):
    #     # plot hyperplane from weights and data
    #     # plot the weights
    #     plt.plot(range(len(self.weights)), self.weights)
    #     # plot the data
    #     sns.scatterplot(x=0, y=1, data=pd.DataFrame(inputs), hue=targets)
    #     # show sns plot
    #     plt.show()


def scalar_multiply(vector1, vector2):
    return sum(i * j for i, j in zip(vector1, vector2))


# normalize vector using second norm without numpy library
def normalize(vector):
    norm = 0
    for i in range(len(vector)):
        norm += vector[i] ** 2
    norm = norm ** 0.5
    for i in range(len(vector)):
        vector[i] = vector[i] / norm
    return vector


if __name__ == '__main__':
    train_df = pd.read_csv('iristrain.csv')
    test_df = pd.read_csv('iristest.csv')

    # tested_species = input("Enter the species you want to test: ")
    tested_species = ["setosa", "versicolor"]

    # if tested_species == "":
    #     tested_species = ["setosa", "versicolor"]
    # else:
    #     tested_species = re.split(r'\s+', tested_species)

    train_df = train_df[train_df['Species'].isin(tested_species)]
    test_df = test_df[test_df['Species'].isin(tested_species)]

    perceptron = Perceptron(alpha=0.5, beta=0.5, theta=1, output_possibilities=train_df.iloc[:, -1].unique())

    # print(train_df.iloc[:, -1].values)

    perceptron.train(inputs=train_df.iloc[:, 1:-1].values, targets=train_df.iloc[:, -1].values, epochs=1000)
    print(perceptron.weights)

    perceptron.test(inputs=test_df.iloc[:, 1:-1].values, targets=test_df.iloc[:, -1].values)

    print(perceptron.error_rate())
    print(perceptron.errors)

    # perceptron.plot_error_rate()
    perceptron.plot_separation_line(inputs=test_df.iloc[:, 1:-1].values, targets=test_df.iloc[:, -1].values)
    # perceptron.plot_weights(inputs=train_df.iloc[:, 1:-1].values, targets=train_df.iloc[:, -1].values)
