import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


class Perceptron:
    def __init__(self, language, theta=0, alpha=0.1, beta=0.1 ):
        # self.X = np.zeros(ord("z")-ord("a")+1) # without theta
        # self.X = np.zeros(ord("z")-ord("a")+2) # with theta
        # self.X[-1] = -1
        self.lang = language

        self.W = np.random.rand(ord("z")-ord("a")+1) # without theta
        # self.W = np.random.rand(ord("z")-ord("a")+2) # with theta
        # self.W[-1] = theta

        self.theta = theta
        self.alpha = alpha
        self.beta = beta

    def calc_linear_output(self, X):
        return np.dot(self.W, X)

    def calc_discrete_output(self, X):
        return 1 if self.calc_linear_output(X) >= self.theta else 0

    def train(self, langs_dir, epochs):
        guess = 0
        answer = 1 if langs_dir.split("/")[-1] == self.lang else 0

        X = np.zeros(ord("z")-ord("a")+1)

        for i in range(epochs):
            for root, dirs, files in os.walk(langs_dir):
                for file in files:
                    # print(os.path.join(root, file))
                    delete_non_ascii_characters(os.path.join(root, file))
                    X = count_distinct_letters(os.path.join(root, file))
                    guess = self.calc_discrete_output(X)

            self.W += (guess - answer) * self.alpha * X
            # normalize weights
            self.W /= np.linalg.norm(self.W)

            self.theta -= (guess - answer) * self.beta
        # print Perceptron properties
        print("W:", self.W)
        print("theta:", self.theta)


# read all files in the directory recursively
def read_files(directory):
    res = []
    # langs = os.listdir(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            res.append(os.path.join(root, file))
    return res


# create method deleting all non-ascii characters from file and save it
def delete_non_ascii_characters(file):
    parsed_path = "\\".join(file.split('\\')[:-1])+'\\'+'parsed_'+file.split("\\")[-1]

    with open(file, 'r', encoding='utf8') as infile, open(parsed_path, 'w',encoding='ascii') as outfile:
        for line in infile:
            # print(line)
            try:
                outfile.write(''.join(i.lower() for i in line if ord('z') >= ord(i) >= ord('a') or ord('Z') >= ord(i) >= ord('A')))
            except UnicodeDecodeError:
                pass
    os.remove(file)
    os.rename(parsed_path, file)


# count distinct letters in file and return it as a list
def count_distinct_letters(file):
    res = np.zeros(ord("z")-ord("a")+1)
    with open(file, 'r') as f:
        for line in f:
            for char in line:
                res[ord(char.lower())-ord("a")] += 1
                # res[(ord(char)-ord('a')) % len(res)] += 1
    return res


# print(read_files("langs"))

p = Perceptron("czeski")
p.train("langs", 50)

# np.norm(p.W)
# numpy second norm of vector:
print(np.linalg.norm(p.W, ord=2))
