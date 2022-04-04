import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


class Perceptron:
    def __init__(self, language, theta=0, alpha=0.1, beta=0.1):
        self.lang = language
        self.W = np.random.rand(ord("z")-ord("a")+1)
        self.theta = theta
        self.alpha = alpha
        self.beta = beta

    def calc_linear_output(self, X):
        return np.dot(self.W, X)

    def calc_discrete_output(self, X):
        return 1 if self.calc_linear_output(X) >= self.theta else 0

    def test_all(self, langs_dir):
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        X = np.zeros(ord("z")-ord("a")+1)

        for root, dirs, files in os.walk(langs_dir):
            answer = 1 if root.split("\\")[-1] == self.lang else 0
            for file in files:
                delete_non_ascii_characters(os.path.join(root, file))
                X = count_distinct_letters(os.path.join(root, file))
                guess = self.calc_discrete_output(X)
                if guess == answer:
                    if guess == 1:
                        true_positives += 1
                    else:
                        true_negatives += 1
                else:
                    if guess == 1:
                        false_positives += 1
                    else:
                        false_negatives += 1

        return {"true_positives": true_positives, "true_negatives": true_negatives, "false_positives": false_positives,
                "false_negatives": false_negatives}

    def guess_linear(self, lang_doc):
        delete_non_ascii_characters(lang_doc)
        X = count_distinct_letters(lang_doc)
        return self.calc_linear_output(X)

    def train(self, langs_dir, epochs):
        X = np.zeros(ord("z")-ord("a")+1)

        for i in range(epochs):
            for root, dirs, files in os.walk(langs_dir):
                answer = 1 if root.split("\\")[-1] == self.lang else 0
                for file in files:
                    # print(os.path.join(root, file))
                    delete_non_ascii_characters(os.path.join(root, file))
                    X = count_distinct_letters(os.path.join(root, file))
                    guess = self.calc_discrete_output(X)

                    # if self.lang == "hiszpanski":
                    #     print(root.split("\\")[-1] + "\t\t" + str(guess), answer-guess, self.theta)

                    self.W += (answer - guess) * self.alpha * X
                    self.theta -= (answer - guess) * self.beta
            # normalize weights
            self.W /= np.linalg.norm(self.W, ord=2)

    def print_properties(self):
        print(self.lang)
        print("W:", self.W)
        print("theta:", self.theta)
        print()


# read all files in the directory recursively
def read_files(directory):
    res = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            res.append(os.path.join(root, file))
    return res


# create method deleting all non-ascii characters from file and save it
def delete_non_ascii_characters(file):
    parsed_path = "\\".join(file.split('\\')[:-1])+'\\'+'parsed_'+file.split("\\")[-1]
    with open(file, 'r', encoding='utf8') as infile, open(parsed_path, 'w', encoding='ascii') as outfile:
        for line in infile:
            try:
                outfile.write(''.join(i.lower() for i in line if ord('z') >= ord(i) >= ord('a')))
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
    return res


def plot_input_for_lang(lang, langs_dir):
    X = np.zeros(ord("z")-ord("a")+1)
    for root, dirs, files in os.walk(langs_dir):
        if root.split("\\")[-1] == lang:
            for file in files:
                delete_non_ascii_characters(os.path.join(root, file))
                X += count_distinct_letters(os.path.join(root, file))

    X /= np.linalg.norm(X, ord=2)
    plt.plot(X)


def guess_language(doc_path, perceptrons):
    p_guess = dict()
    for p in perceptrons:
        p_guess[p.lang] = p.guess_linear(doc_path)
    return p_guess.get(max(p_guess, key=p_guess.get))


plot_input_for_lang("polski", "langs")
plot_input_for_lang("niemiecki", "langs")
plot_input_for_lang("hiszpanski", "langs")
plt.legend(["polski", "niemiecki", "hiszpanski"])
plt.show()

p_pl = Perceptron("polski")
p_de = Perceptron("niemiecki")
p_es = Perceptron("hiszpanski", alpha=0.001, beta=0.001)

p_pl.train("langs", 1)
p_de.train("langs", 1)
p_es.train("langs", 10)

p_pl.print_properties()
p_de.print_properties()
p_es.print_properties()

print("pl:", p_pl.test_all("langs"))
print("de:", p_de.test_all("langs"))
print("es:", p_es.test_all("langs"))

# plot weights for each perceptron
plt.plot(p_pl.W, label="pl")
plt.plot(p_de.W, label="de")
plt.plot(p_es.W, label="es")
plt.legend()
plt.show()

# testowy
# plot_input_for_lang("jezykA", "testowy")
# plot_input_for_lang("jezykB", "testowy")
# plot_input_for_lang("jezykC", "testowy")
#
# plt.legend(["jezykA", "jezykB", "jezykC"])
# plt.show()
#
# p_a = Perceptron("jezykA")
# p_b = Perceptron("jezykB")
# p_c = Perceptron("jezykC")
#
# p_a.train("testowy", 200)
# p_b.train("testowy", 200)
# p_c.train("testowy", 200)
#
# p_a.print_properties()
# p_b.print_properties()
# p_c.print_properties()
#
# print(p_a.test_all("testowy"))
# print(p_b.test_all("testowy"))
# print(p_c.test_all("testowy"))
#
# # plot weights for each perceptron
# plt.plot(p_a.W, label="a")
# plt.plot(p_b.W, label="b")
# plt.plot(p_c.W, label="c")
# plt.legend()
# plt.show()
