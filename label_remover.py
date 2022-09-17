import os
import sys
import numpy as np
np.random.seed(31415)
import itertools

BASE_DIR = "/content/gdrive/My Drive/PunjabiNER/"
GLOVE_DIR = BASE_DIR + "/glove/"
FASTTEXT_DIR = BASE_DIR + "/fasttext/"
MAX_LEN = 64
USE_PRETRAINED_EMB = False
USE_BIDIRECTIONAL = True

my_stopwords = [":", "‘", "’", '","', "...", "+", "(", ")", "", "…"]


def remove_mystopwords(sentence):
    tokens = sentence.split(" ")
    tokens_filtered = [word for word in tokens if not word in my_stopwords]
    return (" ").join(tokens_filtered)


# --------------------------------------------------------
# list of lists
root_path = BASE_DIR + "/Dataset/"
X, Y = [], []
MAX_LEN = 64
n = 4
new_sentences = []
new_labels = []

file_object = open(root_path + "Raw_Data_punjabi.txt", "a", encoding="utf-8")
with open(root_path + "Datasettest.txt", "r") as f:
    content = f.read()
# print(content)
# sentences are separated by \n\n
sentences = content.split("\n\n")
# print(sentences)
for sentence in sentences:
    tokens = sentence.split("\n")
    # print(tokens)
    for token in tokens:
        tuple = token.split("\t")
        urdu_tokens = tuple[0].split(" ")
        for word in urdu_tokens:
            filtered_text = remove_mystopwords(word)
            if word in filtered_text:
                print(word)
                file_object.write(word + " ")
    file_object.write("\n")
