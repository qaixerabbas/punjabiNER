from google.colab import drive

drive.mount("/content/gdrive")

import os

import sys
import numpy as np

# keras >= 2.0.3
from keras.models import Sequential
from keras.layers import (
    Bidirectional,
    LSTM,
    Activation,
    Dense,
    Dropout,
    TimeDistributed,
    Embedding,
)
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
import itertools

np.random.seed(31415)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

BASE_DIR = "."
GLOVE_DIR = BASE_DIR + "/glove/"
MAX_LEN = 64
USE_PRETRAINED_EMB = False
USE_BIDIRECTIONAL = True
iopub_data_rate_limit = 10000000000

# --------------------------------------------------------
# list of lists
root_path = "gdrive/My Drive/PunjabiNER/Dataset/"
X, Y = [], []
MAX_LEN = 64
with open(root_path + "Datasettest.txt", "r") as f:
    content = f.read()

# sentences are separated by \n\n
sentences = content.split("\n\n")
# print(sentences[0])
for sentence in sentences:
    tokens = sentence.split("\n")
    # print(tokens)
    x, y = [], []
    for token in tokens:
        tuple = token.split("\t")
        x.append(tuple[0])
        y.append(tuple[1])

    # ignore sentences with more than MAX_LEN words
    if (len(x) > MAX_LEN) or (len(x) <= 1):
        continue

    X.append(x)
    Y.append(y)

# --------------------------------------------------------

# take a list of values and hot-encode
# returns list of lists
def encode(arr, num_labels):
    ret = []
    for item in arr:
        a = np.zeros(num_labels, dtype=np.int32)
        a[item] = 1
        ret.append(a)
    return ret


# build vocabulary of words and entities
words = set(itertools.chain(*X))
entities = set(itertools.chain(*Y))

# reserve index 0 for padding/masking
idx2word = dict((i + 1, v) for i, v in enumerate(words))
word2idx = dict((v, i + 1) for i, v in enumerate(words))

idx2entity = dict((i + 1, v) for i, v in enumerate(entities))
entity2idx = dict((v, i + 1) for i, v in enumerate(entities))

num_entities = len(entity2idx) + 1
num_words = len(word2idx) + 1

print("num_words = {0}, num_entities = {1}".format(num_words, num_entities))

# index encoder
X_enc = list(map(lambda x: [word2idx[wx] for wx in x], X))
Y_enc = list(map(lambda y: [entity2idx[wy] for wy in y], Y))


# one-hot encoder
Y_oh_enc = list(map(lambda y: encode(y, num_labels=num_entities), Y_enc))


# pad and truncate
X_all = pad_sequences(X_enc, MAX_LEN)  # has shape (#samples, 64)
Y_all = pad_sequences(Y_oh_enc, MAX_LEN)  # has shape (#samples, 64, 6)


# construct training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X_all, Y_all, test_size=10 * 32, train_size=40 * 32, random_state=42
)
print(
    "Training and testing tensor shapes:",
    X_train.shape,
    X_test.shape,
    Y_train.shape,
    Y_test.shape,
)


embedding_size = 100
num_cells = 64
batch_size = 32
num_epochs = 10

# construct the NN model
model = Sequential()

# embed into vector space of dimension embedding_size
# input value 0 is a special "padding" value that should be masked out

# initialize with pretrained wordvectors
if USE_PRETRAINED_EMB:
    print("Indexing word vectors.")
    embeddings_index = {}
    try:
        f = open(os.path.join(GLOVE_DIR, "glove.6B.100d.txt"))
    except:
        print(
            "Cannot open Glove file. Please download from http://nlp.stanford.edu/data/glove.6B.zip"
        )
        sys.exit(0)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((num_words, embedding_size))
    for word, i in word2idx.items():
        # lowercase the words
        embedding_vector = embeddings_index.get(word.lower())
        if embedding_vector is not None:
            # words not found in embedding index will be zero
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(
        len(word2idx) + 1,
        embedding_size,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=True,
        mask_zero=True,
    )
    model.add(embedding_layer)
else:
    # initialize with random vectors
    model.add(
        Embedding(
            len(word2idx) + 1, embedding_size, input_length=MAX_LEN, mask_zero=True
        )
    )

# add LSTM layer; return all sequences for the output
if USE_BIDIRECTIONAL:
    model.add(Bidirectional(LSTM(num_cells, return_sequences=True)))
else:
    model.add(LSTM(num_cells, return_sequences=True))

# some regularization
model.add(Dropout(0.2))

# applies a same Dense (fully-connected) operation at every timestep
model.add(TimeDistributed(Dense(len(entity2idx) + 1)))
model.add(Activation("softmax"))

# use multi-class loss function and adaptive gradient descent optimizer
model.compile(optimizer="adam", loss="categorical_crossentropy")
print(model.summary())

# train the model
model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    nb_epoch=num_epochs,
    validation_data=(X_test, Y_test),
)

# test the model
Y_test_pred = model.predict_classes(X_test)

# to generate confusion matrix first remove the zero masked inputs and outputs
def clean(y_pred, y_gnd):
    coords = [np.where(y > 0)[0][0] for y in y_gnd]
    y_pred_unpad = [y[coord:] for coord, y in zip(coords, y_pred)]
    y_gnd_unpad = [y[coord:] for coord, y in zip(coords, y_gnd)]
    return y_pred_unpad, y_gnd_unpad


y_p_u, y_g_u = clean(Y_test_pred, Y_test.argmax(2))

# flatten to single array with class labels
y_p_u = list(itertools.chain(*y_p_u))  # predicted
y_g_u = list(itertools.chain(*y_g_u))  # ground

print("\nTesting accuracy (all entities):", accuracy_score(y_g_u, y_p_u))
print("\nconfusion matrix:")
print(confusion_matrix(y_g_u, y_p_u))
precision, recall, fscore, support = precision_recall_fscore_support(y_g_u, y_p_u)
print("class | precision,recall,fscore,support")
for tag, i in entity2idx.items():
    if tag == "O":
        continue
    print(
        "{0} | {1:1.2f}\t{2:1.2f}\t{3:1.2f}\t{4}".format(
            tag, precision[i - 1], recall[i - 1], fscore[i - 1], support[i - 1]
        )
    )

# visualize output for some random inputs
for _ in range(5):
    vis_idx = np.random.randint(1000)
    Y_vis_pred = model.predict_classes(X_all[vis_idx].reshape(1, MAX_LEN))
    y_vis_u, _ = clean(Y_vis_pred, [Y_all[vis_idx].argmax(1)])
    y_vis_u_l = [idx2entity[val] for val in y_vis_u[0]]
    print("\nInput sentence: {}".format(" ".join(X[vis_idx])))
    print("Predict entities: {}".format(" ".join(y_vis_u_l)))
    print("Correct entities: {}".format(" ".join(Y[vis_idx])))
