from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, Flatten
from keras.layers import LSTM, Convolution1D, MaxPooling1D, Bidirectional, TimeDistributed, GRU, Input, merge, AveragePooling1D, SimpleRNN, TimeDistributedDense
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.decomposition import PCA
import numpy as np
import random
import sys
import csv
import os
import h5py

embeddings_path = "glove.840B.300d-char.txt"
embedding_dim = 300
batch_size = 128
use_pca = False
lr = 0.001
lr_decay = 1e-4
maxlen = 40
device = 'cpu'

text = open('magic_cards.txt').read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters

step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))


print('Vectorization...')
X = np.zeros((len(sentences), maxlen), dtype=np.int)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_indices[char]
    y[i, char_indices[next_chars[i]]] = 1


# test code to sample on 1% for functional model testing

def random_subset(X, y, p=0.1):

    idx = np.random.randint(X.shape[0], size=int(X.shape[0] * p))
    X = X[idx, :]
    y = y[idx]
    return (X, y)


# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
print('Processing pretrained character embeds...')
embedding_vectors = {}
with open(embeddings_path, 'r') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        char = line_split[0]
        embedding_vectors[char] = vec

embedding_matrix = np.zeros((len(chars), 300))
for char, i in char_indices.items():
    embedding_vector = embedding_vectors.get(char)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Use PCA from sklearn to reduce 300D -> 50D
if use_pca:
    pca = PCA(n_components=embedding_dim)
    pca.fit(embedding_matrix)
    embedding_matrix_pca = np.array(pca.transform(embedding_matrix))
    print (embedding_matrix_pca)
    print (embedding_matrix_pca.shape)


print('Build model...')
main_input = Input(shape=(maxlen,))
embedding_layer = Embedding(
    len(chars), embedding_dim, input_length=maxlen, weights=[embedding_matrix_pca] if use_pca else [embedding_matrix])
embedded = embedding_layer(main_input)

lstm = LSTM(128, consume_less=device)(embedded)
lstm = BatchNormalization()(lstm)

hidden = Dense(2048)(lstm)
hidden = Activation('relu')(hidden)
hidden = BatchNormalization()(hidden)

main_output = Dense(len(chars))(hidden)
main_output = Activation('softmax')(main_output)

model = Model(input=main_input, output=main_output)

optimizer = Adam(lr=lr, decay=lr_decay)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    if diversity > 0:
        preds = np.log(preds + 1e-6) / temperature
        exp_preds = np.exp(preds)
    else:
        exp_preds = preds
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


if not os.path.exists('output'):
    os.makedirs('output')

f = open('output/log.csv', 'w')
log_writer = csv.writer(f)
log_writer.writerow(['iteration', 'batch', 'loss'])

checkpointer = ModelCheckpoint(
    "output/model.hdf5", monitor='loss', save_best_only=True)


class BatchLossLogger(Callback):

    def on_batch_begin(self, batch, logs={}):
        if batch % 100 == 0:
            log_writer.writerow([iteration, batch, logs.get('loss')])


for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    logger = BatchLossLogger()
    # X_train, y_train = random_subset(X, y)
    history = model.fit(X, y, batch_size=batch_size,
                        nb_epoch=1, callbacks=[logger, checkpointer])
    loss = str(history.history['loss'][-1]).replace(".", "_")

    f2 = open('output/iter-{:02}-{:.6}.txt'.format(iteration, loss), 'w')

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0, 0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)
        f2.write('----- diversity:' + ' ' + str(diversity) + '\n')

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        f2.write('----- Generating with seed: "' + sentence + '"' + '\n---\n')
        sys.stdout.write(generated)

        for i in range(1200):
            x = np.zeros((1, maxlen), dtype=np.int)
            for t, char in enumerate(sentence):
                x[0, t] = char_indices[char]

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        f2.write(generated + '\n')
        print()
    f2.close()
f.close()
