from __future__ import print_function
from keras.models import Model
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, Input
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.decomposition import PCA
from keras.utils import plot_model
import numpy as np
import random
import sys
import csv
import os
import h5py
import time

embeddings_path = "glove.840B.300d-char.txt"
embedding_dim = 300
batch_size = 128
use_pca = False
lr = 0.001
lr_decay = 1e-4
maxlen = 40
consume_less = 2   # 0 for cpu, 2 for gpu

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


# test code to sample on 10% for functional model testing

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
#embedding_matrix = np.random.uniform(-1, 1, (len(chars), 300))
for char, i in char_indices.items():
    #print ("{}, {}".format(char, i))
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
    len(chars), embedding_dim, input_length=maxlen,
    weights=[embedding_matrix_pca] if use_pca else [embedding_matrix])
# embedding_layer = Embedding(
#     len(chars), embedding_dim, input_length=maxlen)
embedded = embedding_layer(main_input)

# RNN Layer
rnn = LSTM(256, implementation=consume_less)(embedded)

aux_output = Dense(len(chars))(rnn)
aux_output = Activation('softmax', name='aux_out')(aux_output)

# Hidden Layers
hidden_1 = Dense(512, use_bias=False)(rnn)
hidden_1 = BatchNormalization()(hidden_1)
hidden_1 = Activation('relu')(hidden_1)

hidden_2 = Dense(256, use_bias=False)(hidden_1)
hidden_2 = BatchNormalization()(hidden_2)
hidden_2 = Activation('relu')(hidden_2)

main_output = Dense(len(chars))(hidden_2)
main_output = Activation('softmax', name='main_out')(main_output)

model = Model(inputs=main_input, outputs=[main_output, aux_output])

optimizer = Adam(lr=lr, decay=lr_decay)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, loss_weights=[1., 0.2])
model.summary()

# plot_model(model, to_file='model.png', show_shapes=True)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-6) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


if not os.path.exists('output'):
    os.makedirs('output')

f = open('output/log.csv', 'w')
log_writer = csv.writer(f)
log_writer.writerow(['iteration', 'batch', 'batch_loss',
                     'epoch_loss', 'elapsed_time'])

checkpointer = ModelCheckpoint(
    "output/model.hdf5", monitor='main_out_loss', save_best_only=True)


class BatchLossLogger(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('main_out_loss'))
        if batch % 50 == 0:
            log_writer.writerow([iteration, batch,
                                 logs.get('main_out_loss'),
                                 np.mean(self.losses),
                                 round(time.time() - start_time, 2)])

start_time = time.time()
for iteration in range(1, 20):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    logger = BatchLossLogger()
    # X_train, y_train = random_subset(X, y)
    # history = model.fit(X_train, [y_train, y_train], batch_size=batch_size,
    #                     epochs=1, callbacks=[logger, checkpointer])
    history = model.fit(X, [y, y], batch_size=batch_size,
                        epochs=1, callbacks=[logger, checkpointer])
    loss = str(history.history['main_out_loss'][-1]).replace(".", "_")

    f2 = open('output/iter-{:02}-{:.6}.txt'.format(iteration, loss), 'w')

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
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

            preds = model.predict(x, verbose=0)[0][0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        f2.write(generated + '\n')
        print()
    f2.close()

    # Write embeddings for current characters to file
    # The second layer has the embeddings.

    embedding_weights = model.layers[1].get_weights()[0]
    f3 = open('output/char-embeddings.txt', 'w')
    for char in char_indices:
        if ord(char) < 128:
            embed_vector = embedding_weights[char_indices[char], :]
            f3.write(char + " " + " ".join(str(x)
                                           for x in embed_vector) + "\n")
    f3.close()

f.close()
