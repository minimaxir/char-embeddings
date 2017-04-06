from __future__ import print_function
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, Input
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

maxlen = 40  # must match length which generated model
num_char_generated = 30000

text = open('magic_cards.txt').read()

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-6) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

print('Loading model...')
model = load_model('output/model.hdf5')
f2 = open('output/text_sample.txt', 'w')

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

    for i in range(num_char_generated):
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
