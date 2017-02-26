'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, Flatten
from keras.layers import LSTM, Convolution1D, MaxPooling1D, Bidirectional, TimeDistributed, GRU, Input, merge, AveragePooling1D
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger
from sklearn.decomposition import PCA
import numpy as np
import random
import sys

embeddings_path = "glove.840B.300d-char.txt"
embedding_dim = 50
batch_size = 128
use_pca = True
lr = 0.01
lr_decay = 1e-4

#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open('shakespeare.txt').read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 80
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

# print (X[0, :])
# print (y[0, :])


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

# Use PCA from sklearn to reduce 300D -> 100D
if use_pca:
    pca = PCA(n_components=embedding_dim)
    pca.fit(embedding_matrix)
    embedding_matrix_pca = np.array(pca.transform(embedding_matrix))
    print (embedding_matrix_pca)
    print (embedding_matrix_pca.shape)

# build the model: a single LSTM
print('Build model...')
#model = Sequential()
main_input = Input(shape=(maxlen,))
embedding_layer = Embedding(
    len(chars), embedding_dim, input_length=maxlen, weights=[embedding_matrix_pca] if use_pca else [embedding_matrix])
embedded = embedding_layer(main_input)

# we add a Convolution1D for each filter length, which will learn nb_filters[i]
# word group filters of size filter_lengths[i]:
convs = []
#filter_lengths = [1, 2, 3, 4, 5, 6, 7]
#nb_filters = [50, 50, 100, 100, 100, 100, 100]

filter_lengths = [1, 2]
nb_filters = [5, 5]

for i in range(len(nb_filters)):
    conv_layer = Convolution1D(nb_filter=nb_filters[i],
                               filter_length=filter_lengths[i],
                               border_mode='valid',
                               activation='relu',
                               subsample_length=1)
    conv_out = conv_layer(embedded)
    conv_out = Flatten()(conv_out)
    convs.append(conv_out)

# concat all conv outputs
x = merge(convs, mode='concat')

# model.add(Convolution1D(32, 3, border_mode='valid', subsample_length=1))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Flatten())

# model.add(MaxPooling1D())
# model.add(Activation('relu'))
# model.add(BatchNormalization())

# model.add(Convolution1D(64, 3, border_mode='valid', subsample_length=1))
# model.add(Activation('relu'))
# model.add(BatchNormalization())

# model.add(MaxPooling1D())
# model.add(Activation('relu'))
# model.add(BatchNormalization())

# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(BatchNormalization())

# model.add(TimeDistributed(Dense(16)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())

# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(BatchNormalization())

x = Dense(128)(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# model.add(Convolution1D(128, 3, border_mode='valid', subsample_length=1))
# model.add(Activation('relu'))
# model.add(BatchNormalization())

# model.add(Bidirectional(LSTM(16, return_sequences=True)))
# model.add(BatchNormalization())

# model.add(Bidirectional(GRU(16)))
# model.add(BatchNormalization())

# model.add(Dense(len(chars)))
# model.add(Activation('softmax'))

main_output = Dense(len(chars), activation='softmax')(x)

model = Model(input=main_input, output=main_output)

optimizer = Adam(lr=lr, decay=lr_decay)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-6) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# https://keras.io/callbacks/#history


# csv_logger = CSVLogger('training.csv', append=True)
# train the model, output generated text after each iteration

#f = open('log.csv' 'wb')
# log_writer = csv.writer(f)
# log_writer.writerow(['iteration', 'batch', 'loss'])


for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    history = model.fit(X, y, batch_size=batch_size, nb_epoch=1)
    loss = str(history.history['loss'][-1]).replace(".", "_")

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

        for i in range(400):
            x = np.zeros((1, maxlen), dtype=np.int)
            for t, char in enumerate(sentence):
                x[0, t] = char_indices[char]

            # print(x)

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
