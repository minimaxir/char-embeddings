import numpy as np
import os

file_path = "/Volumes/My Passport/GloVe/glove.6B.50d.txt"

vectors = {}
f = open(file_path, 'rb')
for i, line in enumerate(f):
    line_split = line.strip().split(" ")
    vec = np.array(line_split[1:], dtype=float)
    word = line_split[0]

    for char in word:
        if char in vectors:
            vectors[char] = (vectors[char][0] + vec, vectors[char][1] + 1)
        else:
            vectors[char] = (vec, 1)

f.close()

base_name = os.path.splitext(os.path.basename(file_path))[0] + '-char.txt'
f2 = open(base_name, 'wb')
for word in vectors:
    avg_vector = np.round((vectors[word][0] / vectors[word][1]), 6).tolist()
    f2.write(word + " " + " ".join(str(x) for x in avg_vector) + "\n")

f2.close()
