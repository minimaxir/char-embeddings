import numpy as np

file_path = "/Volumes/My Passport/GloVe/glove.6B.50d.txt"

vectors = {}
f = open(file_path, 'rb')
for i, line in enumerate(f):
    line_split = line.strip().split(" ")
    vec = np.array(line_split[1:], dtype=float)
    word = line_split[0]
    if word in vectors:
        vectors[word] = (vectors[word][0] + 1, vectors[word][1] + vec)
    else:
        vectors[word] = (1, vec)

f.close()

print(vectors)
