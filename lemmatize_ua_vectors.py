import numpy as  np
import io

import pymorphy2

morph = pymorphy2.MorphAnalyzer()

def load_vec(emb_path, nmax=1000000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            if i == 0:
                continue
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

embd, id2w, w2id = load_vec('data/wiki.multi.ru.vec')
print(len(embd))
lem_w2_id = {}
for w in w2id.keys():
    l_w = morph.parse(w)[0].normal_form
    if l_w in lem_w2_id:
        lem_w2_id[l_w].append(w2id[w])
    else:
        lem_w2_id[l_w] = [w2id[w]]

new_vectors = []
lem_id2w = {}
for i, w in enumerate(lem_w2_id):
    lem_id2w[i] = w
    new_vectors.append(np.array(sum([embd[j] for j in lem_w2_id[w]])/len(lem_w2_id[w])))

with open('wiki.multi.ru_lemmatized.vec', 'w') as f:
    for id in lem_id2w:
        f.write(lem_id2w[id] + ' ')
        for i in new_vectors[id]:
            f.write(str(i) + ' ')
        f.write('\r\n')



