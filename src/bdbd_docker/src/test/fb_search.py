import numpy as np
from bdbd_common.utils import fstr, gstr
d = 10                           # dimension
nb = 100000                      # database size
nq = 2                         # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
#xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
#xq[:, 0] += np.arange(nq) / 1000.

import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 3                          # we want to see k nearest neighbors
#D, I = index.search(xb[:5], k) # sanity check
#print(I)
#print(D)
D, I = index.search(xq, k)     # actual search
print(len(I))
print('xq\n' + gstr(xq))
print('I\n' + gstr(I))                   # neighbors of the 5 first queries
print('D\n' + gstr(D))                  # neighbors of the 5 last queries

print('query' + gstr(xq))
print(xq.shape)
for k in range(len(I)):
    print('query vector' + gstr(xq[k, :]))
    print('results')
    qresults = I[k]
    for qresult in qresults:
        print(gstr(xb[qresult,:]))
