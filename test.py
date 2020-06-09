import numpy as np
import pdb
from scipy.special import gammaln

class Test():
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def update(self):
        print(self._x)
        print(self._y)
        pdb.set_trace()
        self._x = self._y

        self._y +=2

        print(self._y)
        print(self._x)

        pdb.set_trace()

#m = Test(x=2, y=5)
#m.update()

pdb.set_trace()

N = 4
dim = 2
K = 3

alphak = np.random.randint(low=1, high=3, size=(K,))

X = np.random.randint(low=1, high= 10, size=(N, dim))

pdb.set_trace()


mk = np.random.randint(low=1, high=3, size=(K, dim))
vk = np.random.randint(low=1, high=3, size=(K,))
betak = np.ones(K)
wk = np.ones((K, dim, dim))


pdb.set_trace()

X = np.ones((N, dim))
Z = np.random.randint(low=1, high=10, size=(N, K))



M = X - Y
Y = np.matmul(np.transpose(Z), X)
