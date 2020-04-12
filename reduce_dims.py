import numpy as np

#datasets = ['0580', '1080', '1580', '2080', '2580', '3080', '3580', '4080']
#labels = ['bs010_is010', 'bs020_is010', 'bs015_is015']


datasets = ['0580', '1080']
labels = ['bs010_is010']

for dataset in datasets:
    for label in labels:
        ptf = 'data/results/' + dataset + '/' + label + '/fv' + dataset + '.npy'
        pts = 'data/results/' + dataset + '/' + label + '/rfv' + dataset + '.npy'
        X = np.load(ptf)
        print(ptf)

        if label == 'bs020_is010':
            #Looks between 0.2nm-1.0nm.
            X = X[:, 1:11]
            X = np.delete(X, 4, axis=1)
            X = np.delete(X, 4, axis=1)

        elif label == 'bs010_is010':
            # Looks between 0.3nm-0.7nm.
            X = X[:, 2:18]
            X = np.delete(X, 4, axis=1)
            X = np.delete(X, 4, axis=1)
            X = np.delete(X, 4, axis=1)
            X = np.delete(X, 4, axis=1)
            X = np.delete(X, 4, axis=1)
            X = np.delete(X, 4, axis=1)
            X = np.delete(X, 4, axis=1)
            X = np.delete(X, 4, axis=1)

        elif label == 'bs015_is010':
            # Looks between 0.15nm-0.75nm.
            #
            X = X[:, 1:13]
            X = np.delete(X, 4, axis=1)
            X = np.delete(X, 4, axis=1)
            X = np.delete(X, 4, axis=1)
            X = np.delete(X, 4, axis=1)

        print(X.shape[1])
        np.save(pts, X)