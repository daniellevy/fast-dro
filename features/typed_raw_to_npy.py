"""
Loads the images downloaded from
http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ into an .npy file
"""

import os

import scipy
import matplotlib.pyplot as plt
import numpy as np
import imageio, skimage

print_rate = 200  # write a log line after that many images
test_split_fraction = 0.2

base_dir = 'English/Fnt'
path_to_npy = 'data/typedMNIST'

X = []
Y = []

for directory in os.listdir(base_dir):
    y = int(directory[-2:]) - 1 # Label of the image is in the directory
    
    print(f'-- Processing digit {y} --')
    
    for i, f in enumerate(os.listdir('%s/%s/*.png' % (base_dir, directory))):
        if i % print_rate == 0:
            print(f'Done {i} images of digit {y}')
        i += 1
        v = (255 - imageio.imread('%s/%s/%s' % (base_dir, directory, f)))
        w = skimage.transform.resize(v, (28, 28), preserve_range=True)

        X.append(w)
        Y.append(y)

X = np.array(X)
Y = np.array(Y)

p = test_split_fraction
n = X.shape[0]

n_test = int(p * n)
n_train = n - n_test

idx = np.random.permutation(np.arange(n))

idx_train = idx[:n_train]
idx_test = idx[n_train:]

Y = Y.astype('int32')

dataset = {
    'X_train': X[idx_train] / 255,
    'y_train': Y[idx_train],
    'X_test': X[idx_test] / 255,
    'y_test': Y[idx_test]
}

np.save(path_to_npy, dataset)
