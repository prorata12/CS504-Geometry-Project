import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import collections
from sklearn.datasets import make_blobs

# Dataset selection
# =================================================== #
# Random Point generation - Hank Ditton's suggestion

# https://stackoverflow.com/questions/19668463/generating-multiple-random-x-y-coordinates-excluding-duplicates
# Hank Ditton's suggestion
radius = 200 # no other points in 200
rangeX = (0, 2500) # randPoints X range
rangeY = (0, 2500) # randPoints Y range
qty = 20  # or however many points you want

# Generate a set of all points within 200 of the origin, to be used as offsets later
# There's probably a more efficient way to do this.
deltas = set()
for x in range(-radius, radius+1):
    for y in range(-radius, radius+1):
        if x*x + y*y <= radius*radius:
            deltas.add((x,y))

randPoints = []
excluded = set()
i = 0
while i<qty:
    x = random.randrange(*rangeX) #uniform distribution
    y = random.randrange(*rangeY)
    if (x,y) in excluded: continue
    randPoints.append(list((x,y)))
    i += 1
    excluded.update((x+dx, y+dy) for (dx,dy) in deltas)



# =================================================== #
# Create Delaunay Triangulation
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html

#points = np.array(randPoints)
#points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]]) # x, y coordinate
features, true_labels = make_blobs(
     n_samples=200,
     centers=3,
     cluster_std=2.75,
     random_state=42
)
# =================================================== #

import pickle
# test_216, test_310, test_571
# numbers mean the number of points included
dataset = []
for fname in ['test_216','test_310','test_571']:
    with open("./dataset/{}.txt".format(fname), "rb") as fp:   # Unpickling
        # read data as list type
        # [[x1,y1],[x2,y2],…]
        data = pickle.load(fp)
        print('data with {} points loaded.'.format(len(data)))

    data = np.array(data)
    dataset.append(data)