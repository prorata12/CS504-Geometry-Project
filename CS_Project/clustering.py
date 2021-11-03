# An adaptive spatial clustering algorithm based on Delaunay triangulation 의 points dataset 구현
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift

# k = number of clusters
def kmeans(data,k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    return kmeans.labels_

# k= maximum distance between two samples of same cluster
def dbscan(data,k):
    clustering = DBSCAN(eps=k, min_samples=5).fit(data)
    return clustering.labels_

# k = number of clusters
def gmm(data, k):
    gmm = GaussianMixture(n_components=k, random_state=42)
    return gmm.fit_predict(data)

# k = bandwidth
def meanshift(data):
    clustering = MeanShift().fit(data)
    return clustering.labels_

def execute_func(function_name, data, k):
    global cname 
    cname ='{}_{}'.format(function_name, k)
    return {
        'kmeans': lambda: kmeans(data,k),
        'dbscan': lambda: dbscan(data, k),
        'gmm': lambda: gmm(data,k),
        'meanshift': lambda: meanshift(data)
    }[function_name]()

# def nn(data,k):
#     knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
#     return knn.predict(data)

import pickle
# test_216, test_310, test_571
# numbers mean the number of points included
for fname in ['test_216','test_310','test_571']:
    with open("./dataset/{}.txt".format(fname), "rb") as fp:   # Unpickling
        # read data as list type
        # [[x1,y1],[x2,y2],...]
        data = pickle.load(fp)
        print('data with {} points loaded.'.format(len(data)))

    data = np.array(data)

    # clustering

    #cname='kmeans_5k'
    #clusters = kmeans(data, 5)

    # 0.05<k <0.5
    #cname='dbscan_0.1'
    #clusters = dbscan(data, 0.1)
    
    #cname='gmm_5'
    #clusters = gmm(data, 5)
    
    #cname = 'meanshift'
    #clusters = meanshift(data)

    clusters = execute_func('dbscan',data,0.05)
    df = {'x':data[:, 0], 'y':data[:, 1], 'clusters':clusters}
    df = pd.DataFrame(df)

    # create scatter plot for samples from each class
    for cluster_value in set(clusters):
        # get row indexes for samples with this class
        row_ix = np.where(clusters == cluster_value)[0]
        # create scatter of these samples
        plt.scatter(data[row_ix, 0], data[row_ix, 1])

    print('saved plot file as {}_{}.png'.format(cname,fname))
    plt.savefig('./figs/{}_{}.png'.format(cname,fname), dpi=300)
    plt.title('{} : {}'.format(cname, fname))
    # show the plot
    plt.show()