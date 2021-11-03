import dataset as ds
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import collections
from sklearn.datasets import make_blobs
import copy

### 1. Data Selection

# points = ds.features
# points = ds.randPoints
#points = ds.dataset[0]
#points = ds.dataset[1]
points = ds.dataset[2]


#duplicated value deletion
points = points.tolist()
new_points = []
for point in points:
    if point not in new_points:
        new_points.append(point)
points = np.array(new_points)

tri = Delaunay(points)


### 2. Plotting Original data with Delaunay Triangulation

plt.subplot(131)
plt.triplot(points[:,0], points[:,1], tri.simplices) # [[0,1,2], [2,3,1]] vertices of each trinagle
checker = []

# code bleow can checker points which is not involved in triangles

# for triangle in tri.simplices:
#     if triangle[0] not in checker:
#         checker.append(triangle[0])
#     if triangle[1] not in checker:
#         checker.append(triangle[1])
#     if triangle[2] not in checker:
#         checker.append(triangle[2])
# for i in range(len(points)):
#     if i not in checker:
#         print(i,' th vertex is not in triangle')


# simplices stores the indices of the three points of each triangle
plt.plot(points[:,0], points[:,1], 'o')
plt.show(block=False)



### 3. Get edge information from the Delaunay Triangulated result

# get all edges (from point_index to point_index)
_edges = [] # empty list to avoid edge duplication (list of (vertex1_index, vertex2_index))

edges = {} # dictionary {edge_index : (vertex1_index, vertex2_index)}
# edges_length = [] # edges_length[i] stores length of the edge of index i
    # edges_length will be implemented below

ind = 0 #index of the edge
for vertices in tri.simplices:
    # use 'set' data structure to ignore order
    edge1 = set([vertices[0],vertices[1]])
    edge2 = set([vertices[1],vertices[2]])
    edge3 = set([vertices[2],vertices[0]])
    if not edge1 in _edges:
        _edges.append(edge1)
        edges[ind] = edge1
        ind += 1
    if not edge2 in _edges:
        _edges.append(edge2)
        edges[ind] = edge2
        ind += 1
    if not edge3 in _edges:
        _edges.append(edge3)
        edges[ind] = edge3
        ind += 1
#print("_edges",_edges)
#print("edges",edges)



### 4. Calculate lenght of each edge (keeping its index)
# this part can be merged with above one(3)

edges_length = [] # list of [edge_index, its length]
edges_length_values = [] # list of edge length (can be deleted, just for convenience)

for i, _edge in enumerate(_edges):
    # _edge : list of (from_vertex, tovertex) without order(undirected)
    edge = list(_edge)
    point1_index = edge[0]
    point2_index = edge[1]
    
    point1 = points[point1_index]
    point2 = points[point2_index]

    edge_length = ((((point2[0] - point1[0] )**2) + ((point2[1]-point1[1])**2) )**0.5)
    edges_length.append([i,edge_length])
    edges_length_values.append(edge_length)

#edges_length = sorted(edges_length, key=lambda x: x[1], reverse=True) -> O(nlogn), slower than just pop up(O(n))
#print("edges_length", edges_length)



### 5. Delete long edges to cluster graph
# delete all edges whose length > Global_Cut_Value(mean + std)
# it doesn't consider any local long edge

lengths = np.array(edges_length_values) # [length1, length2, length3, ...]
Global_mean = np.average(lengths)
Global_std = np.std(lengths)

#Global_Cut_Value = 1.3
Global_Cut_Value = Global_mean + Global_std

processed_edges = copy.deepcopy(edges) # this stores only short edge(i.e., after clustering)

for edge in edges_length: # [[index1,length1], [index2,length2], ...]
    edge_index = edge[0]
    edge_length = edge[1]
    if edge_length > Global_Cut_Value:
        del processed_edges[edge_index]

#print("edges",edges)
#print("processed_edges",processed_edges)



### 6. Plot processed edges

# plt.triplot(points[:,0], points[:,1], tri.simplices) # [[0,1,2], [2,3,1]] vertices of each trinagle
# the line above will plot unprocessed edges

plt.subplot(132)
for _edge in processed_edges.values(): # this stores list of (from_point_index, to_point_index)
    edge = list(_edge)
    x1 = points[edge[0]][0] # x value of from_point
    y1 = points[edge[0]][1]
    x2 = points[edge[1]][0] # x value of to_point
    y2 = points[edge[1]][1]
    plt.plot([x1,x2],[y1,y2], 'b') # plot all processed edges
plt.plot(points[:,0], points[:,1], 'o') # plot all points



### 7. From the processed edges, find clusters using DFS
# 7.1 Initialize the nodes before DFS

class Node:
    # this is for all data points in the graph 
    def __init__(self):
        self.adjacent_points = [] # all adjacent points (edge outward)

    def set_adjs(self,adjacent_points): # initialize edges
        self.adjacent_points = copy.deepcopy(adjacent_points)

    def get_adjs(self):
        return self.adjacent_points

    def add_adj(self,point):
        self.adjacent_points.append(point)

# initialize nodes of the graph
Nodes = []
for i in range(len(points)):
    _Node = Node()
    Nodes.append(_Node) # index in Nodes = index of the point(node) # Nodes[i] = point with i index

# get all adjacent points of each node
for _edge in processed_edges.values():
    edge = list(_edge)
    point1 = edge[0]
    point2 = edge[1]
    Nodes[point1].add_adj(point2)
    Nodes[point2].add_adj(point1)
#print(Nodes[0].get_adjs())


# 7.2 Traversal using DFS with Nodes
# https://itholic.github.io/python-bfs-dfs/

not_visited = []
for i in range(len(points)):
    not_visited.append(i)

clusters = [] #clusters[i] stores a cluster of index i
while not_visited:
    visited = []
    stack = [not_visited.pop()]
    
    while stack:
        current_node_index = stack.pop()
        if current_node_index not in visited:
            visited.append(current_node_index)
            if current_node_index in not_visited:
                not_visited.remove(current_node_index)
            stack.extend(Nodes[current_node_index].get_adjs())

    clusters.append(visited)

#print(clusters)



### 8. Plot clustered result

plt.subplot(133)
num_clusters = len(clusters)

#plotting edges
for _edge in processed_edges.values():
    edge = list(_edge)
    x1 = points[edge[0]][0]
    y1 = points[edge[0]][1]
    x2 = points[edge[1]][0]
    y2 = points[edge[1]][1]
    involved_cluster = -1
    for i in range(num_clusters):
        if edge[0] in clusters[i]:
            assert edge[1] in clusters[i]
            involved_cluster = i
            break
    assert involved_cluster != -1
    cluster_color = involved_cluster/num_clusters
    color = plt.cm.viridis(cluster_color)
    plt.plot([x1,x2],[y1,y2], c=color) #assigning different colors for different clusters
    #plt.scatter([x1,x2],[y1,y2], s=[10,10],c=[color,color])

#plotting points
checker=[]
for index in range(len(points)):
    involved_cluster = -1
    for i in range(num_clusters):
        if index in clusters[i]:
            involved_cluster = i
            break 
    assert involved_cluster != -1
    cluster_color = involved_cluster/num_clusters
    color = plt.cm.viridis(cluster_color)
    plt.scatter([points[index][0]],[points[index][1]], s=[80],c=[color])

    # emphasize one point for each cluster
    # if cluster_color not in checker:
    #     checker.append(cluster_color)
    #     plt.scatter([points[index][0]],[points[index][1]], s=[80],c=[color])
    #     print(Nodes[index].get_adjs(),points[index][0],points[index][1],index)
    # else:
    #     plt.scatter([points[index][0]],[points[index][1]], s=[10],c=[color])

plt.show()


print("number of points=", len(points))
print("number of clusters= ", len(clusters))