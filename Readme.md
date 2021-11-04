Computational Geometry class Project

You can see the result by runnning del.py

![image](https://user-images.githubusercontent.com/28648962/140023387-9ed3de4c-3dab-4a8a-bfa5-a43fa8fb5723.png)

This project is aim to cluster data in 2D (can be extended N-dim using Delaunay Triangulation for high dimension)

(1) The left figure shows raw-data
(2) The middel figure shows delaunay triangulation result
(3) The right figure shows clustered result using delaunay triangulation


The abstract algorithm is:

(1) Delaunay-Triangualte given data
(2) Remove global effects (remove globally long edges) with respect to mean and variation of length
(3) Remove local effects (remove locally long edges)
    (3-1) by considering vertex dimension
    (3-2) by applying other clustering algorithms (KMeans, DBSCAN, GMM, Mean shift)

Presentation File
: https://docs.google.com/presentation/d/1J7F2x0UvBO2rQap1j_PKufk_gjNILNFGBiLs-DkWoKk/edit?usp=sharing

