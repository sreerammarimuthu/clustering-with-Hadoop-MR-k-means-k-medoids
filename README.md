# Centroid & Medoid clustering with Hadoop MapReduce

This project explores scalable unsupervised learning using Hadoop MapReduce, with the implementations of both K-Means and K-Medoids clustering algorithms in Java. Here, we process large sets of 2D data points to identify natural groupings via iterative centroid or medoid assignment. Each version supports flexible hyperparameter tuning (k) and outputs both the intermediate and final results.

## Contents

- `java-k-means/`, `java-k-medoids/`: Implements K-Means and K-Medoids clustering algorithms respectively via MapReduce. Supports dynamic k, iterative convergence, and early stopping.

- `data/`: Clean x,y coordinate data for clustering. Separate folders for K-Means and K-Medoids with same data points (x,y) but different seed points.  

- `output/`: Organized by algorithm and depends on the k value. Contains both: Final cluster centers or medoids. Cluster-to-point assignments for visual confirmation.

- `utils/data-creation.ipynb`: Jupyter notebook to create the dataset with random data and seed points.

## Notes

- All Java programs follow Hadoop’s Mapper → Combiner → Reducer flow for optimization. Centroid/medoid assignment is done via Euclidean distance.
- Output format varies slightly for each algorithm. For KMeans outputs both centers and assigned points. KMedoids outputs final medoids and their associated members. Works with any 2D numeric dataset in CSV format: x,y

