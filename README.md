# Centroid & Medoid clustering with Hadoop MapReduce

This project explores scalable unsupervised learning using Hadoop MapReduce, with the implementations of both K-Means and K-Medoids clustering algorithms in Java. Here, we process large sets of 2D data points to identify natural groupings via iterative centroid or medoid assignment. Each version supports flexible hyperparameter tuning (k) and outputs both the intermediate and final results.

## Contents

- Java Code (java-k-means/, java-k-medoids/): Implements K-Means and K-Medoids clustering via MapReduce. Supports dynamic k, iterative convergence, and early stopping.

- Data (data/): Clean x,y coordinate data for clustering. Separate folders for K-Means and K-Medoids.

- Output (output/): Organized by algorithm and k value. Contains both: Final cluster centers or medoids. Cluster-to-point assignments for visual verification.

- Utils: (utils/data-creation.ipynb): Jupyter notebook to generate custom datasets or random seed points

## Notes

- All Java programs follow Hadoop’s Mapper → Combiner → Reducer flow. Centroid/medoid assignment is done via Euclidean distance.
- Output format varies slightly for each algorithm. KMeans outputs both centers and assigned points. KMedoids outputs final medoids and their associated members. Works with any 2D numeric dataset in CSV format: x,y

