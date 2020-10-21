# Unsupervised learning

## Classification algorithm : K Means 

Not to be confused with KNN, which is supervised. K Means, is an algorithm meant to deal with clustering.
* Let's say you have a set of points which are clustered linearly.
* You take K random points and calculate the distance of the datapoints from these k points using a metric. 
* You assign the datapoints to label of  the point that's nearest to the data point.

Now you have k clusters, although it isn't actually clustered yet.

What do you think can be used in order to refine these? This is a 'property' or a certain 'point' that clusters have, but empty space doesn't. 
It's the **centroid**. Find the centroids of the data under each label and reassign the k initial points to them.
Then redo the other algorithm.

THAT'S IT!

The points will converge to the centres of the clusters.

Disadvantages: 
1. Doesn't work for non linear clusters
2. Need to specify k
3. Can't cluster points that aren't in defined clusters (obviously).

Advantages:
Simple algorithm, takes less time than others (like DBSCAN), and is more popular as a result.
