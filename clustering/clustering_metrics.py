import numpy as np

def euclidean_distance_between_points(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1-point2)**2))

def calculate_silhouette_score(X: np.ndarray,y:np.ndarray) -> float:
        unique_labels = np.unique(y)
        n_clusters = len(unique_labels)

        # Check if there's only one cluster or empty cluster
        if n_clusters == 1 or n_clusters == len(X):
            return 0
        a = np.zeros(X.shape[0])
        b = np.zeros(X.shape[0])
        
        for i,xi in enumerate(X):
            label = y[i]
            same_cluster = X[y==label]
            a[i] = np.mean([euclidean_distance_between_points(xi,point) for point in same_cluster if not np.array_equal(xi,point)])
            
            nearest_dist = float("inf")
            for curr_label in unique_labels:
                if curr_label!=label:
                    other_cluster = X[y==curr_label]
                    curr_dist = np.mean([euclidean_distance_between_points(xi,point) for point in other_cluster])
                    if curr_dist<nearest_dist:
                        nearest_dist = curr_dist
            b[i] = nearest_dist
        silhouette_scores = (b-a)/np.maximum(a,b)
        return np.mean(silhouette_scores)

def calculate_dunn_index(X: np.ndarray,y: np.ndarray) -> float:
        unique_labels = np.unique(y)
        max_intra_cluster_distance = 0
        for label in unique_labels:
            same_cluster = X[y==label]
            intra_cluster_distances = [
                euclidean_distance_between_points(same_cluster[i], same_cluster[j])
                for i in range(len(same_cluster))
                for j in range(i + 1, len(same_cluster))  
            ]
            max_intra_cluster_distance = max(max_intra_cluster_distance,max(intra_cluster_distances))
        
        min_inter_cluster_distance = float("inf")
        for label1 in unique_labels:
            cluster1 = X[y==label1]
            for label2 in unique_labels:
                if label1!=label2:
                    cluster2 = X[y==label2]
                    inter_cluster_distances = [euclidean_distance_between_points(point1,point2) for point1 in cluster1 for point2 in cluster2]
                    min_inter_cluster_distance = min(min_inter_cluster_distance,min(inter_cluster_distances))

        # Avoid division by zero
        if max_intra_cluster_distance == 0:
            raise ValueError("Max intra-cluster distance is zero, possibly due to identical points.")
        
        return min_inter_cluster_distance/max_intra_cluster_distance
    
def calculate_davis_bouldin_index(X: np.ndarray,y: np.ndarray, centroids: np.ndarray) -> float:
    unique_labels = np.unique(y)
    avg_intra_cluster_distances = []
    for label in unique_labels:
        cluster = X[y==label]
        intra_cluster_distances = []
        for point in cluster:
            dist_from_centroid = euclidean_distance_between_points(point,centroids[label])
            intra_cluster_distances.append(dist_from_centroid)
        avg_intra_cluster_distances.append(np.mean(intra_cluster_distances))
    
    R_ij = []
    for label1 in unique_labels:
        max_R_ij = 0
        centroid1 = centroids[label1]
        for label2 in unique_labels:
            if label1!=label2:
                centroid2 = centroids[label2]
                d_ij = euclidean_distance_between_points(centroid1,centroid2)
                if d_ij==0: # avoid division by zero
                    continue
                curr_R_ij = (avg_intra_cluster_distances[label1]+avg_intra_cluster_distances[label2])/d_ij
                max_R_ij = max(max_R_ij,curr_R_ij)
        R_ij.append(max_R_ij)
    dbi = np.mean(R_ij)
    return dbi
