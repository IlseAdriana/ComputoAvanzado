import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.datasets import make_moons, load_iris, load_breast_cancer

# Euclidean distance
def distance(x, y):
    return np.sqrt(np.sum(np.square(x-y)))


# Function to assign clusters
def set_chunk_clusters(data_chunk, centroids):
    clusters_chunk = []

    for x in data_chunk:
        distances = [distance(x, c) for c in centroids]
        clusters_chunk.append(np.argmin(distances))

    return clusters_chunk


# Function to update the centroid values
def update_centroids(data, centroids, clusters):
    # Store the values of the last centroids
    old_centroids = centroids.copy()

    for i in range(len(centroids)):
        centroids[i] = np.mean(data[clusters==i, :], axis=0)
    
    return old_centroids, centroids


# Function to compare old and new centroids
def equal_centroids(old_centroids, new_centroids):
    for old, new in zip(old_centroids, new_centroids):
        if np.array_equal(old, new) == False:
            return False
        
    return True


# Funcion containing the KMeans algorithm
def kmeans(k, data):
    # Select k elements randomly from the dataset as centroids
    centroids = np.asarray([data[np.random.randint(len(data))] for _ in range(k)])

    n_proc = mp.cpu_count() # Number of processors 

    iterations = 0
    while True:
        iterations += 1

        # Create chunks of the data to apply paralelism
        chunks = np.array_split(data, n_proc)
        
        pool = mp.Pool(len(chunks))

        # Obtain clusters by multiprocessing
        results = [] 
        for chunk in chunks:
            results.append(pool.apply_async(set_chunk_clusters, args=(chunk, centroids)))

        pool.close()
        pool.join()

        clusters = []
        for list_ in results:
            clusters += list_.get()

        clusters = np.asarray(clusters)

        # Update centroids
        old_centroids, centroids = update_centroids(data, centroids, clusters)

        # When the centroids stop updating, the algorithm ends
        if (equal_centroids(old_centroids, centroids)): break

    print(f'El algoritmo convergió en la iteración: {iterations}')
    
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='r', s=75, linewidths=3)
    plt.title('Generated clusters')
    plt.show()


def main():
    k = 4 # Number of clusters

    # Make-moons dataset
    # data, _ = make_moons(n_samples=200, noise=0.15)

    # Iris-plant dataset
    # data, _ = load_iris(return_X_y=True)

    # Breast cancer wisconsin dataset
    data, _ = load_breast_cancer(return_X_y=True)
    

    plt.scatter(data[:, 0], data[:, 1])
    plt.title('Original data')
    plt.show()
    
    kmeans(k, data)


if __name__ == '__main__':
    main()