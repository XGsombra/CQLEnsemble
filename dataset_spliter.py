import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs


def split_dataset(X, y, num_datasets=10, is_GMM=True, scatterness=-1):

    N, d = X.shape
    datasets = []
    datasets_indices = []

    # split the dataset by GMM
    if is_GMM:
        gmm = GaussianMixture(n_components=num_datasets).fit(X)
        labels = gmm.predict(X)
        # classify each data point to a cluster
        for i in range(num_datasets):
            datasets_indices.append(labels == i)
        # if scatterness is 1, return the split done by GMM
        if scatterness == -1:
            for i in range(num_datasets):
                datasets.append([X[datasets_indices[i]], y[datasets_indices[i]]])
            return datasets
        else:
            # find the statistics of the GMM clusters
            means = np.zeros((num_datasets, d))
            covs = np.zeros((num_datasets, d, d))
            for i in range(num_datasets):
                means[i] = np.average(X[datasets_indices[i]], axis=0)
                covs[i] = np.cov(X[datasets_indices[i]].T) * scatterness

            # calculate the probability for each data point to be in cluster i
            dist = (X[:, :, np.newaxis]-means.T[np.newaxis, :, :])
            dist = np.transpose(dist, axes=[0, 2, 1])[:, :, np.newaxis, :]
            distT = np.transpose(dist, axes=[0, 1, 3, 2])
            nominator = np.exp(-dist @ np.linalg.inv(covs)[np.newaxis, :, :] @ distT / 2).reshape((N, num_datasets))
            denominator = np.sqrt(np.linalg.det(covs)) * (2*np.pi)**(d/2)
            probs = nominator / denominator
            probs = probs / np.sum(probs, axis=1)[:, np.newaxis]

            # Assign each data entry to some datasets
            threshold = 1 / (num_datasets * scatterness)
            for dataset_idx in range(num_datasets):
                dataset_indices = probs[:, dataset_idx] > threshold
                datasets.append([X[dataset_indices], y[dataset_indices]])

    # split the dataset using bootstrap
    else:
        pass


    return datasets


if __name__ == "__main__":
    num_datasets = 5
    N = 1000
    num_dim = 2
    scatterness = 2

    X, y = make_blobs(n_samples=N, centers=num_datasets,
                           cluster_std=1.5, random_state=1)

    datasets = split_dataset(X, y, num_datasets, True, scatterness)
    X = []
    labels = []
    for i in range(num_datasets):
        X.extend(datasets[i][0])
        labels.extend([i] * len(datasets[i][0]))
    X = np.array(X)
    for i in range(num_datasets):
        print(f"Dataset {i} has {len(datasets[i][0])} entries.")
    labels = np.array(labels)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    plt.show()
