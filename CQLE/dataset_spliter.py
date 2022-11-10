import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs


def split_dataset(X, y, num_datasets=10, is_GMM=True, scatterness=1):

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
        if scatterness == 1:
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
        np.random.shuffle(X)
        dataset_size = N // num_datasets
        for i in range(num_datasets):
            one_left = 1 if i == num_datasets - 1 and dataset_size < N / num_datasets else 0
            start = i * dataset_size
            end = min((i + 1) * dataset_size + one_left, N)
            indices = np.arange(start, end)
            extra_size = int((scatterness - 1) * dataset_size)
            extra_indices = np.random.choice(
                np.hstack((np.arange(0, start), np.arange(end, N))),
                extra_size,
                replace=False
            )
            indices = np.hstack((indices, extra_indices))
            datasets.append([X[indices.astype(int)], y[indices.astype(int)]])

    return datasets


if __name__ == "__main__":
    num_datasets = 5
    N = 1000
    num_dim = 2
    scatterness = 2

    X, y = make_blobs(n_samples=N, centers=num_datasets,
                           cluster_std=1.5, random_state=1)

    print(X)

    datasets = split_dataset(X, y, num_datasets, False, scatterness)
    XGMM = []
    labels = []

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))
    for i in range(num_datasets):
        axs[i//3, i%3].scatter(datasets[i][0][:, 0], datasets[i][0][:, 1], s=5)
        axs[i//3, i%3].set_xlim([-15, 4])
        axs[i//3, i%3].set_ylim([-12.5, 10])
        axs[i//3, i%3].set_title(f"{len(datasets[i][0])} data entries")
        # axs[i // 3, i % 3].set_axis_off()
        XGMM.extend(datasets[i][0])
        labels.extend([i] * len(datasets[i][0]))
    XGMM = np.array(XGMM)
    for i in range(num_datasets):
        print(f"Dataset {i} has {len(datasets[i][0])} entries.")
    labels = np.array(labels)
    axs[1, 2].scatter(XGMM[:, 0], XGMM[:, 1], c=labels, s=5, cmap='viridis')
    axs[1, 2].set_xlim([-15, 4])
    axs[1, 2].set_ylim([-12.5, 10])
    axs[1, 2].set_title(f"{sum([len(datasets[i][0]) for i in range(num_datasets)])} data entries")
    # plt.show()
