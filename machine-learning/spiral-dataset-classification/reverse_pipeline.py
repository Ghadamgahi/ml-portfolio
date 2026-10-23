"""
Reverse the data generation process on the assignment dataset:
  1. PCA to go from 10D → 2D (recovers the spiral)
  2. KMeans clustering within each class to identify rings
  3. Radial rescaling to reconstruct the bullseye
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def load_data(input_path, label_path):
    X = pd.read_csv(input_path, index_col=0).values
    y = pd.read_csv(label_path, index_col=0).values.ravel()
    return X, y


def get_spiral(X, n_components=2):
    """Step c reversal: PCA from 10D to 2D."""
    pca = PCA(n_components=n_components)
    X_2d = pca.fit_transform(X)
    print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
    return X_2d, pca


def reconstruct_bullseye(X_2d, y, band_width=None):
    """
    Step b reversal: reassign radii to reconstruct concentric rings.
    - Class -1: 1 cluster (middle ring)
    - Class  1: 2 clusters (inner + outer ring)
    """
    angles = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    radii = np.sqrt(X_2d[:, 0]**2 + X_2d[:, 1]**2)

    # Find ring centres via KMeans
    all_centers = []
    for label, n_clusters in zip([-1, 1], [1, 2]):
        idx = np.where(y == label)[0]
        r = radii[idx].reshape(-1, 1)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        km.fit(r)
        for c in km.cluster_centers_.ravel():
            all_centers.append(c)
    all_centers = sorted(all_centers)
    print(f'Ring centres (PCA radius): {[f"{c:.3f}" for c in all_centers]}')

    # Band width = distance between 2nd and 3rd ring centres
    if band_width is None:
        band_width = all_centers[2] - all_centers[1]
    print(f'Band width: {band_width:.3f}')

    bands = [(c - band_width / 2, c + band_width / 2) for c in all_centers]

    # Rescale each cluster to its target band
    new_radii = radii.copy()
    for label, n_clusters, band_indices in zip([-1, 1], [1, 2], [[1], [0, 2]]):
        idx = np.where(y == label)[0]
        r = radii[idx].reshape(-1, 1)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        km.fit(r)
        sorted_centers = np.argsort(km.cluster_centers_.ravel())
        for i, band_idx in zip(sorted_centers, sorted(band_indices)):
            cluster_mask = km.labels_ == i
            r_cluster = r[cluster_mask].ravel()
            r_min, r_max = r_cluster.min(), r_cluster.max()
            t_min, t_max = bands[band_idx]
            rescaled = t_min + (r_cluster - r_min) / (r_max - r_min + 1e-8) * (t_max - t_min)
            new_radii[idx[cluster_mask]] = rescaled

    X_bull = np.column_stack([new_radii * np.cos(angles),
                               new_radii * np.sin(angles)])
    return X_bull


def plot_reversal(X_2d, X_bull, y, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {-1: 'purple', 1: 'gold'}
    for label in [-1, 1]:
        mask = y == label
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        s=1, c=colors[label], label=str(label))
        axes[1].scatter(X_bull[mask, 0], X_bull[mask, 1],
                        s=1, c=colors[label], label=str(label))
    axes[0].set_title('PCA projection (spiral — step b)')
    axes[1].set_title('Reconstructed bullseye (step a)')
    for ax in axes:
        ax.axis('equal')
        ax.legend(markerscale=5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


if __name__ == '__main__':
    X, y = load_data('input_train.csv', 'label_train.csv')
    X_2d, pca = get_spiral(X)
    X_bull = reconstruct_bullseye(X_2d, y)
    plot_reversal(X_2d, X_bull, y, save_path='bullseye_reconstructed.png')
