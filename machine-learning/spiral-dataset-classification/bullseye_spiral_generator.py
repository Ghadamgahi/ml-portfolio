"""
Bullseye → Spiral synthetic dataset generator.
Recreates the data generation process described in the assignment:
  a. bullseye (concentric rings)
  b. spiral transform (rotate angle by radius)
  c. flood to 10 dimensions (add noise dimensions)
"""

import numpy as np
import matplotlib.pyplot as plt


def make_ring(n, r_inner, r_outer, label, seed=None):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(r_inner, r_outer, n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y, np.full(n, label)


def make_bullseye(n=5000, seed=42):
    """3-layer bullseye: class 0 / class 1 / class 0."""
    rng = np.random.default_rng(seed)
    x0, y0, l0 = make_ring(n, 0.0,  40.0,  1)
    x1, y1, l1 = make_ring(n, 40.0, 80.0, -1)
    x2, y2, l2 = make_ring(n, 80.0, 130.0, 1)
    X = np.column_stack([np.concatenate([x0, x1, x2]),
                         np.concatenate([y0, y1, y2])])
    y = np.concatenate([l0, l1, l2])
    return X, y


def spiral_transform(X, turns=2.5):
    """
    Rotate each point's angle by an amount proportional to its radius.
    Points further from center rotate more, winding the rings into a spiral.
    """
    angles = np.arctan2(X[:, 1], X[:, 0])
    radii = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    radii_norm = (radii - radii.min()) / (radii.max() - radii.min())
    new_angles = angles + turns * 2 * np.pi * radii_norm
    X_spiral = np.column_stack([radii * np.cos(new_angles),
                                 radii * np.sin(new_angles)])
    return X_spiral


def flood_to_nd(X, n_dims=10, noise_scale=1.0, seed=42):
    """Embed 2D data into n_dims by appending Gaussian noise dimensions."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_scale, (X.shape[0], n_dims - X.shape[1]))
    return np.hstack([X, noise])


def plot_comparison(X_bull, X_spiral, y, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {-1: 'purple', 1: 'gold'}
    for label in [-1, 1]:
        mask = y == label
        axes[0].scatter(X_bull[mask, 0], X_bull[mask, 1],
                        s=1, c=colors[label], label=str(label))
        axes[1].scatter(X_spiral[mask, 0], X_spiral[mask, 1],
                        s=1, c=colors[label], label=str(label))
    axes[0].set_title('Step a: Bullseye')
    axes[1].set_title('Step b: Spiral')
    for ax in axes:
        ax.axis('equal')
        ax.legend(markerscale=5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


if __name__ == '__main__':
    X_bull, y = make_bullseye(n=5000)
    X_spiral = spiral_transform(X_bull, turns=2.5)
    X_10d = flood_to_nd(X_spiral, n_dims=10, noise_scale=15.0)

    print(f'Bullseye shape:  {X_bull.shape}')
    print(f'Spiral shape:    {X_spiral.shape}')
    print(f'10D shape:       {X_10d.shape}')
    print(f'Label counts:    {dict(zip(*np.unique(y, return_counts=True)))}')

    plot_comparison(X_bull, X_spiral, y, save_path='bullseye_spiral.png')
