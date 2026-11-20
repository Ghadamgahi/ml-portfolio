"""
Helper function to visualize the results of the GNSS path projection, including the original and projected paths.
"""


import matplotlib.pyplot as plt
import numpy as np


def plot_gnss_movement( time_s: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray, projected_x_mm: np.ndarray, projected_y_mm: np.ndarray, save_path: str) -> None:


    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_mm, y_mm, color='blue', s=20, label='Original GNSS Path')
    ax.scatter(projected_x_mm, projected_y_mm, color='green', s=20, label='Projected GNSS Path')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Original vs Projected GNSS Path")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(save_path, dpi=120)

def plot_headings(time_s: np.ndarray, headings_rad: np.ndarray, save_path: str) -> None:

    headings_deg = np.degrees(headings_rad)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_s, headings_deg, color='red', marker='o', label='Heading (degrees)')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (degrees)")
    ax.set_title("Vehicle Heading Over Time")
    ax.legend()
    ax.grid(True)
    fig.savefig(save_path, dpi=120)
