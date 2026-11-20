"""
Main function to calculate the projected GNSS path based on the original GNSS data and the roll and pitch angles. It reads the input data from INPUT_DATA_CSV file, performs the necessary calculations, visualizes the results, and saves the output to OUTPUT_DATA_CSV file.
"""

import pandas as pd
import numpy as np

from calculations import calculate_headings, calculate_gnss_projection
from visualize import plot_gnss_movement, plot_headings

INPUT_DATA_CSV = "gnss_data.csv"
OUTPUT_DATA_CSV = "output_gnss_data_with_projections.csv"
GNSS_MOVEMENT_PLOT_SAVE_PATH = "output_gnss_movement.png"
HEADINGS_PLOT_SAVE_PATH = "output_headings.png"


def main():


    data = pd.read_csv(INPUT_DATA_CSV)

    time_s = data["time_s"].to_numpy()
    x_mm = data["x_mm"].to_numpy()
    y_mm = data["y_mm"].to_numpy()
    roll_deg = data["roll_deg"].to_numpy()
    pitch_deg = data["pitch_deg"].to_numpy()

    headings = calculate_headings(x_mm, y_mm)
    projected_x_mm, projected_y_mm = calculate_gnss_projection(
        x_mm, y_mm, roll_deg, pitch_deg, headings
    )

    plot_gnss_movement(time_s, x_mm, y_mm,projected_x_mm, projected_y_mm,save_path=GNSS_MOVEMENT_PLOT_SAVE_PATH)

    plot_headings(time_s, headings, save_path=HEADINGS_PLOT_SAVE_PATH)

    data["projected_x_mm"] = projected_x_mm
    data["projected_y_mm"] = projected_y_mm
    data["heading_deg"] = np.degrees(headings)


    data.to_csv(OUTPUT_DATA_CSV, index=False)


if __name__ == "__main__":
    main()
