"""Create aligned (clean | synthetic-defect) image pairs for CycleGAN / pix2pix training.

Usage:
    python make_synthetic_defects.py INPUT_DIR OUTPUT_DIR --roi-top 60 --roi-bottom 70 --num-dots 10 --num-lines 10
"""
import argparse
import os
import random

import cv2
import numpy as np


def create_noisy_dataset(input_folder, output_folder, num_dots=10, roi_top=100, roi_bottom=200, num_lines=10):
    """
    Creates a noisy dataset by adding black or gray dots randomly within an ROI in images.

    Args:
        input_folder: Path to the folder containing the input images.
        output_folder: Path to the folder where the processed images will be saved.
        num_dots: The number of dots to add to each image (default: 10).
        roi_top: Top boundary of the region of interest (default: 100).
        roi_bottom: Bottom boundary of the region of interest (default: 200).
        num_lines: The number of short line segments to add to each image (default: 10).

    Each output file is the original and the corrupted image concatenated horizontally, which is the
    "aligned" format expected by pytorch-CycleGAN-and-pix2pix.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            original_image = image.copy()  # Create a copy of the original image


            roi = image[roi_top:roi_bottom, :]
            max_pixel_value = np.max(roi)  # Find the maximum pixel value in the ROI

            # Find rows where the maximum pixel value occurs
            brightest_rows = np.where(roi == max_pixel_value)[0]


            # Add dots randomly within the ROI
            for _ in range(num_dots):
                best_y_in_roi = roi_top + random.choice(brightest_rows)

                dot_x = random.randint(0, image.shape[1] - 1)  # Random x-coordinate
                dot_y = best_y_in_roi   # on the brightest row(s) of the ROI

                # Randomly select color (black or gray)
                col = random.choice([0, 75])

                # Randomly select radius (1 to 3)
                radius = random.randint(1, 3)

                cv2.circle(image, (dot_x, dot_y), radius=radius, color=(col,col,col), thickness=-1)
            current_x=2
            for _ in range(num_lines):
                best_y_in_roi = roi_top + random.choice(brightest_rows)


                line_y = best_y_in_roi

                line_length = random.randint(0, 10)  # Random line length
                line_x1 = current_x  # Start at the previous line's end
                line_x2 = min(current_x + line_length, image.shape[1] - 1)  # End with random length but within image bounds

                col = random.choice([0, 75])
                thickness = random.randint(1, 2)
                cv2.line(image, (line_x1, line_y), (line_x2, line_y), (col,col,col), thickness)

                current_x = line_x2 + 5  # Update start for next line, with a small gap

            # Save the processed image
            # Concatenate original and processed images horizontally
            concatenated_image = np.concatenate((original_image, image), axis=1)

            # Save the concatenated image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, concatenated_image)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--num-dots", type=int, default=10)
    parser.add_argument("--num-lines", type=int, default=10)
    parser.add_argument("--roi-top", type=int, default=60)
    parser.add_argument("--roi-bottom", type=int, default=70)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    create_noisy_dataset(args.input_folder, args.output_folder, args.num_dots, args.roi_top, args.roi_bottom,
                         args.num_lines)


if __name__ == "__main__":
    main()
