"""
Helper functions to calculate Heading directions and GNSS projections based on roll and pitch angles.
"""




import numpy as np


GNSS_INSTALLATION_Height = 1500

def calculate_headings(x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:


    # compute the differences until the last point
    dy = y_mm[1:] - y_mm[:-1]
    dx = x_mm[1:] - x_mm[:-1]

    headings = np.arctan2(dy, dx)  

    # set heading of the last point to be the same as the second to last point
    headings = np.append(headings, headings[-1])
    
    return headings   




def calculate_gnss_projection(x_mm: np.ndarray, y_mm: np.ndarray, roll_deg: np.ndarray, pitch_deg: np.ndarray, heading_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    roll_rad = np.radians(roll_deg)
    pitch_rad = np.radians(pitch_deg)

    side_orientation = np.sin(roll_rad)
    forward_orientation = np.sin(pitch_rad)

    x_delta = GNSS_INSTALLATION_Height * (forward_orientation * np.cos(heading_rad) + side_orientation * np.sin(heading_rad))
    y_delta = GNSS_INSTALLATION_Height * (forward_orientation * np.sin(heading_rad) - side_orientation * np.cos(heading_rad))



    projection_x_mm = x_mm - x_delta
    projection_y_mm = y_mm - y_delta

    
    return projection_x_mm, projection_y_mm

