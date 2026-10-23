# Robust model fitting: globally optimal consensus maximisation (A*) vs. RANSAC

A study and experimental comparison of
[Chin et al., *Efficient Globally Optimal Consensus Maximisation with Tree Search* (CVPR 2015)](https://openaccess.thecvf.com/content_cvpr_2015/html/Chin_Efficient_Globally_Optimal_2015_CVPR_paper.html)
against RANSAC, applied to homography estimation for image stitching.

📄 **Report:** [`consensus_maximisation_astar_report.pdf`](consensus_maximisation_astar_report.pdf)

## Problem and approach

Maximum consensus looks for the model parameters that agree with the largest number of data points (inliers) within a
threshold. RANSAC solves this fast but only approximately. Chin et al. reformulate it as a search over the bases of an
LP-type (minimax) problem and solve it **exactly** with A* and an admissible heuristic.

The study had three parts:
1. Explain the method: minimax fitting, bases and support sets, the tree of bases, and the A* heuristic.
2. **Line fitting** on synthetic data with outliers, comparing RANSAC and A*.
3. **Image stitching** on five image pairs (one with deliberately poor overlap):
   * SIFT keypoints and matching (VLFeat);
   * a homography from RANSAC and from A*;
   * warping and blending into a panorama;
   * a qualitative and quantitative comparison.

## Results (from the report)

| Image pair | RANSAC inliers | A* inliers | RANSAC time (s) | A* time (s) |
|---|---|---|---|---|
| 1 | 279 | 282 | 0.0027 | 4.25 |
| 2 | 437 | 437 | 0.0038 | 2.79 |
| 3 | 344 | 460 | 0.0061 | 0.085 |
| 4 | 171 | 177 | 0.0017 | 5.68 |
| 5 | 79 | 103 | 0.0034 | 0.96 |

Line fitting: RANSAC found 84 inliers and A* found 89.

* A* always finds **at least as many inliers**, as its optimality guarantee predicts.
* A* is **orders of magnitude slower**.
* In 4 of 5 panoramas, the RANSAC homography gave the visually better stitch.
* Conclusion: for image stitching, RANSAC is the more practical choice.

## Tech

MATLAB · VLFeat (SIFT) · RANSAC · A* tree search / LP-type problems · homography estimation

## How to run

The experiments used the authors' public MATLAB package (A* consensus maximisation), Peter Kovesi's RANSAC functions,
and VLFeat. Those third-party files are not redistributed here. Get the package from the paper's authors, add
`vlfeat/toolbox/vl_setup` to the MATLAB path, and run `homography_maxcon_Keble.m`.
