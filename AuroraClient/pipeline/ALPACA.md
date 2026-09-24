# ALPACA Documentation

## Overview
ALPACA (Automated Landmarking through Pointcloud Alignment and Correspondence Analysis) is a tool for aligning 3D landmarks from a source mesh to a target mesh. This implementation is based on the original ALPACA code by Arthur Porto, with modifications by Alejandro Gutierrez.

## Class Structure

### ALPACA Class
The main class that handles the alignment process.

#### Default Parameters
- `point_density`: 0.5 - Controls how densely points are sampled
- `normal_search_radius`: 1 - Radius for computing surface normals
- `fpfh_search_radius`: 2.5 - Radius for computing FPFH features
- `distance_threshold`: 0.5 - Maximum distance for point matching
- `max_ransac_iter`: 10,000,000 - Maximum RANSAC iterations
- `icp_distance_threshold`: 0.1 - Maximum distance for ICP matching
- `alpha`: 2.0 - CPD rigidity parameter
- `beta`: 2.0 - CPD motion coherence
- `cpd_iterations`: 100 - Maximum CPD iterations
- `cpd_tolerance`: 0.0001 - Convergence tolerance for CPD

## Main Functions

### 1. create_landmarks_from_mesh
Creates evenly distributed landmarks on a mesh surface.

**Input:**
- Vertices and faces of a 3D mesh
- Number of desired landmarks

**Output:**
- Array of landmark coordinates

### 2. align_landmarks_to_mesh
The core alignment function that processes in multiple phases:

#### Phase 1: Initial Alignment (FPFH + RANSAC)
1. **Preparation**
   - Creates multiple resolution point clouds (low, medium, full)
   - Centers and normalizes the point clouds
   - Computes surface normals

2. **Feature Matching**
   - Computes multi-scale FPFH (Fast Point Feature Histograms) features
   - Finds robust correspondences between source and target
   - Uses RANSAC to find initial alignment

#### Phase 2-4: Progressive Refinement (ICP)
Performs Iterative Closest Point (ICP) alignment at increasing resolutions:
1. Low resolution for rough alignment
2. Medium resolution for intermediate refinement
3. Full resolution for final precise alignment

### Validation and Error Handling
- Includes validation checks for alignment quality
- Stores successful parameters in CSV for future reference
- Updates metadata to track alignment success/failure
- Includes multiple retry attempts if alignment fails

### Debug Visualization
- Creates plots at each major step (disabled by default)
- Helps visualize the alignment process and results

## Key Algorithms Used

1. **FPFH (Fast Point Feature Histograms)**
   - Describes local geometry around each point
   - Used for initial feature matching

2. **RANSAC (Random Sample Consensus)**
   - Finds initial transformation between point clouds
   - Robust to outliers

3. **ICP (Iterative Closest Point)**
   - Refines alignment progressively
   - Works at multiple resolutions for better results

## Error Handling
- Multiple retry attempts for failed alignments
- Parameter randomization for robustness
- Metadata updates to track alignment status
- CSV logging of successful parameters

## Usage Notes
- The algorithm works best with clean, well-prepared meshes
- Multiple resolution approach helps avoid local minima
- Validation steps ensure alignment quality
- Debug plots can be enabled for visualization 