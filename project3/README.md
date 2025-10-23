# CS 180 Project 3: Image Warping and Mosaicing

This project implements comprehensive image warping and mosaicing techniques using homography transformations for advanced computer vision applications.

## Project Overview

The implementation includes all required components:

- **A.1**: Image acquisition and digitization
- **A.2**: Homography recovery from point correspondences using least squares
- **A.3**: Image warping with nearest neighbor and bilinear interpolation
- **A.4**: Image blending and mosaicing with weighted averaging


## Key Features

### Homography Recovery (A.2)
- **`computeH(im1_pts, im2_pts)`**: Recovers 3x3 homography matrices from point correspondences
- Uses least squares optimization for robust results with overdetermined systems
- Handles noisy correspondences gracefully

### Image Warping (A.3)
- **`warpImageNearestNeighbor(im, H)`**: Fast warping with pixel-level accuracy
- **`warpImageBilinear(im, H)`**: High-quality warping with smooth interpolation
- Both methods use inverse warping to eliminate holes in output images
- Custom interpolation implementations (no library dependencies)

### Rectification
- **`demonstrate_rectification()`**: Transforms perspective views into fronto-parallel views
- Perfect for rectifying paintings, posters, or floor tiles
- Uses known geometric constraints for accurate results

### Image Mosaicing (A.4)
- **`create_mosaic(images, homographies)`**: Creates seamless panoramic images
- Advanced alpha blending with distance-based falloff
- Prevents edge artifacts and ghosting effects


## Technical Implementation

### Homography Matrix
The 3x3 homography matrix H represents perspective transformations between planes:
```
H = [[h11, h12, h13],
     [h21, h22, h23],
     [h31, h32, h33]]
```

### Least Squares Solution
For n point correspondences, the system Ah = b is solved using:
```python
h, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
```

### Inverse Warping
To avoid holes in output images, we use inverse warping:
1. For each pixel in the output image
2. Apply inverse homography to find corresponding pixel in source
3. Sample from source using interpolation

### Alpha Blending
Distance-based alpha masks create natural transitions:
```python
alpha = max(0, 1 - distance_from_center / falloff_radius)   
```

## Results

The implementation generates several output images:

- **Point Correspondences**: Visualization of matched points
- **Warped Images**: Both nearest neighbor and bilinear results
- **Rectification**: Perspective to fronto-parallel transformation
- **Mosaics**: Seamlessly blended panoramic images

