import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import cv2
from scipy.ndimage import distance_transform_edt
import time

#I did my best to consolidate my code into one main.py file - but due to a lot of testing that I was doing to make sure it works,
#I  was using a separate file for A4
class ImageMosaicing:
    """
    Consolidated implementation for A2 (Homography), A3 (Warping/Rectification),
    and A4 (Mosaicing & Blending).
    """
    
    def __init__(self):
        self.correspondences = {}
    
    # A.2: Recover Homographies
    
    def get_point_correspondences(self, im1, im2, n_points=8):
        """
        Interactive tool to select corresponding points between two images. Used some reference code that was given in the asssignment.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Image 1 - Click {n_points} points')
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Image 2 - Click corresponding {n_points} points')
        ax2.axis('off')
        
        print(f"Click {n_points} points on Image 1, then {n_points} corresponding points on Image 2")
        print("Close the window when done.")
        
        pts = plt.ginput(n_points * 2, timeout=-1)
        plt.close()
        
        im1_pts = np.array(pts[:n_points])
        im2_pts = np.array(pts[n_points:])
        
        return im1_pts, im2_pts
    
    def visualize_correspondences(self, im1, im2, im1_pts, im2_pts, out_path=None):
        """Visualize point correspondences between two images."""
        fig = plt.figure(figsize=(15, 7))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        ax1.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
        ax1.plot(im1_pts[:, 0], im1_pts[:, 1], 'ro', markersize=8)
        for i, pt in enumerate(im1_pts):
            ax1.text(pt[0], pt[1], str(i+1), color='yellow', fontsize=12, 
                    fontweight='bold', ha='center', va='bottom')
        ax1.set_title('Image 1')
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
        ax2.plot(im2_pts[:, 0], im2_pts[:, 1], 'go', markersize=8)
        for i, pt in enumerate(im2_pts):
            ax2.text(pt[0], pt[1], str(i+1), color='yellow', fontsize=12, 
                    fontweight='bold', ha='center', va='bottom')
        ax2.set_title('Image 2')
        ax2.axis('off')
        
        # Draw connecting lines
        for i in range(len(im1_pts)):
            con = ConnectionPatch(xyA=im1_pts[i], xyB=im2_pts[i], 
                                 coordsA="data", coordsB="data",
                                 axesA=ax1, axesB=ax2, color="cyan", 
                                 linewidth=1, alpha=0.5)
            ax2.add_artist(con)
        
        plt.tight_layout()
        if out_path is not None:
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def computeH(self, im1_pts, im2_pts):
        """
        Compute homography matrix H such that im2_pts = H * im1_pts.
        Returns:
            H: 3 x 3 homography matrix
        """
        n = im1_pts.shape[0]
        A = []
        for i in range(n):
            x, y = im1_pts[i]
            x_prime, y_prime = im2_pts[i]
            A.append([-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime])
        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1, :]
        H = h.reshape(3, 3)
        H = H / H[2, 2]
        return H
    
    def print_system_equations(self, im1_pts, im2_pts):
        """Print the system of equations for homography computation."""
        print("\n" + "="*80)
        print("SYSTEM OF EQUATIONS FOR HOMOGRAPHY")
        print("="*80)
        print("\nHomography equation: [x'; y'; 1] = H * [x; y; 1]")
        print("\nWhere H = [[h11, h12, h13],")
        print("           [h21, h22, h23],")
        print("           [h31, h32, 1  ]]")
        print("\nRearranging to linear form Ah = b:")
        print("  -x, -y, -1,  0,  0,  0, x*x', y*x', x'")
        print("   0,  0,  0, -x, -y, -1, x*y', y*y', y'")
        
    # A.3: Warp the Images (from-scratch, inverse warping)
    
    def warpImageNearestNeighbor(self, im, H, output_shape=None):
        """Warp image using nearest neighbor interpolation (inverse warping) - optimized."""
        h, w = im.shape[:2]
        if output_shape is None:
            output_shape, offset = self._compute_output_shape(im.shape, H)
        else:
            offset = (0, 0)
        out_h, out_w = output_shape
        channels = im.shape[2] if len(im.shape) == 3 else 1
        #computing inverse-homography using np.linalg.inv
        H_inv = np.linalg.inv(H)
        
        # Process in smaller chunks to reduce memory usage (huge issue because my laptop is very slow and has less than 8gb of memory)
        chunk_size = min(1000, out_h)  
        warped = np.zeros((out_h, out_w) + (() if channels == 1 else (channels,)), dtype=im.dtype)
        alpha = np.zeros((out_h, out_w), dtype=np.float32)
        
        for y_start in range(0, out_h, chunk_size):
            y_end = min(y_start + chunk_size, out_h)
            chunk_h = y_end - y_start
            
            y_chunk, x_chunk = np.mgrid[y_start:y_end, 0:out_w]
            x_adj = x_chunk + offset[0]
            y_adj = y_chunk + offset[1]
            
            # Transform coordinates
            coords_out = np.stack([x_adj.flatten(), y_adj.flatten(), np.ones(chunk_h * out_w)])
            coords_in = H_inv @ coords_out
            coords_in = coords_in / coords_in[2, :]
            x_in = coords_in[0, :].reshape(chunk_h, out_w)
            y_in = coords_in[1, :].reshape(chunk_h, out_w)
            
            # Nearest neighbor sampling
            x_in_nn = np.round(x_in).astype(int)
            y_in_nn = np.round(y_in).astype(int)
            valid_mask = (x_in_nn >= 0) & (x_in_nn < w) & (y_in_nn >= 0) & (y_in_nn < h)
            
            # Sample from source image
            if channels == 1:
                warped[y_start:y_end][valid_mask] = im[y_in_nn[valid_mask], x_in_nn[valid_mask]]
            else:
                for c in range(channels):
                    warped[y_start:y_end, :, c][valid_mask] = im[y_in_nn[valid_mask], x_in_nn[valid_mask], c]
            
            alpha[y_start:y_end][valid_mask] = 1.0
        
        return warped, alpha, offset
    
    def warpImageBilinear(self, im, H, output_shape=None):
        h, w = im.shape[:2]
        if output_shape is None:
            output_shape, offset = self._compute_output_shape(im.shape, H)
        else:
            offset = (0, 0)
        out_h, out_w = output_shape
        channels = im.shape[2] if len(im.shape) == 3 else 1
        
        # Pre-compute inverse homography
        H_inv = np.linalg.inv(H)
        
        # Process in chunks to reduce memory usage
        chunk_size = min(1000, out_h)  # Process in smaller chunks
        warped = np.zeros((out_h, out_w) + (() if channels == 1 else (channels,)), dtype=np.float32)
        alpha = np.zeros((out_h, out_w), dtype=np.float32)
        
        for y_start in range(0, out_h, chunk_size):
            y_end = min(y_start + chunk_size, out_h)
            chunk_h = y_end - y_start
            
            # Create coordinate grids for this chunk
            y_chunk, x_chunk = np.mgrid[y_start:y_end, 0:out_w]
            x_adj = x_chunk + offset[0]
            y_adj = y_chunk + offset[1]
            
            # Transform coordinates
            coords_out = np.stack([x_adj.flatten(), y_adj.flatten(), np.ones(chunk_h * out_w)])
            coords_in = H_inv @ coords_out
            coords_in = coords_in / coords_in[2, :]
            x_in = coords_in[0, :].reshape(chunk_h, out_w)
            y_in = coords_in[1, :].reshape(chunk_h, out_w)
            
            # Bilinear interpolation
            x0 = np.floor(x_in).astype(int)
            x1 = x0 + 1
            y0 = np.floor(y_in).astype(int)
            y1 = y0 + 1
            wx = x_in - x0
            wy = y_in - y0
            
            # Create valid mask
            valid_mask = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
            
            if np.any(valid_mask):
                if channels == 1:
                    # Vectorized bilinear interpolation for grayscale
                    I00 = im[y0[valid_mask], x0[valid_mask]]
                    I01 = im[y0[valid_mask], x1[valid_mask]]
                    I10 = im[y1[valid_mask], x0[valid_mask]]
                    I11 = im[y1[valid_mask], x1[valid_mask]]
                    
                    wx_valid = wx[valid_mask]
                    wy_valid = wy[valid_mask]
                    value = ((1-wx_valid)*(1-wy_valid)*I00 + 
                            wx_valid*(1-wy_valid)*I01 + 
                            (1-wx_valid)*wy_valid*I10 + 
                            wx_valid*wy_valid*I11)
                    warped[y_start:y_end][valid_mask] = value
                else:
                    # Vectorized bilinear interpolation for color
                    for c in range(channels):
                        I00 = im[y0[valid_mask], x0[valid_mask], c]
                        I01 = im[y0[valid_mask], x1[valid_mask], c]
                        I10 = im[y1[valid_mask], x0[valid_mask], c]
                        I11 = im[y1[valid_mask], x1[valid_mask], c]
                        
                        wx_valid = wx[valid_mask]
                        wy_valid = wy[valid_mask]
                        value = ((1-wx_valid)*(1-wy_valid)*I00 + 
                                wx_valid*(1-wy_valid)*I01 + 
                                (1-wx_valid)*wy_valid*I10 + 
                                wx_valid*wy_valid*I11)
                        warped[y_start:y_end, :, c][valid_mask] = value
                
                alpha[y_start:y_end][valid_mask] = 1.0
        
        warped = warped.astype(im.dtype)
        return warped, alpha, offset
    
    def _compute_output_shape(self, input_shape, H):
        h, w = input_shape[:2]
        corners = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]]).T
        corners_proj = H @ corners
        corners_proj = corners_proj / corners_proj[2, :]
        x_min = np.min(corners_proj[0, :])
        x_max = np.max(corners_proj[0, :])
        y_min = np.min(corners_proj[1, :])
        y_max = np.max(corners_proj[1, :])
        out_w = int(np.ceil(x_max - x_min))
        out_h = int(np.ceil(y_max - y_min))
        offset = (int(np.floor(x_min)), int(np.floor(y_min)))
        return (out_h, out_w), offset
    
    # A.4: Mosaic Utilities (manual only; no high-level warping APIs)
    
    def create_alpha_mask(self, shape, method='distance'):
        h, w = shape
        if method == 'distance':
            mask = np.ones((h, w), dtype=np.uint8)
            mask[0, :] = 0
            mask[-1, :] = 0
            mask[:, 0] = 0
            mask[:, -1] = 0
            alpha = distance_transform_edt(mask)
            max_val = np.max(alpha)
            if max_val > 0:
                alpha = alpha / max_val
        elif method == 'linear':
            # Vectorized linear alpha mask
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            dist_y = np.minimum(y_coords, h - 1 - y_coords) / (h / 2.0)
            dist_x = np.minimum(x_coords, w - 1 - x_coords) / (w / 2.0)
            alpha = np.minimum(dist_x, dist_y)
            alpha = np.clip(alpha, 0, 1).astype(np.float32)
        return alpha
    
    def blend_images(self, images, alphas, offsets):
        """
        Goal: blend multiple images using weighted averaging (optimized with vectorized operations).
        """
        # Calculate mosaic bounds
        x_min = min(offset[0] for offset in offsets)
        y_min = min(offset[1] for offset in offsets)
        x_max = max(offset[0] + img.shape[1] for img, offset in zip(images, offsets))
        y_max = max(offset[1] + img.shape[0] for img, offset in zip(images, offsets))
        mosaic_w = x_max - x_min
        mosaic_h = y_max - y_min
        channels = images[0].shape[2] if len(images[0].shape) == 3 else 1
        
        # Initialize output arrays
        accumulated = np.zeros((mosaic_h, mosaic_w) + (() if channels == 1 else (channels,)), dtype=np.float32)
        weight_sum = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)
        
        # Process each image
        for img, alpha, offset in zip(images, alphas, offsets):
            x_start = offset[0] - x_min
            y_start = offset[1] - y_min
            h, w = img.shape[:2]
            # Calculate valid region bounds
            x_end = min(x_start + w, mosaic_w)
            y_end = min(y_start + h, mosaic_h)
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            
            if x_end <= x_start or y_end <= y_start:
                continue
                
            # Calculate source region
            src_x_start = x_start - offset[0] + x_min
            src_y_start = y_start - offset[1] + y_min
            src_x_end = src_x_start + (x_end - x_start)
            src_y_end = src_y_start + (y_end - y_start)
            
            # Ensure source bounds are valid
            src_x_start = max(0, src_x_start)
            src_y_start = max(0, src_y_start)
            src_x_end = min(w, src_x_end)
            src_y_end = min(h, src_y_end)
            
            if src_x_end <= src_x_start or src_y_end <= src_y_start:
                continue
            
            # Extract regions
            src_region = img[src_y_start:src_y_end, src_x_start:src_x_end]
            alpha_region = alpha[src_y_start:src_y_end, src_x_start:src_x_end]
            
            # Calculate destination region
            dst_x_start = x_start
            dst_y_start = y_start
            dst_x_end = dst_x_start + (src_x_end - src_x_start)
            dst_y_end = dst_y_start + (src_y_end - src_y_start)
            
            # Vectorized blending
            if channels == 1:
                weighted_img = src_region.astype(np.float32) * alpha_region[..., np.newaxis]
                accumulated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += weighted_img
                weight_sum[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += alpha_region
            else:
                weighted_img = src_region.astype(np.float32) * alpha_region[..., np.newaxis]
                accumulated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += weighted_img
                weight_sum[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += alpha_region
        
        # Normalize by weights (vectorized)
        mask = weight_sum > 0
        if channels == 1:
            mosaic = np.zeros_like(accumulated)
            mosaic[mask] = accumulated[mask] / weight_sum[mask]
        else:
            mosaic = np.zeros_like(accumulated)
            for c in range(channels):
                mosaic[..., c][mask] = accumulated[..., c][mask] / weight_sum[mask]
        
        mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
        return mosaic
    
    # Utility: Rectification for A3
    
    def rectify_image(self, im, src_pts, dst_pts=None):
        """Rectify an image by warping a quadrilateral to a rectangle."""
        if dst_pts is None:
            width = np.linalg.norm(src_pts[1] - src_pts[0])
            height = np.linalg.norm(src_pts[2] - src_pts[0])
            dst_pts = np.array([[0, 0], [width, 0], [0, height], [width, height]])
        H = self.computeH(src_pts, dst_pts)
        rectified, alpha, offset = self.warpImageBilinear(im, H)
        return rectified, H


def run_rectification_demo():
    mosaic = ImageMosaicing()
    img_path = 'drawings_on_a_wall.jpg'
    im = cv2.imread(img_path)
    if im is None:
        print(f"Image not found: {img_path}")
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title('Click 4 corners of a rectangular area: TL, TR, BL, BR')
    plt.show(block=False)
    src_pts = np.array(plt.ginput(4, timeout=-1))
    plt.close()
    rectified, H = mosaic.rectify_image(im, src_pts)
    cv2.imwrite('results/drawings_original.jpg', im)
    cv2.imwrite('results/drawings_rectified_bil.jpg', rectified)
    np.savetxt('results/drawings_homography.txt', H, fmt='%.6f')
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); axes[0].axis('off'); axes[0].set_title('Original')
    axes[1].imshow(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)); axes[1].axis('off'); axes[1].set_title('Rectified (Bilinear)')
    plt.tight_layout(); plt.show()


def run_mosaic_demo():
    """Optimized A4 mosaic using distance-based feather blending."""
    print("Starting optimized mosaic demo...")
    mosaic = ImageMosaicing()
    
    # Load images with memory optimization
    img1_path = 'photo_examples/image_1.jpg'
    img2_path = 'photo_examples/image_2.jpg'
    im1 = cv2.imread(img1_path)
    im2 = cv2.imread(img2_path)
    
    if im1 is None or im2 is None:
        print(f"Images not found: {img1_path}, {img2_path}")
        return
    
    # Resize images if they're too large to prevent memory issues
    max_size = 2000  # Maximum dimension
    if max(im1.shape[:2]) > max_size:
        scale = max_size / max(im1.shape[:2])
        new_h, new_w = int(im1.shape[0] * scale), int(im1.shape[1] * scale)
        im1 = cv2.resize(im1, (new_w, new_h))
        print(f"Resized image 1 to {new_h}x{new_w}")
    
    if max(im2.shape[:2]) > max_size:
        scale = max_size / max(im2.shape[:2])
        new_h, new_w = int(im2.shape[0] * scale), int(im2.shape[1] * scale)
        im2 = cv2.resize(im2, (new_w, new_h))
        print(f"Resized image 2 to {new_h}x{new_w}")
    
    print("Getting point correspondences...")
    im1_pts, im2_pts = mosaic.get_point_correspondences(im1, im2, n_points=8)
    
    # Save correspondences
    np.savetxt('results/mosaic_im1_points.txt', im1_pts, fmt='%.2f')
    np.savetxt('results/mosaic_im2_points.txt', im2_pts, fmt='%.2f')
    
    print("Visualizing correspondences...")
    mosaic.visualize_correspondences(im1, im2, im1_pts, im2_pts, out_path='results/correspondences_visualization.jpg')
    
    print("Computing homography...")
    H = mosaic.computeH(im1_pts, im2_pts)
    np.savetxt('results/mosaic_homography.txt', H, fmt='%.6f')
    
    print("Warping image 2...")
    warped_bil, alpha_bil, offset_bil = mosaic.warpImageBilinear(im2, H)
    
    print("Creating alpha masks...")
    alpha1 = mosaic.create_alpha_mask(im1.shape[:2], method='distance')
    
    print("Blending images...")
    mosaic_img = mosaic.blend_images([im1, warped_bil], [alpha1, alpha_bil], [(0, 0), offset_bil])
    
    print("Saving results...")
    cv2.imwrite('results/mosaic_bilinear.jpg', mosaic_img)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)); axes[0].set_title('Image 1'); axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)); axes[1].set_title('Image 2'); axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB)); axes[2].set_title('Mosaic'); axes[2].axis('off')
    plt.tight_layout(); plt.show()
    
    print("Mosaic demo completed successfully!")


def main():
    print("="*80)
    print("CS 180 Project 3 - A2/A3/A4 Consolidated (main.py)")
    print("="*80)
    print("1) Rectification demo (A3)\n2) Mosaic demo (A2+A3+A4)\n3) Exit")
    try:
        choice = input("Select an option [1/2/3]: ").strip()
    except EOFError:
        choice = '2'
    if choice == '1':
        run_rectification_demo()
    elif choice == '2':
        run_mosaic_demo()
    else:
        print("Bye.")


if __name__ == "__main__":
    main()


