"""
CS 180 Project 2: Fun with Filters and Frequencies!
Computer Vision and Computational Photography

This file contains implementations for:
- Part 1: Fun with Filters (convolutions, edge detection, DoG filters)
- Part 2: Fun with Frequencies (sharpening, hybrid images, blending)
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import os

#part 1
#---------
# part 1.1: CONVOLUTIONS FROM SCRATCH
def convolve_4loops(image, kernel, padding='zero'):
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    
    if padding == 'zero':
        #zero padding
        pad_h = ker_h // 2
        pad_w = ker_w // 2
        padded_img = np.zeros((img_h + 2*pad_h, img_w + 2*pad_w))
        padded_img[pad_h:img_h+pad_h, pad_w:img_w+pad_w] = image
        output_h, output_w = img_h, img_w
        start_h, start_w = 0, 0
    else:
        #no padding
        padded_img = image
        output_h, output_w = img_h - ker_h + 1, img_w - ker_w + 1
        start_h, start_w = 0, 0
    
    result = np.zeros((output_h, output_w))
    
    #four loops
    for i in range(output_h):
        for j in range(output_w):
            for k in range(ker_h):
                for l in range(ker_w):
                    result[i, j] += padded_img[start_h + i + k, start_w + j + l] * kernel[k, l]
    
    return result

def convolve_2loops(image, kernel, padding='zero'):
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    
    if padding == 'zero':
        pad_h = ker_h //2
        pad_w = ker_w // 2
        padded_img = np.zeros((img_h + 2*pad_h, img_w + 2*pad_w))
        padded_img[pad_h:img_h+pad_h, pad_w:img_w+pad_w] = image
        output_h, output_w = img_h, img_w
        start_h, start_w = 0, 0
    else:
        #nopadding
        padded_img = image
        output_h, output_w = img_h - ker_h + 1, img_w - ker_w + 1
        start_h, start_w = 0, 0
    
    result = np.zeros((output_h, output_w))
    
    #two loops with vectorized inner computation
    #vectorized numpy functions should make this slightly faster
    for i in range(output_h):
        for j in range(output_w):
            # Extract patch and compute convolution
            patch = padded_img[start_h + i:start_h + i + ker_h, start_w + j:start_w + j + ker_w]
            result[i, j] = np.sum(patch * kernel)
    
    return result

def create_box_filter(size):
    """ box filter of given size! """
    return np.ones((size, size)) / (size * size)

def create_finite_difference_operators():
    """PART 1.1: Create finite difference operators Dx and Dy"""
    #horizontal gradient
    Dx = np.array([[-1, 1]])  
    # Vertical gradient
    Dy = np.array([[-1], [1]])
    return Dx, Dy

#part 1.2
def compute_gradient_magnitude(image):
    """
    PART 1.2: Compute gradient magnitude using finite difference operators
    Args:
        image: 2D numpy array (grayscale image)
    Returns:
        Ix, Iy, magnitude: partial derivatives and gradient magnitude
    """
    Dx, Dy = create_finite_difference_operators()
    
    # Compute partial derivatives
    Ix = signal.convolve2d(image, Dx, mode='same', boundary='symm')
    Iy = signal.convolve2d(image, Dy, mode='same', boundary='symm')
    
    # Compute gradient magnitude
    magnitude = np.sqrt(Ix**2 + Iy**2)
    
    return Ix, Iy, magnitude

def binarize_edges(magnitude, threshold):
    """PART 1.2: Binarize gradient magnitude image"""
    return (magnitude > threshold).astype(np.uint8) * 255

#part 1.3 

def create_gaussian_kernel(size, sigma):
    """PART 1.3: Create 2D Gaussian kernel"""
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d.T)
    return kernel_2d

def create_dog_filters(size, sigma):
    """PART 1.3: Create Derivative of Gaussian filters"""
    # Create Gaussian kernel
    gaussian = create_gaussian_kernel(size, sigma)
    
    # Create finite difference operators
    Dx, Dy = create_finite_difference_operators()
    
    # Create DoG filters by convolving Gaussian with difference operators
    dog_x = signal.convolve2d(gaussian, Dx, mode='same')
    dog_y = signal.convolve2d(gaussian, Dy, mode='same')
    
    return dog_x, dog_y, gaussian

def apply_dog_filtering(image, size=5, sigma=1.0):
    """PART 1.3: Apply DoG filtering to image"""
    dog_x, dog_y, gaussian = create_dog_filters(size, sigma)
    
    # Apply Gaussian smoothing first
    smoothed = signal.convolve2d(image, gaussian, mode='same', boundary='symm')
    
    # Apply DoG filters
    Ix_dog = signal.convolve2d(smoothed, dog_x, mode='same', boundary='symm')
    Iy_dog = signal.convolve2d(smoothed, dog_y, mode='same', boundary='symm')
    
    # Compute gradient magnitude
    magnitude_dog = np.sqrt(Ix_dog**2 + Iy_dog**2)
    
    return Ix_dog, Iy_dog, magnitude_dog, smoothed

def compute_gradient_orientations(Ix, Iy):
    """PART 1.3: Compute gradient orientations in degrees"""
    orientations = np.arctan2(Iy, Ix)
    return orientations

def visualize_gradient_orientations(Ix, Iy, magnitude):
    """PART 1.3: Visualize gradient orientations using HSV color space"""
    orientations = compute_gradient_orientations(Ix, Iy)
    
    orientations_normalized = (orientations + np.pi) / (2 * np.pi)

    saturation = np.clip(magnitude / magnitude.max(), 0, 1)
    value = np.ones_like(saturation)
    
    hsv = np.zeros((orientations.shape[0], orientations.shape[1], 3))
    hsv[:, :, 0] = orientations_normalized  # Hue
    hsv[:, :, 1] = saturation  # Saturation
    hsv[:, :, 2] = value  # Value
    
    rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return rgb, orientations

#part 2: fun with frequenceis

# part 2.1

def create_unsharp_mask_filter(sigma, amount):
    """PART 2.1: Create unsharp mask filter"""
    # Create Gaussian kernel
    gaussian = create_gaussian_kernel(5, sigma)
    
    # Create unsharp mask filter: I + amount * (I - I_blurred)
    # This is equivalent to: (1 + amount) * I - amount * I_blurred
    # So the filter is: (1 + amount) * delta - amount * gaussian
    delta = np.zeros_like(gaussian)
    delta[gaussian.shape[0]//2, gaussian.shape[1]//2] = 1
    
    unsharp_filter = (1 + amount) * delta - amount * gaussian
    return unsharp_filter

def sharpen_image(image, sigma=1.0, amount=1.0):
    """PART 2.1: Sharpen image using unsharp masking with single convolution operation"""
    # Ensure image is in [0,1] range
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    
    # Create unsharp mask filter for single convolution operation
    unsharp_filter = create_unsharp_mask_filter(sigma, amount)
    
    # Handle both grayscale and color images
    if len(image.shape) == 3:
        # Color image - process each channel separately
        sharpened = np.zeros_like(image)
        
        for i in range(image.shape[2]):
            channel = image[:, :, i]
            # Apply single convolution with unsharp mask filter
            sharpened[:, :, i] = signal.convolve2d(channel, unsharp_filter, mode='same', boundary='symm')
        
        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 1)
    else:
        # Grayscale image - apply single convolution with unsharp mask filter
        sharpened = signal.convolve2d(image, unsharp_filter, mode='same', boundary='symm')
        sharpened = np.clip(sharpened, 0, 1)
    
    return sharpened

#part 2.2 - didn't get to fully debug/calibrate with what went wrong!


def create_frequency_analysis(img1, img2, hybrid, results_dir):
    """PART 2.2: Create frequency analysis visualization"""
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = np.mean(img1, axis=2)
    else:
        img1_gray = img1
        
    if len(img2.shape) == 3:
        img2_gray = np.mean(img2, axis=2)
    else:
        img2_gray = img2
    
    # Compute FFT for each image
    fft1 = np.fft.fftshift(np.fft.fft2(img1_gray))
    fft2 = np.fft.fftshift(np.fft.fft2(img2_gray))
    fft_hybrid = np.fft.fftshift(np.fft.fft2(hybrid))
    
    # Compute log magnitude
    log_mag1 = np.log(np.abs(fft1) + 1)
    log_mag2 = np.log(np.abs(fft2) + 1)
    log_mag_hybrid = np.log(np.abs(fft_hybrid) + 1)
    
    # Create frequency analysis visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Frequency Analysis of Hybrid Image Creation')
    
    # Original images
    axes[0, 0].imshow(img1_gray, cmap='gray')
    axes[0, 0].set_title('Derek (High Freq Source)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_gray, cmap='gray')
    axes[0, 1].set_title('Nutmeg (Low Freq Source)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(hybrid, cmap='gray')
    axes[0, 2].set_title('Hybrid Image')
    axes[0, 2].axis('off')
    
    # Frequency domain
    axes[1, 0].imshow(log_mag1, cmap='hot')
    axes[1, 0].set_title('Derek FFT (Log Magnitude)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(log_mag2, cmap='hot')
    axes[1, 1].set_title('Nutmeg FFT (Log Magnitude)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(log_mag_hybrid, cmap='hot')
    axes[1, 2].set_title('Hybrid FFT (Log Magnitude)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'hybrid_frequency_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Frequency analysis saved to results/hybrid_frequency_analysis.png")

def create_hybrid_image(img1, img2, cutoff_freq1, cutoff_freq2):
    """PART 2.2: Create hybrid image by blending low-freq of img1 and high-freq of img2"""
    # Import alignment functions
    import sys
    sys.path.append('/Users/danieljung/Desktop/CS 180/CS 180 Project 2/hybrid_python')
    from align_image_code import align_images_auto
    
    # Align the images first
    print("Aligning images...")
    img1_aligned, img2_aligned = align_images_auto(img1, img2)
    
    # Convert to grayscale for processing
    if len(img1_aligned.shape) == 3 and img1_aligned.shape[2] == 3:
        img1_gray = cv2.cvtColor(img1_aligned, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1_aligned
    
    if len(img2_aligned.shape) == 3 and img2_aligned.shape[2] == 3:
        img2_gray = cv2.cvtColor(img2_aligned, cv2.COLOR_RGB2GRAY)
    else:
        img2_gray = img2_aligned
    
    # Ensure images are in the right format and 2D
    if len(img1_gray.shape) > 2:
        img1_gray = img1_gray[:, :, 0]  # Take first channel if still 3D
    if len(img2_gray.shape) > 2:
        img2_gray = img2_gray[:, :, 0]  # Take first channel if still 3D
    
    if img1_gray.dtype != np.float32:
        img1_gray = img1_gray.astype(np.float32) / 255.0
    if img2_gray.dtype != np.float32:
        img2_gray = img2_gray.astype(np.float32) / 255.0
    
    # Create Gaussian kernels for low-pass and high-pass filtering
    gaussian_low = create_gaussian_kernel(5, cutoff_freq1)
    gaussian_high = create_gaussian_kernel(5, cutoff_freq2)
    
    # Low-pass filter img1
    low_freq = signal.convolve2d(img1_gray, gaussian_low, mode='same', boundary='symm')
    
    # High-pass filter img2 (original - low-pass filtered)
    low_freq2 = signal.convolve2d(img2_gray, gaussian_high, mode='same', boundary='symm')
    high_freq = img2_gray - low_freq2
    
    # Combine
    hybrid = low_freq + high_freq
    return np.clip(hybrid, 0, 1)

def compute_fourier_transform(image):
    """PART 2.2: Compute and return log magnitude of FFT"""
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shifted) + 1)  # Add 1 to avoid log(0)
    return magnitude

#part 2.3

def create_gaussian_stack(image, levels=6, sigma=1.0):
    """PART 2.3: Create Gaussian stack"""
    from scipy.ndimage import gaussian_filter
    
    gaussian_stack = [image.copy()]
    
    for i in range(1, levels):
        # Use cumulative sigma for each level (more standard approach)
        level_sigma = sigma * (2 ** (i - 1))
        
        # Handle both grayscale and color images
        if len(image.shape) == 3:
            # Color image - process each channel separately
            blurred = np.zeros_like(image)
            for c in range(image.shape[2]):
                blurred[:, :, c] = gaussian_filter(image[:, :, c], sigma=level_sigma)
            gaussian_stack.append(blurred)
        else:
            # Grayscale image
            blurred = gaussian_filter(image, sigma=level_sigma)
            gaussian_stack.append(blurred)
    
    return gaussian_stack

def create_laplacian_stack(gaussian_stack):
    """PART 2.3: Create Laplacian stack from Gaussian stack"""
    laplacian_stack = []
    
    for i in range(len(gaussian_stack) - 1):
        # Laplacian = difference between consecutive Gaussian levels
        laplacian = gaussian_stack[i] - gaussian_stack[i + 1]
        # Ensure values are in valid range
        laplacian = np.clip(laplacian, -1.0, 1.0)
        laplacian_stack.append(laplacian)
    
    # Add the last level of Gaussian stack (lowest frequency)
    laplacian_stack.append(gaussian_stack[-1])
    
    return laplacian_stack

# part 2.4

def create_mask(height, width, seam_type='vertical'):
    """PART 2.4: Create binary mask for blending"""
    mask = np.zeros((height, width))
    
    if seam_type == 'vertical':
        # Create a step function - left side (0) for img1, right side (1) for img2
        mask[:, width//2:] = 1
    elif seam_type == 'horizontal':
        # Create a step function - top (0) for img1, bottom (1) for img2
        mask[height//2:, :] = 1
    
    return mask

def create_smooth_mask(height, width, seam_type='vertical', transition_width=20):
    """PART 2.4: Create smooth mask for blending with transition zone"""
    mask = np.zeros((height, width))
    
    if seam_type == 'vertical':
        # Create a smooth transition around the center
        center_x = width // 2
        for x in range(width):
            if x < center_x - transition_width // 2:
                mask[:, x] = 0  # img1
            elif x > center_x + transition_width // 2:
                mask[:, x] = 1  # img2
            else:
                # Smooth transition
                t = (x - (center_x - transition_width // 2)) / transition_width
                mask[:, x] = t
    elif seam_type == 'horizontal':
        # Create a smooth transition around the center
        center_y = height // 2
        for y in range(height):
            if y < center_y - transition_width // 2:
                mask[y, :] = 0  # img1
            elif y > center_y + transition_width // 2:
                mask[y, :] = 1  # img2
            else:
                # Smooth transition
                t = (y - (center_y - transition_width // 2)) / transition_width
                mask[y, :] = t
    
    return mask

def create_irregular_mask(height, width):
    """PART 2.4: Create irregular mask for blending"""
    mask = np.zeros((height, width))
    
    # Create a more complex mask (e.g., circular or elliptical)
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    
    # Elliptical mask
    mask = ((x - center_x)**2 / (width//3)**2 + (y - center_y)**2 / (height//3)**2) <= 1
    mask = mask.astype(float)
    
    return mask

def multiresolution_blend(img1, img2, mask, levels=6):
    """PART 2.4: Blend two images using multiresolution blending"""
    # Ensure images are in [0,1] range
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
    if img2.dtype == np.uint8:
        img2 = img2.astype(np.float32) / 255.0
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    
    # Create Gaussian stacks with proper sigma progression
    gaussian_stack1 = create_gaussian_stack(img1, levels, sigma=1.0)
    gaussian_stack2 = create_gaussian_stack(img2, levels, sigma=1.0)
    mask_stack = create_gaussian_stack(mask, levels, sigma=1.0)
    
    # Create Laplacian stacks
    laplacian_stack1 = create_laplacian_stack(gaussian_stack1)
    laplacian_stack2 = create_laplacian_stack(gaussian_stack2)
    
    # Blend at each level
    blended_stack = []
    for i in range(len(laplacian_stack1)):
        # Use the blurred mask to blend the Laplacian coefficients
        # Ensure mask has the same number of dimensions as the images
        if len(mask_stack[i].shape) == 2 and len(laplacian_stack1[i].shape) == 3:
            # Broadcast mask to match image dimensions
            mask_expanded = np.stack([mask_stack[i]] * laplacian_stack1[i].shape[2], axis=2)
        else:
            mask_expanded = mask_stack[i]
        
        # Blend using the mask: mask=1 uses img1, mask=0 uses img2
        blended = mask_expanded * laplacian_stack1[i] + (1 - mask_expanded) * laplacian_stack2[i]
        blended_stack.append(blended)
    
    # Reconstruct final image by summing all Laplacian levels
    result = np.zeros_like(blended_stack[0])
    for level in blended_stack:
        result += level
    
    # Ensure result is in [0,1] range
    result = np.clip(result, 0, 1)
    
    return result

#helper functions

def load_grayscale_image(image_path):
    """Load image as grayscale"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image

def load_color_image(image_path):
    """Load image as color (RGB)"""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    # Convert BGR to RGB for proper color display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def display_results(original, result, title, save_path=None):
    """Display and optionally save comparison results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Handle color vs grayscale images
    if len(original.shape) == 3:
        # Color image - images are already in RGB format
        axes[0].imshow(original)
        axes[1].imshow(result)
    else:
        # Grayscale image
        axes[0].imshow(original, cmap='gray')
        axes[1].imshow(result, cmap='gray')
    
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].set_title(title)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved result to: {save_path}")
    
    plt.close()

def display_multiple_results(images, titles, save_path=None):
    """Display multiple images in a grid"""
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(n_images):
        # For gradient images, use appropriate scaling
        if 'Derivative' in titles[i] or 'Gradient' in titles[i]:
            # For derivative images, show both positive and negative values
            im = axes[i].imshow(images[i], cmap='gray', vmin=-np.max(np.abs(images[i])), vmax=np.max(np.abs(images[i])))
        else:
            # For regular images, use normal scaling
            im = axes[i].imshow(images[i], cmap='gray')
        
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved result to: {save_path}")
    
    plt.close()

# image generation functions

def test_part_1_2():
    """Generate Part 1.2: Finite Difference Operator results"""
    print("Generating Part 1.2: Finite Difference Operator...")
    
    # Load the cameraman image
    image_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/cs-180-cameraman.png"
    
    try:
        # Load image as grayscale
        image = load_grayscale_image(image_path)
        
        # Create results directory
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Compute partial derivatives and gradient magnitude
        Ix, Iy, magnitude = compute_gradient_magnitude(image)
        
        # Display results
        images = [image, Ix, Iy, magnitude]
        titles = ['Cameraman Image', 'Partial Derivative Ix', 'Partial Derivative Iy', 'Gradient Magnitude']
        display_multiple_results(images, titles, os.path.join(results_dir, 'part1_2_cameraman_gradients.png'))
        
        # Test different thresholds for edge binarization
        magnitude_mean = np.mean(magnitude)
        magnitude_std = np.std(magnitude)
        
        # Use thresholds based on statistics
        thresholds = [magnitude_mean * 0.5, magnitude_mean, magnitude_mean * 1.5, magnitude_mean * 2]
        
        for threshold in thresholds:
            binary_edges = binarize_edges(magnitude, threshold)
            display_results(image, binary_edges, f'Binary Edges (threshold={threshold:.1f})', 
                           os.path.join(results_dir, f'edges_threshold_{threshold:.1f}.png'))
        
        print("✓ Part 1.2 results generated successfully")
        
    except Exception as e:
        print(f"✗ Error in Part 1.2: {e}")

def test_part_1_3():
    """Generate Part 1.3: Derivative of Gaussian (DoG) Filter results"""
    print("Generating Part 1.3: Derivative of Gaussian (DoG) Filter...")
    
    # Load the cameraman image
    image_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/cs-180-cameraman.png"
    
    try:
        # Load image as grayscale
        image = load_grayscale_image(image_path)
        
        # Create results directory
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # First, show the noisy results from Part 1.2 for comparison
        Ix_noisy, Iy_noisy, magnitude_noisy = compute_gradient_magnitude(image)
        
        # Create Gaussian blurred version
        gaussian_kernel = create_gaussian_kernel(5, 1.0)
        blurred_image = signal.convolve2d(image, gaussian_kernel, mode='same', boundary='symm')
        
        # Apply finite difference operators to blurred image
        Ix_blurred, Iy_blurred, magnitude_blurred = compute_gradient_magnitude(blurred_image)
        
        # Create DoG filters
        dog_x, dog_y, gaussian = create_dog_filters(5, 1.0)
        
        # Apply DoG filters directly to original image
        Ix_dog = signal.convolve2d(image, dog_x, mode='same', boundary='symm')
        Iy_dog = signal.convolve2d(image, dog_y, mode='same', boundary='symm')
        magnitude_dog = np.sqrt(Ix_dog**2 + Iy_dog**2)
        
        # Show the difference between noisy and smoothed results
        images_comparison = [image, Ix_noisy, Iy_noisy, magnitude_noisy, 
                           blurred_image, Ix_blurred, Iy_blurred, magnitude_blurred]
        titles_comparison = ['Original', 'Ix (Noisy)', 'Iy (Noisy)', 'Magnitude (Noisy)',
                           'Blurred', 'Ix (Blurred)', 'Iy (Blurred)', 'Magnitude (Blurred)']
        display_multiple_results(images_comparison, titles_comparison, 
                               os.path.join(results_dir, 'part1_3_noisy_vs_blurred.png'))
        
        # Show DoG filter results
        images_dog = [image, Ix_dog, Iy_dog, magnitude_dog]
        titles_dog = ['Original', 'Ix (DoG)', 'Iy (DoG)', 'Magnitude (DoG)']
        display_multiple_results(images_dog, titles_dog, 
                               os.path.join(results_dir, 'part1_3_dog_results.png'))
        
        # Display the DoG filters as images
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(gaussian, cmap='gray')
        plt.title('Gaussian Filter')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(dog_x, cmap='gray')
        plt.title('DoG X Filter')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(dog_y, cmap='gray')
        plt.title('DoG Y Filter')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'part1_3_dog_filters.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Test edge binarization on DoG results
        magnitude_mean = np.mean(magnitude_dog)
        thresholds = [magnitude_mean * 0.5, magnitude_mean, magnitude_mean * 1.5, magnitude_mean * 2]
        
        for threshold in thresholds:
            binary_edges = binarize_edges(magnitude_dog, threshold)
            display_results(image, binary_edges, f'DoG Binary Edges (threshold={threshold:.1f})', 
                           os.path.join(results_dir, f'cameraman_edges_threshold_{threshold:.1f}.png'))
        
        print("✓ Part 1.3 results generated successfully")
        
    except Exception as e:
        print(f"✗ Error in Part 1.3: {e}")

def test_part_2_1():
    """Test PART 2.1: Image Sharpening with Taj Mahal color image"""
    print("CS 180 Project 2 - Part 2.1: Image Sharpening!")
    print("=" * 60)
    
    # Load the provided Taj Mahal color image
    image_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/taj.jpg"
    
    try:
        # Load image as color
        print("Loading Taj Mahal color image...")
        image = load_color_image(image_path)
        print(f"Image loaded successfully. Shape: {image.shape}")
        
        # Create results directory
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 1. Generate complete unsharp masking process
        print("\n1. Generating complete unsharp masking process...")
        sigma = 0.5
        amount = 0.5
        
        # Convert image to [0,1] range for processing
        if image.max() > 1.0:
            image_normalized = image.astype(np.float32) / 255.0
        else:
            image_normalized = image.astype(np.float32)
        
        # Create Gaussian kernel for blurring
        gaussian_kernel = create_gaussian_kernel(5, sigma)
        
        # Apply blur to each color channel
        blurred = np.zeros_like(image_normalized)
        for c in range(image_normalized.shape[2]):
            blurred[:, :, c] = signal.convolve2d(image_normalized[:, :, c], gaussian_kernel, mode='same', boundary='symm')
        
        # Extract high frequencies
        high_freq = image_normalized - blurred
        
        # Apply sharpening using the new single convolution method
        sharpened = sharpen_image(image_normalized, sigma=sigma, amount=amount)
        
        # Save the complete process
        plt.figure(figsize=(16, 4))
        
        plt.subplot(1, 4, 1)
        plt.imshow(image_normalized)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(blurred)
        plt.title('Blurred (σ=0.5)')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        # High frequencies should be displayed in grayscale
        if len(high_freq.shape) == 3:
            high_freq_gray = np.mean(high_freq, axis=2)
            plt.imshow(high_freq_gray, cmap='gray')
        else:
            plt.imshow(high_freq, cmap='gray')
        plt.title('High Frequencies')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(sharpened)
        plt.title('Sharpened')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'part2_1_complete_process.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Test different sharpening parameters
        print("\n2. Testing different sharpening parameters...")
        sigmas = [0.5, 1.0, 1.5]
        amounts = [0.3, 0.5, 0.8, 1.0]
        
        for sigma in sigmas:
            for amount in amounts:
                print(f"Processing sigma={sigma}, amount={amount}...")
                sharpened = sharpen_image(image_normalized, sigma=sigma, amount=amount)
                plt.imsave(os.path.join(results_dir, f'taj_sharpened_sigma_{sigma}_amount_{amount}.png'), sharpened)
        
        # 2. Create a sharp image, blur it, then sharpen it back
        print("\n2. Testing with artificially blurred image...")
        # Use the Antelope Canyon image as a sharp reference
        sharp_image_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/2.1_test_image.jpg"
        sharp_image = load_color_image(sharp_image_path)
        
        # Convert to [0,1] range for processing
        if sharp_image.max() > 1.0:
            sharp_image_normalized = sharp_image.astype(np.float32) / 255.0
        else:
            sharp_image_normalized = sharp_image.astype(np.float32)
        
        # Blur the sharp image with a much stronger Gaussian blur
        gaussian_blur = create_gaussian_kernel(15, 5.0)  # Much larger kernel and higher sigma for very visible blur
        blurred_image = np.zeros_like(sharp_image_normalized)
        for c in range(sharp_image_normalized.shape[2]):
            blurred_image[:, :, c] = signal.convolve2d(sharp_image_normalized[:, :, c], gaussian_blur, mode='same', boundary='symm')
        
        # Sharpen the blurred image using our unsharp masking method
        sharpened_blurred = sharpen_image(blurred_image, sigma=2.0, amount=1.5)
        
        # Display comparison
        images = [sharp_image_normalized, blurred_image, sharpened_blurred]
        titles = ['Original Sharp Image (Antelope Canyon, Arizona)', 'Artificially Blurred', 'Sharpened Back']
        display_multiple_results(images, titles, os.path.join(results_dir, 'part2_1_sharpening_comparison.png'))
        
        # 3. Show the unsharp mask filter
        print("\n3. Visualizing the unsharp mask filter...")
        unsharp_filter = create_unsharp_mask_filter(sigma=1.0, amount=1.0)
        
        # Display the filter
        plt.figure(figsize=(8, 6))
        plt.imshow(unsharp_filter, cmap='gray')
        plt.title('Unsharp Mask Filter (σ=1.0, amount=1.0)')
        plt.colorbar()
        plt.savefig(os.path.join(results_dir, 'unsharp_mask_filter.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Test with different amounts to show the effect
        print("\n4. Testing different sharpening amounts on Taj Mahal...")
        amounts_test = [0.2, 0.4, 0.6, 0.8]
        
        for amount in amounts_test:
            print(f"Processing amount={amount}...")
            sharpened = sharpen_image(image_normalized, sigma=1.0, amount=amount)
            display_results(image_normalized, sharpened, f'Taj Mahal Sharpened (amount={amount})', 
                           os.path.join(results_dir, f'taj_amount_{amount}.png'))
        
        print("\nPart 2.1 completed successfully!")
        print("All results saved to 'results' directory.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the image path is correct and the image exists.")

def test_part_1_1():
    """Generate Part 1.1: Convolutions from Scratch results"""
    print("Generating Part 1.1: Convolutions from Scratch...")
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Load selfie image
    image_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/cs180-selfie.jpg"
    
    try:
        image = load_grayscale_image(image_path)
        
        # Create 9x9 box filter
        box_filter = create_box_filter(9)
        
        # Test 4-loop implementation
        result_4loops = convolve_4loops(image, box_filter, padding='zero')
        
        # Test 2-loop implementation
        result_2loops = convolve_2loops(image, box_filter, padding='zero')
        
        # Test scipy implementation
        result_scipy = signal.convolve2d(image, box_filter, mode='same', boundary='symm')
        
        # Save results
        plt.imsave(os.path.join(results_dir, 'box_filter_4loops.png'), result_4loops, cmap='gray')
        plt.imsave(os.path.join(results_dir, 'box_filter_2loops.png'), result_2loops, cmap='gray')
        plt.imsave(os.path.join(results_dir, 'box_filter_scipy.png'), result_scipy, cmap='gray')
        
        # Test finite difference operators on selfie
        Dx, Dy = create_finite_difference_operators()
        
        # Apply Dx to selfie
        dx_result = convolve_2loops(image, Dx, padding='zero')
        plt.imsave(os.path.join(results_dir, 'dx_4loops.png'), dx_result, cmap='gray')
        
        print("✓ Part 1.1 results generated successfully")
        
    except Exception as e:
        print(f"✗ Error in Part 1.1: {e}")

def test_part_2_3():
    """Generate Part 2.3: Gaussian and Laplacian Stacks"""
    print("Generating Part 2.3: Gaussian and Laplacian Stacks...")
    
    # Load orange and apple images
    orange_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/spline/orange.jpeg"
    apple_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/spline/apple.jpeg"
    
    try:
        # Load images and normalize to [0,1] range
        orange = plt.imread(orange_path)
        apple = plt.imread(apple_path)
        
        # Normalize to [0,1] range if needed
        if orange.dtype == np.uint8:
            orange = orange.astype(np.float32) / 255.0
        if apple.dtype == np.uint8:
            apple = apple.astype(np.float32) / 255.0
        
        print(f"Orange shape: {orange.shape}, dtype: {orange.dtype}, range: {orange.min():.3f} to {orange.max():.3f}")
        print(f"Apple shape: {apple.shape}, dtype: {apple.dtype}, range: {apple.min():.3f} to {apple.max():.3f}")
        
        # Create results directory
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Create Gaussian and Laplacian stacks for orange
        print("Creating Gaussian and Laplacian stacks for orange...")
        orange_gaussian_stack = create_gaussian_stack(orange, levels=5, sigma=2.0)
        orange_laplacian_stack = create_laplacian_stack(orange_gaussian_stack)
        
        # Create Gaussian and Laplacian stacks for apple
        print("Creating Gaussian and Laplacian stacks for apple...")
        apple_gaussian_stack = create_gaussian_stack(apple, levels=5, sigma=2.0)
        apple_laplacian_stack = create_laplacian_stack(apple_gaussian_stack)
        
        # Save Gaussian stack for orange
        print("Saving orange Gaussian stack...")
        for i, level in enumerate(orange_gaussian_stack):
            plt.imsave(os.path.join(results_dir, f'orange_gaussian_level_{i}.png'), level)
        
        # Save Laplacian stack for orange
        print("Saving orange Laplacian stack...")
        for i, level in enumerate(orange_laplacian_stack):
            plt.imsave(os.path.join(results_dir, f'orange_laplacian_level_{i}.png'), level)
        
        # Save Gaussian stack for apple
        print("Saving apple Gaussian stack...")
        for i, level in enumerate(apple_gaussian_stack):
            plt.imsave(os.path.join(results_dir, f'apple_gaussian_level_{i}.png'), level)
        
        # Save Laplacian stack for apple
        print("Saving apple Laplacian stack...")
        for i, level in enumerate(apple_laplacian_stack):
            plt.imsave(os.path.join(results_dir, f'apple_laplacian_level_{i}.png'), level)
        
        # Create visualization of both stacks
        print("Creating stack visualizations...")
        
        # Orange Gaussian stack visualization
        plt.figure(figsize=(20, 4))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            # Ensure proper color display
            if len(orange_gaussian_stack[i].shape) == 3:
                plt.imshow(orange_gaussian_stack[i])
            else:
                plt.imshow(orange_gaussian_stack[i], cmap='gray')
            plt.title(f'Orange Gaussian Level {i}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'orange_gaussian_stack_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Orange Laplacian stack visualization
        plt.figure(figsize=(20, 4))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            # Laplacian stacks should show colored frequency differences
            if len(orange_laplacian_stack[i].shape) == 3:
                plt.imshow(orange_laplacian_stack[i])
            else:
                plt.imshow(orange_laplacian_stack[i], cmap='gray')
            plt.title(f'Orange Laplacian Level {i}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'orange_laplacian_stack_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Apple Gaussian stack visualization
        plt.figure(figsize=(20, 4))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            # Ensure proper color display
            if len(apple_gaussian_stack[i].shape) == 3:
                plt.imshow(apple_gaussian_stack[i])
            else:
                plt.imshow(apple_gaussian_stack[i], cmap='gray')
            plt.title(f'Apple Gaussian Level {i}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'apple_gaussian_stack_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Apple Laplacian stack visualization
        plt.figure(figsize=(20, 4))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            # Laplacian stacks should show colored frequency differences
            if len(apple_laplacian_stack[i].shape) == 3:
                plt.imshow(apple_laplacian_stack[i])
            else:
                plt.imshow(apple_laplacian_stack[i], cmap='gray')
            plt.title(f'Apple Laplacian Level {i}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'apple_laplacian_stack_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Part 2.3 results generated successfully")
        
    except Exception as e:
        print(f"✗ Error in Part 2.3: {e}")

def test_part_2_2():
    """Test PART 2.2: Hybrid Images using starter code approach"""
    print("CS 180 Project 2 - Part 2.2: Hybrid Images!")
    print("=" * 60)
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    try:
        # Load Derek and Nutmeg images exactly as in starter code
        print("Loading Derek and Nutmeg images...")
        derek_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/hybrid_python/DerekPicture.jpg"
        nutmeg_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/hybrid_python/nutmeg.jpg"
        
        # Load as RGB and normalize to [0,1] exactly like starter code
        derek = plt.imread(derek_path) / 255.0
        nutmeg = plt.imread(nutmeg_path) / 255.0
        
        print(f"Derek shape: {derek.shape}")
        print(f"Nutmeg shape: {nutmeg.shape}")
        
        # Use starter code alignment exactly
        print("Aligning images using starter code...")
        import sys
        sys.path.append('/Users/danieljung/Desktop/CS 180/CS 180 Project 2/hybrid_python')
        from align_image_code import align_images_auto
        from scipy.ndimage import gaussian_filter
        
        derek_aligned, nutmeg_aligned = align_images_auto(derek, nutmeg)
        
        print(f"Derek aligned shape: {derek_aligned.shape}")
        print(f"Nutmeg aligned shape: {nutmeg_aligned.shape}")
        
        # Create hybrid image exactly as in starter code
        print("Creating hybrid image...")
        sigma1 = 2.0  # High-pass filter cutoff for Derek (high freq)
        sigma2 = 3.0  # Low-pass filter cutoff for Nutmeg (low freq)
        
        # Convert to grayscale as in starter code
        if len(derek_aligned.shape) == 3:
            derek_gray = np.mean(derek_aligned, axis=2)
        else:
            derek_gray = derek_aligned
        
        if len(nutmeg_aligned.shape) == 3:
            nutmeg_gray = np.mean(nutmeg_aligned, axis=2)
        else:
            nutmeg_gray = nutmeg_aligned
        
        # High-pass filter Derek (keep high frequencies)
        derek_blurred = gaussian_filter(derek_gray, sigma=sigma1)
        derek_high_freq = derek_gray - derek_blurred
        
        # Low-pass filter Nutmeg (keep low frequencies)
        nutmeg_low_freq = gaussian_filter(nutmeg_gray, sigma=sigma2)
        
        # Combine
        hybrid = derek_high_freq + nutmeg_low_freq
        hybrid = np.clip(hybrid, 0, 1)
        
        # Save hybrid image
        plt.imsave(os.path.join(results_dir, 'hybrid_derek_nutmeg.png'), hybrid, cmap='gray')
        print("Hybrid image saved to results/hybrid_derek_nutmeg.png")
        
        # Create visualization exactly like starter code
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Hybrid Image Creation Process')
        
        # Handle different image shapes for display (exactly like starter code)
        if len(derek_aligned.shape) == 3 and derek_aligned.shape[2] == 2:
            derek_display = np.concatenate([derek_aligned, derek_aligned[:, :, -1:]], axis=2)
        else:
            derek_display = derek_aligned
        
        if len(nutmeg_aligned.shape) == 3 and nutmeg_aligned.shape[2] == 2:
            nutmeg_display = np.concatenate([nutmeg_aligned, nutmeg_aligned[:, :, -1:]], axis=2)
        else:
            nutmeg_display = nutmeg_aligned
        
        axes[0, 0].imshow(derek_display)
        axes[0, 0].set_title('Derek (High Freq Source)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(nutmeg_display)
        axes[0, 1].set_title('Nutmeg (Low Freq Source)')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(hybrid, cmap='gray')
        axes[1, 0].set_title('Hybrid Image (Grayscale)')
        axes[1, 0].axis('off')
        
        # Create color hybrid - ensure 3 channels
        if len(derek_aligned.shape) == 3 and derek_aligned.shape[2] == 2:
            # Convert 2-channel to 3-channel by duplicating the last channel
            derek_3ch = np.concatenate([derek_aligned, derek_aligned[:, :, -1:]], axis=2)
            nutmeg_3ch = np.concatenate([nutmeg_aligned, nutmeg_aligned[:, :, -1:]], axis=2)
        elif len(derek_aligned.shape) == 3 and derek_aligned.shape[2] >= 3:
            derek_3ch = derek_aligned
            nutmeg_3ch = nutmeg_aligned
        else:
            derek_3ch = derek_aligned
            nutmeg_3ch = nutmeg_aligned
        
        # Create color hybrid
        hybrid_color = np.zeros_like(derek_3ch)
        for c in range(derek_3ch.shape[2]):
            derek_channel = derek_3ch[:, :, c]
            nutmeg_channel = nutmeg_3ch[:, :, c]
            
            derek_blurred_c = gaussian_filter(derek_channel, sigma=sigma1)
            derek_high_c = derek_channel - derek_blurred_c
            nutmeg_low_c = gaussian_filter(nutmeg_channel, sigma=sigma2)
            
            hybrid_color[:, :, c] = derek_high_c + nutmeg_low_c
        
        hybrid_color = np.clip(hybrid_color, 0, 1)
        axes[1, 1].imshow(hybrid_color)
        axes[1, 1].set_title('Hybrid Image (Color)')
        plt.imsave(os.path.join(results_dir, 'hybrid_derek_nutmeg_color.png'), hybrid_color)
        print("Color hybrid image saved to results/hybrid_derek_nutmeg_color.png")
        
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'hybrid_creation_process.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("Process visualization saved to results/hybrid_creation_process.png")
        
        # Create frequency analysis
        print("Creating frequency analysis...")
        create_frequency_analysis(derek, nutmeg, hybrid, results_dir)
        
        print("Part 2.2 completed successfully!")
        print("All results saved to 'results' directory.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the required images are available.")

def create_hybrid_pair(img1_path, img2_path, name1, name2, results_dir, sigma1=2.0, sigma2=3.0):
    """Create hybrid image for a pair of images"""
    print(f"Creating hybrid: {name1} + {name2}")
    
    # Load images exactly as in starter code
    img1 = plt.imread(img1_path) / 255.0
    img2 = plt.imread(img2_path) / 255.0
    
    print(f"{name1} shape: {img1.shape}")
    print(f"{name2} shape: {img2.shape}")
    
    # Use starter code alignment exactly
    import sys
    sys.path.append('/Users/danieljung/Desktop/CS 180/CS 180 Project 2/hybrid_python')
    from align_image_code import align_images_auto
    from scipy.ndimage import gaussian_filter
    
    img1_aligned, img2_aligned = align_images_auto(img1, img2)
    
    print(f"{name1} aligned shape: {img1_aligned.shape}")
    print(f"{name2} aligned shape: {img2_aligned.shape}")
    
    # Convert to grayscale as in starter code
    if len(img1_aligned.shape) == 3:
        img1_gray = np.mean(img1_aligned, axis=2)
    else:
        img1_gray = img1_aligned
    
    if len(img2_aligned.shape) == 3:
        img2_gray = np.mean(img2_aligned, axis=2)
    else:
        img2_gray = img2_aligned
    
    # High-pass filter img1 (keep high frequencies)
    img1_blurred = gaussian_filter(img1_gray, sigma=sigma1)
    img1_high_freq = img1_gray - img1_blurred
    
    # Low-pass filter img2 (keep low frequencies)
    img2_low_freq = gaussian_filter(img2_gray, sigma=sigma2)
    
    # Combine
    hybrid = img1_high_freq + img2_low_freq
    hybrid = np.clip(hybrid, 0, 1)
    
    # Save hybrid image
    hybrid_filename = f'hybrid_{name1.lower().replace(" ", "_")}_{name2.lower().replace(" ", "_")}'
    plt.imsave(os.path.join(results_dir, f'{hybrid_filename}.png'), hybrid, cmap='gray')
    print(f"Hybrid image saved to results/{hybrid_filename}.png")
    
    # Create color hybrid - ensure 3 channels
    if len(img1_aligned.shape) == 3 and img1_aligned.shape[2] == 2:
        img1_3ch = np.concatenate([img1_aligned, img1_aligned[:, :, -1:]], axis=2)
        img2_3ch = np.concatenate([img2_aligned, img2_aligned[:, :, -1:]], axis=2)
    elif len(img1_aligned.shape) == 3 and img1_aligned.shape[2] >= 3:
        img1_3ch = img1_aligned
        img2_3ch = img2_aligned
    else:
        img1_3ch = img1_aligned
        img2_3ch = img2_aligned
    
    # Create color hybrid
    hybrid_color = np.zeros_like(img1_3ch)
    for c in range(img1_3ch.shape[2]):
        img1_channel = img1_3ch[:, :, c]
        img2_channel = img2_3ch[:, :, c]
        
        img1_blurred_c = gaussian_filter(img1_channel, sigma=sigma1)
        img1_high_c = img1_channel - img1_blurred_c
        img2_low_c = gaussian_filter(img2_channel, sigma=sigma2)
        
        hybrid_color[:, :, c] = img1_high_c + img2_low_c
    
    hybrid_color = np.clip(hybrid_color, 0, 1)
    plt.imsave(os.path.join(results_dir, f'{hybrid_filename}_color.png'), hybrid_color)
    print(f"Color hybrid image saved to results/{hybrid_filename}_color.png")
    
    return hybrid, hybrid_color

def test_part_2_2_additional():
    """Test PART 2.2: Additional Hybrid Images"""
    print("CS 180 Project 2 - Part 2.2: Additional Hybrid Images!")
    print("=" * 60)
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    try:
        # Elon Musk + Zuckerberg hybrid
        print("Creating Elon Musk + Zuckerberg hybrid...")
        elon_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/elon.jpg"
        zuck_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/zuck.jpg"
        
        elon_zuck_hybrid, elon_zuck_color = create_hybrid_pair(
            elon_path, zuck_path, "Elon", "Zuckerberg", results_dir, sigma1=2.0, sigma2=3.0
        )
        
        # Selfie + Timothy hybrid
        print("\nCreating Selfie + Timothy hybrid...")
        selfie_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/cs180-selfie.jpg"
        timothy_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/timothy_2.2.jpg"
        
        selfie_timothy_hybrid, selfie_timothy_color = create_hybrid_pair(
            selfie_path, timothy_path, "Selfie", "Timothy", results_dir, sigma1=2.0, sigma2=3.0
        )
        
        print("\nPart 2.2 additional hybrids completed successfully!")
        print("All results saved to 'results' directory.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the required images are available.")

def test_part_2_4():
    """Test PART 2.4: Multiresolution Blending"""
    print("CS 180 Project 2 - Part 2.4: Multiresolution Blending!")
    print("=" * 60)
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    try:
        # Test vertical seam blending
        print("Testing vertical seam blending...")
        test_vertical_seam_blending()
        
        # Test irregular mask blending
        print("Testing irregular mask blending...")
        test_irregular_mask_blending()
        
        # Test pyramid visualization
        print("Testing pyramid visualization...")
        test_pyramid_visualization()
        
        # Test blending process visualization
        print("Testing blending process visualization...")
        test_blending_process_visualization()
        
        print("Part 2.4 completed successfully!")
        print("All results saved to 'results' directory.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the required images are available.")

def test_vertical_seam_blending():
    """Test multiresolution blending with vertical seam"""
    print("Testing vertical seam blending...")
    
    # Load apple and orange images
    apple_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/spline/apple.jpeg"
    orange_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/spline/orange.jpeg"
    
    # Load images using matplotlib to preserve colors
    img1 = plt.imread(apple_path)  # Apple
    img2 = plt.imread(orange_path)  # Orange
    
    # Normalize to [0,1] range if needed
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
    if img2.dtype == np.uint8:
        img2 = img2.astype(np.float32) / 255.0
    
    print(f"Apple shape: {img1.shape}, dtype: {img1.dtype}, range: {img1.min():.3f} to {img1.max():.3f}")
    print(f"Orange shape: {img2.shape}, dtype: {img2.dtype}, range: {img2.min():.3f} to {img2.max():.3f}")
    
    # Create vertical mask (1 for apple side, 0 for orange side)
    # Try both binary and smooth masks
    mask_binary = create_mask(img1.shape[0], img1.shape[1], 'vertical')
    mask_smooth = create_smooth_mask(img1.shape[0], img1.shape[1], 'vertical', transition_width=30)
    print(f"Binary mask shape: {mask_binary.shape}, min: {mask_binary.min()}, max: {mask_binary.max()}")
    print(f"Smooth mask shape: {mask_smooth.shape}, min: {mask_smooth.min()}, max: {mask_smooth.max()}")
    
    # Perform multiresolution blending with both masks
    print("Performing multiresolution blending...")
    blended_binary = multiresolution_blend(img1, img2, mask_binary, levels=6)
    blended_smooth = multiresolution_blend(img1, img2, mask_smooth, levels=6)
    
    # Create a more comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle('Multiresolution Blending - Vertical Seam (Oraple)', fontsize=16)
    
    # Display images
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('Image 1 (Apple)', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title('Image 2 (Orange)', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mask_binary, cmap='gray')
    axes[0, 2].set_title('Binary Mask', fontsize=14)
    axes[0, 2].axis('off')
    
    # Show the blended results
    axes[1, 0].imshow(blended_binary)
    axes[1, 0].set_title('Multires Blend (Binary Mask)', fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(blended_smooth)
    axes[1, 1].set_title('Multires Blend (Smooth Mask)', fontsize=14)
    axes[1, 1].axis('off')
    
    # Show smooth mask
    axes[1, 2].imshow(mask_smooth, cmap='gray')
    axes[1, 2].set_title('Smooth Mask', fontsize=14)
    axes[1, 2].axis('off')
    
    # Show simple linear blends for comparison
    simple_blend_binary = mask_binary[:, :, np.newaxis] * img1 + (1 - mask_binary[:, :, np.newaxis]) * img2
    simple_blend_smooth = mask_smooth[:, :, np.newaxis] * img1 + (1 - mask_smooth[:, :, np.newaxis]) * img2
    
    axes[2, 0].imshow(simple_blend_binary)
    axes[2, 0].set_title('Simple Linear (Binary)', fontsize=14)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(simple_blend_smooth)
    axes[2, 1].set_title('Simple Linear (Smooth)', fontsize=14)
    axes[2, 1].axis('off')
    
    # Show the difference between multiresolution and simple blend
    difference = np.abs(blended_smooth - simple_blend_smooth)
    axes[2, 2].imshow(difference)
    axes[2, 2].set_title('Difference (Multires vs Simple)', fontsize=14)
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    plt.savefig('results/multiresolution_blending_vertical.png', dpi=150, bbox_inches='tight')
    print("Saved vertical seam blending result to results/multiresolution_blending_vertical.png")
    
    # Save individual images
    plt.imsave('results/apple_original.png', img1)
    plt.imsave('results/orange_original.png', img2)
    plt.imsave('results/blended_oraple_binary.png', blended_binary)
    plt.imsave('results/blended_oraple_smooth.png', blended_smooth)
    plt.imsave('results/blended_oraple.png', blended_smooth)  # Use smooth as main result
    
    plt.close()
    
    return blended_smooth

def test_irregular_mask_blending():
    """Test multiresolution blending with irregular mask"""
    print("Testing irregular mask blending...")
    
    # Try to load apple and orange images first
    try:
        img1 = cv2.imread('spline/apple.jpeg')
        img2 = cv2.imread('spline/orange.jpeg')
        
        if img1 is None or img2 is None:
            raise FileNotFoundError("Apple/orange images not found")
            
        print("Using apple and orange images for irregular mask blending")
        # Resize images to same size
        img1 = cv2.resize(img1, (400, 300))
        img2 = cv2.resize(img2, (400, 300))
        
    except:
        # Fallback to other images
        try:
            img1 = cv2.imread('cs-180-cameraman.png')
            img2 = cv2.imread('cs180-selfie.jpg')
            
            if img1 is None or img2 is None:
                raise FileNotFoundError("Sample images not found")
                
            # Resize images to same size
            img1 = cv2.resize(img1, (400, 300))
            img2 = cv2.resize(img2, (400, 300))
            
        except:
            print("Using sample images...")
            img1, img2 = create_sample_images()
    
    # Create irregular mask
    mask = create_irregular_mask(img1.shape[0], img1.shape[1])
    
    # Convert images to grayscale for blending
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2
    
    # Perform multiresolution blending
    blended = multiresolution_blend(img1_gray, img2_gray, mask, levels=6)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Multiresolution Blending - Irregular Mask')
    
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1, cmap='gray')
    axes[0, 0].set_title('Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2, cmap='gray')
    axes[0, 1].set_title('Image 2')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('Irregular Mask')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(blended, cmap='gray')
    axes[1, 1].set_title('Blended Result')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save results
    plt.savefig('results/multiresolution_blending_irregular.png', dpi=150, bbox_inches='tight')
    print("Saved irregular mask blending result to results/multiresolution_blending_irregular.png")
    
    plt.close()
    
    return blended

def test_pyramid_visualization():
    """Test and visualize Gaussian and Laplacian pyramids"""
    print("Testing pyramid visualization...")
    
    # Load an image
    try:
        img = cv2.imread('cs-180-cameraman.png', cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("Sample image not found")
    except:
        # Create a sample image
        img = np.random.rand(200, 200) * 255
        img = img.astype(np.uint8)
    
    # Create Gaussian and Laplacian stacks
    gaussian_stack = create_gaussian_stack(img, levels=6)
    laplacian_stack = create_laplacian_stack(gaussian_stack)
    
    # Display Gaussian pyramid
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle('Gaussian and Laplacian Pyramids')
    
    for i in range(6):
        # Gaussian pyramid
        axes[0, i].imshow(gaussian_stack[i], cmap='gray')
        axes[0, i].set_title(f'Gaussian Level {i}')
        axes[0, i].axis('off')
        
        # Laplacian pyramid
        axes[1, i].imshow(laplacian_stack[i], cmap='gray')
        axes[1, i].set_title(f'Laplacian Level {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save results
    plt.savefig('results/pyramid_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved pyramid visualization to results/pyramid_visualization.png")
    
    plt.close()

def test_blending_process_visualization():
    """Visualize the multiresolution blending process step by step"""
    print("Testing blending process visualization...")
    
    # Load apple and orange images
    apple_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/spline/apple.jpeg"
    orange_path = "/Users/danieljung/Desktop/CS 180/CS 180 Project 2/spline/orange.jpeg"
    
    img1 = plt.imread(apple_path)  # Apple
    img2 = plt.imread(orange_path)  # Orange
    
    # Normalize to [0,1] range if needed
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
    if img2.dtype == np.uint8:
        img2 = img2.astype(np.float32) / 255.0
    
    # Create vertical mask
    mask = create_mask(img1.shape[0], img1.shape[1], 'vertical')
    
    # Create Gaussian stacks
    gaussian_stack1 = create_gaussian_stack(img1, levels=6, sigma=2.0)
    gaussian_stack2 = create_gaussian_stack(img2, levels=6, sigma=2.0)
    mask_stack = create_gaussian_stack(mask, levels=6, sigma=2.0)
    
    # Create Laplacian stacks
    laplacian_stack1 = create_laplacian_stack(gaussian_stack1)
    laplacian_stack2 = create_laplacian_stack(gaussian_stack2)
    
    # Visualize the process
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    fig.suptitle('Multiresolution Blending Process - Laplacian Stack Visualization', fontsize=16)
    
    for i in range(6):
        # Show original images and mask at different levels
        if i < len(gaussian_stack1):
            axes[0, i].imshow(gaussian_stack1[i])
            axes[0, i].set_title(f'Apple Gaussian Level {i}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(gaussian_stack2[i])
            axes[1, i].set_title(f'Orange Gaussian Level {i}')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(mask_stack[i], cmap='gray')
            axes[2, i].set_title(f'Mask Level {i}')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/blending_process_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved blending process visualization to results/blending_process_visualization.png")
    plt.close()
    
    # Visualize Laplacian stacks
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    fig.suptitle('Laplacian Stack Visualization', fontsize=16)
    
    for i in range(6):
        if i < len(laplacian_stack1):
            # Show Laplacian levels (handle negative values properly)
            laplacian1 = laplacian_stack1[i]
            laplacian2 = laplacian_stack2[i]
            
            # Normalize Laplacian values for display
            if len(laplacian1.shape) == 3:
                # For color images, show the magnitude
                laplacian1_display = np.mean(np.abs(laplacian1), axis=2)
                laplacian2_display = np.mean(np.abs(laplacian2), axis=2)
            else:
                laplacian1_display = np.abs(laplacian1)
                laplacian2_display = np.abs(laplacian2)
            
            axes[0, i].imshow(laplacian1_display, cmap='gray')
            axes[0, i].set_title(f'Apple Laplacian Level {i}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(laplacian2_display, cmap='gray')
            axes[1, i].set_title(f'Orange Laplacian Level {i}')
            axes[1, i].axis('off')
            
            # Show blended Laplacian
            if len(mask_stack[i].shape) == 2 and len(laplacian_stack1[i].shape) == 3:
                mask_expanded = np.stack([mask_stack[i]] * laplacian_stack1[i].shape[2], axis=2)
            else:
                mask_expanded = mask_stack[i]
            
            blended_laplacian = mask_expanded * laplacian_stack1[i] + (1 - mask_expanded) * laplacian_stack2[i]
            
            # Normalize blended Laplacian for display
            if len(blended_laplacian.shape) == 3:
                blended_display = np.mean(np.abs(blended_laplacian), axis=2)
            else:
                blended_display = np.abs(blended_laplacian)
            
            axes[2, i].imshow(blended_display, cmap='gray')
            axes[2, i].set_title(f'Blended Laplacian Level {i}')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/laplacian_stack_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved Laplacian stack visualization to results/laplacian_stack_visualization.png")
    plt.close()

def create_sample_images():
    """Create sample images for testing if apple/orange images are not available"""
    # Create a simple red and blue image for testing
    height, width = 300, 400
    
    # Red image (left side)
    red_img = np.zeros((height, width, 3), dtype=np.uint8)
    red_img[:, :, 2] = 255  # Red channel
    
    # Blue image (right side) 
    blue_img = np.zeros((height, width, 3), dtype=np.uint8)
    blue_img[:, :, 0] = 255  # Blue channel
    
    return red_img, blue_img

def run_all_tests():
    """Run all tests to generate all results"""
    print("CS 180 Project 2: Running All Tests!")
    print("=" * 60)
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Run all parts
        test_part_1_1()
        test_part_1_2()
        test_part_1_3()
        test_part_2_1()
    test_part_2_3()
    test_part_2_4()
    
    print("\n" + "=" * 60)
    print("All tests completed! Check the 'results' directory for all generated images.")
    print("Website can now be viewed at http://localhost:8000")

def generate_all_results():
    """Generate all project results and save images"""
    print("CS 180 Project 2: Generating All Results!")
    print("=" * 60)
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print("Generating Part 1.1: Convolutions from Scratch...")
    test_part_1_1()
    
    print("\nGenerating Part 1.2: Finite Difference Operator...")
    test_part_1_2()
    
    print("\nGenerating Part 1.3: Derivative of Gaussian...")
    test_part_1_3()
    
    print("\nGenerating Part 2.1: Image Sharpening...")
    test_part_2_1()
    
    print("\nGenerating Part 2.3: Gaussian and Laplacian Stacks...")
    test_part_2_3()
    
    print("\nGenerating Part 2.4: Multiresolution Blending...")
    test_part_2_4()
    
    print("\n" + "=" * 60)
    print("All project results generated successfully!")
    print("Check the 'results' directory for all images.")
    print("Website can be viewed at http://localhost:8000")
    print("=" * 60)

if __name__ == "__main__":
    generate_all_results()