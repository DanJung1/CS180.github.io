import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from align_image_code import align_images, align_images_auto

def hybrid_image(im1, im2, sigma1, sigma2):
    """
    Create a hybrid image by combining high frequencies from im1 
    with low frequencies from im2.
    
    Args:
        im1: High frequency image (e.g., Derek)
        im2: Low frequency image (e.g., Nutmeg)
        sigma1: Standard deviation for high-pass filter on im1
        sigma2: Standard deviation for low-pass filter on im2
    
    Returns:
        hybrid: The resulting hybrid image
    """
    # Convert to grayscale if needed
    if len(im1.shape) == 3:
        im1_gray = np.mean(im1, axis=2)
    else:
        im1_gray = im1
    
    if len(im2.shape) == 3:
        im2_gray = np.mean(im2, axis=2)
    else:
        im2_gray = im2
    
    # High-pass filter on im1 (keep high frequencies)
    im1_blurred = gaussian_filter(im1_gray, sigma=sigma1)
    im1_high_freq = im1_gray - im1_blurred
    
    # Low-pass filter on im2 (keep low frequencies)
    im2_low_freq = gaussian_filter(im2_gray, sigma=sigma2)
    
    # Combine the images
    hybrid = im1_high_freq + im2_low_freq
    
    # Ensure values are in [0, 1] range
    hybrid = np.clip(hybrid, 0, 1)
    
    return hybrid

def pyramids(image, N):
    """
    Compute and display Gaussian and Laplacian pyramids.
    
    Args:
        image: Input image
        N: Number of pyramid levels
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = np.mean(image, axis=2)
    else:
        gray_image = image
    
    # Gaussian pyramid
    gaussian_pyramid = [gray_image]
    current = gray_image.copy()
    
    for i in range(N-1):
        # Apply Gaussian filter
        blurred = gaussian_filter(current, sigma=1.0)
        gaussian_pyramid.append(blurred)
        current = blurred
    
    # Laplacian pyramid
    laplacian_pyramid = []
    for i in range(N-1):
        # Laplacian = Gaussian[i] - Gaussian[i+1] (upsampled)
        laplacian = gaussian_pyramid[i] - gaussian_pyramid[i+1]
        laplacian_pyramid.append(laplacian)
    
    # Add the smallest Gaussian as the last Laplacian level
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    # Display Gaussian pyramid
    fig, axes = plt.subplots(2, N, figsize=(15, 6))
    fig.suptitle('Gaussian and Laplacian Pyramids')
    
    for i in range(N):
        # Gaussian pyramid
        axes[0, i].imshow(gaussian_pyramid[i], cmap='gray')
        axes[0, i].set_title(f'Gaussian Level {i}')
        axes[0, i].axis('off')
        
        # Laplacian pyramid
        axes[1, i].imshow(laplacian_pyramid[i], cmap='gray')
        axes[1, i].set_title(f'Laplacian Level {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return gaussian_pyramid, laplacian_pyramid

# First load images
print("Loading images...")

# high sf (high spatial frequency - Derek)
im1 = plt.imread('./DerekPicture.jpg')/255.

# low sf (low spatial frequency - Nutmeg)
im2 = plt.imread('./nutmeg.jpg')/255

print("Aligning images...")
# Next align images (this code is provided, but may be improved)
# Using automatic alignment for non-interactive execution
im1_aligned, im2_aligned = align_images_auto(im1, im2)

print("Creating hybrid image...")
## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

sigma1 = 2.0  # High-pass filter cutoff for Derek (high freq)
sigma2 = 3.0  # Low-pass filter cutoff for Nutmeg (low freq)
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Hybrid Image Creation Process')

# Handle different image shapes for display
print(f"im1_aligned shape: {im1_aligned.shape}")
print(f"im2_aligned shape: {im2_aligned.shape}")

# Convert to proper format for display
if len(im1_aligned.shape) == 3 and im1_aligned.shape[2] == 2:
    # Convert 2-channel to 3-channel by duplicating the last channel
    im1_display = np.concatenate([im1_aligned, im1_aligned[:, :, -1:]], axis=2)
else:
    im1_display = im1_aligned

if len(im2_aligned.shape) == 3 and im2_aligned.shape[2] == 2:
    # Convert 2-channel to 3-channel by duplicating the last channel
    im2_display = np.concatenate([im2_aligned, im2_aligned[:, :, -1:]], axis=2)
else:
    im2_display = im2_aligned

axes[0, 0].imshow(im1_display)
axes[0, 0].set_title('Derek (High Freq Source)')
axes[0, 0].axis('off')

axes[0, 1].imshow(im2_display)
axes[0, 1].set_title('Nutmeg (Low Freq Source)')
axes[0, 1].axis('off')

axes[1, 0].imshow(hybrid, cmap='gray')
axes[1, 0].set_title('Hybrid Image')
axes[1, 0].axis('off')

# Show the hybrid image in color if original images were color
if len(im1_aligned.shape) == 3 and im1_aligned.shape[2] >= 3:
    # For color hybrid, we can apply the same process to each channel
    hybrid_color = np.zeros_like(im1_aligned)
    for c in range(im1_aligned.shape[2]):
        hybrid_color[:, :, c] = hybrid_image(im1_aligned[:, :, c], im2_aligned[:, :, c], sigma1, sigma2)
    axes[1, 1].imshow(hybrid_color)
    axes[1, 1].set_title('Hybrid Image (Color)')
else:
    axes[1, 1].imshow(hybrid, cmap='gray')
    axes[1, 1].set_title('Hybrid Image (Grayscale)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# Save the hybrid image
import os
os.makedirs('../results', exist_ok=True)
plt.imsave('../results/hybrid_derek_nutmeg.png', hybrid, cmap='gray')
print("Hybrid image saved to results/hybrid_derek_nutmeg.png")

# Save color hybrid if applicable
if len(im1_aligned.shape) == 3 and im1_aligned.shape[2] >= 3:
    plt.imsave('../results/hybrid_derek_nutmeg_color.png', hybrid_color)
    print("Color hybrid image saved to results/hybrid_derek_nutmeg_color.png")

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
print("Computing pyramids...")
gaussian_pyramid, laplacian_pyramid = pyramids(hybrid, N)

# Save pyramid results
for i, (gauss, laplacian) in enumerate(zip(gaussian_pyramid, laplacian_pyramid)):
    plt.imsave(f'../results/gaussian_pyramid_level_{i}.png', gauss, cmap='gray')
    plt.imsave(f'../results/laplacian_pyramid_level_{i}.png', laplacian, cmap='gray')

print("Pyramid images saved to results folder")