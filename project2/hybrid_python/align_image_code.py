import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr



def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def get_points_auto(im1, im2):
    """Automatic point selection for non-interactive use"""
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    
    # Select points at 1/4 and 3/4 of image dimensions
    p1 = (w1 * 0.25, h1 * 0.25)
    p2 = (w1 * 0.75, h1 * 0.75)
    p3 = (w2 * 0.25, h2 * 0.25)
    p4 = (w2 * 0.75, h2 * 0.75)
    
    print(f'Using automatic points: p1={p1}, p2={p2}, p3={p3}, p4={p4}')
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale)
    else:
        im2 = sktr.rescale(im2, 1./dscale)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    
    # Handle different number of channels
    if len(im1.shape) == 3 and len(im2.shape) == 2:
        im2 = np.stack([im2] * im1.shape[2], axis=2)
    elif len(im1.shape) == 2 and len(im2.shape) == 3:
        im1 = np.stack([im1] * im2.shape[2], axis=2)
    elif len(im1.shape) == 3 and len(im2.shape) == 3 and im1.shape[2] != im2.shape[2]:
        # Handle different number of channels (e.g., RGB vs RG)
        min_channels = min(im1.shape[2], im2.shape[2])
        im1 = im1[:, :, :min_channels]
        im2 = im2[:, :, :min_channels]
    
    # Update dimensions after channel handling
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    
    # Final check
    if im1.shape != im2.shape:
        print(f"Warning: Images still have different shapes: {im1.shape} vs {im2.shape}")
        # Use the smaller dimensions
        min_h = min(im1.shape[0], im2.shape[0])
        min_w = min(im1.shape[1], im2.shape[1])
        im1 = im1[:min_h, :min_w]
        im2 = im2[:min_h, :min_w]
    
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

def align_images_auto(im1, im2):
    """Non-interactive version of align_images"""
    pts = get_points_auto(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    # Skip rotation for automatic alignment to avoid unwanted rotation
    # im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


if __name__ == "__main__":
    # 1. load the image
    # 2. align the two images by calling align_images
    # Now you are ready to write your own code for creating hybrid images!
    pass
