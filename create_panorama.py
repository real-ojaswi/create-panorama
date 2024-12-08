#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import least_squares
import pandas as pd


# In[ ]:


def SIFTDetector(img1, img2, output_dir, num_matches=None):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift_matcher = cv2.SIFT_create()
    kp1, d1 = sift_matcher.detectAndCompute(img1_gray, None)
    kp2, d2 = sift_matcher.detectAndCompute(img2_gray, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher()

    # Match descriptors between the two images
    matches = bf.match(d1, d2)
    
    # Sort matches by distance (ascending order)
    matches = sorted(matches, key=lambda val: val.distance)

    # Initialize lists to store coordinates of matched keypoints
    matched_coords_img1 = []
    matched_coords_img2 = []

    # Extract coordinates from matches
    if num_matches:
        selected_matches = matches[:num_matches]
    else:
        selected_matches = matches

    for match in selected_matches:
        # Get the matching keypoints coordinates
        matched_coords_img1.append(kp1[match.queryIdx].pt)
        matched_coords_img2.append(kp2[match.trainIdx].pt)
    
    # Draw the matches on the images
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, selected_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Save the matched image
    cv2.imwrite(output_dir, matched_img)
    print('SIFT matched image saved!')

    # Return the matched image and the coordinates of the matched keypoints
    return np.array(matched_coords_img1), np.array(matched_coords_img2)


# In[ ]:


def compute_homography(src_pts, dst_pts):
    num_points = src_pts.shape[0]
    A = np.zeros((2 * num_points, 8), dtype=np.float64)
    B = np.zeros((2 * num_points, 1), dtype=np.float64)
    
    for i in range(num_points):
        x_src, y_src = src_pts[i]
        x_dst, y_dst = dst_pts[i]
        
        # Equation 1 for x' (destination x-coordinate)
        A[2 * i] = [x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src]
        B[2 * i] = x_dst
        
        # Equation 2 for y' (destination y-coordinate)
        A[2 * i+1] = [0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src]
        B[2 * i+1] = y_dst

    # Solve the system A * h = B using least squares
    h = np.linalg.lstsq(A, B, rcond=None)[0]
    
    # Homography matrix is 3x3, reshape and append [1] to complete it
    H = np.append(h, 1).reshape(3, 3)
    
    return H


# In[ ]:


def ransac_homography(src_pts, dst_pts, epsilon=0.3, sigma=1.5):
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    
    threshold = 20 * sigma  # Threshold based on sigma
    num_points = src_pts.shape[0]
    
    # Calculate number of iterations based on epsilon (false correspondence probability)
    P = 0.999
    num_iterations = int(np.ceil(np.log(1 - P) / np.log(1 - (1 - epsilon) ** 30)))
    print(num_iterations)

    max_inliers = 0
    best_H = None
    best_inliers = np.zeros(num_points, dtype=bool)
    src_pts_homogeneous = np.hstack((src_pts, np.ones((num_points, 1))))
    
    for _ in range(num_iterations):
        # Randomly sample 4 point correspondences
        sample_indices = np.random.randint(num_points, size=20)
        src_sample = src_pts[sample_indices]
        dst_sample = dst_pts[sample_indices]

        # Compute homography using these 4 points
        H = compute_homography(src_sample, dst_sample)

        # Transform all src points using H
        dst_pts_pred = np.dot(H, src_pts_homogeneous.T).T
        
        # Normalize the predicted points to inhomogeneous coordinates
        dst_pts_pred[:, :-1] /= dst_pts_pred[:, -1:]  # Normalize by the last column

        # Calculate the distance between predicted and actual destination points
        distances = np.linalg.norm(dst_pts_pred[:, :-1] - dst_pts, axis=1)

        # Identify inliers based on the distance threshold (3 * sigma)
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        # Update the best model if the current one has more inliers
        if num_inliers > max_inliers:
            print('Updating the inlier count!')
            max_inliers = num_inliers
            best_H = H
            best_inliers = inliers
            if max_inliers > (0.9) * num_points:
                print('Breaking!!')
                break

    # Recompute homography using all inliers if available
    if np.any(best_inliers):
        inlier_src_pts = src_pts[best_inliers]
        inlier_dst_pts = dst_pts[best_inliers]
        best_H = compute_homography(inlier_src_pts, inlier_dst_pts)

    return best_H, best_inliers


# In[ ]:


def plot_inliers(src_img, dst_img, src_pts, dst_pts, inliers, save_path='inlier_matches.png'):
    def random_color():
        """Generate a random color in BGR format."""
        return tuple(np.random.randint(0, 256, size=3).tolist())

    # Concatenate images side by side
    combined_img = np.hstack((src_img, dst_img))
    
    # Get dimensions for offsets
    width = src_img.shape[1]
    
    # Draw inlier matches
    for i in range(len(src_pts)):
        
        # Source point
        pt1 = (int(src_pts[i][0]), int(src_pts[i][1]))
        # Destination point (offset by width of the source image)
        pt2 = (int(dst_pts[i][0]) + width, int(dst_pts[i][1]))
        if inliers[i]:  # Only plot inliers            
            # Draw green line between points
            cv2.line(combined_img, pt1, pt2, color=(0, 255, 0), thickness=1)
            cv2.circle(combined_img, pt1, radius=2, color=(0, 255, 0), thickness=-1)  # Red circle
            cv2.circle(combined_img, pt2, radius=2, color=(0, 255, 0), thickness=-1)  # Red circle
        else:
            cv2.line(combined_img, pt1, pt2, color=(0, 0, 255), thickness=1)
            # Draw points
            cv2.circle(combined_img, pt1, radius=2, color=(0, 0, 255), thickness=-1)  # Red circle
            cv2.circle(combined_img, pt2, radius=2, color=(0, 0, 255), thickness=-1)  # Red circle
        

    
    # Save the combined image using OpenCV
    cv2.putText(combined_img, 'Inlier Matches', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(save_path, combined_img)
    print(f'Inlier matches image saved at {save_path}')


# In[ ]:


def create_mask_from_images(ref_img, images, homographies):
    # Initialize dimensions for the reference image
    mid_height, mid_width = ref_img.shape[:2]
    
    # Initialize bounding box for the overlapping region
    y_min, x_min = 0, 0
    y_max, x_max = mid_height, mid_width
    
    # Loop through all surrounding images
    for i, img in enumerate(images):
        img_height, img_width = img.shape[:2]
        
        # Define corner points of the current image in homogeneous coordinates
        boundary_pts = np.array([
            [0, 0, 1],            # Top-left
            [0, img_height, 1],    # Bottom-left
            [img_width, 0, 1],     # Top-right
            [img_width, img_height, 1]  # Bottom-right
        ])
        
        # Transform boundary points using the corresponding homography matrix
        transformed_pts = []
        for pt in boundary_pts:
            transformed_pt = np.dot(homographies[i], pt.T)
            transformed_pt /= transformed_pt[2]  # Normalize by homogeneous coordinate
            transformed_pts.append(transformed_pt[:2])  # Only need x, y coordinates
        
        # Convert transformed points to integer coordinates
        transformed_pts = np.array(transformed_pts, dtype=int)
        
        # Update the bounding box based on the transformed points
        x_trans_min, y_trans_min = np.min(transformed_pts[:, 0]), np.min(transformed_pts[:, 1])
        x_trans_max, y_trans_max = np.max(transformed_pts[:, 0]), np.max(transformed_pts[:, 1])
        
        # Adjust the overall bounding box for all transformed images
        x_min = min(x_min, x_trans_min)
        y_min = min(y_min, y_trans_min)
        x_max = max(x_max, x_trans_max)
        y_max = max(y_max, y_trans_max)
    
    # Create the mask with dimensions based on the bounding box
    mask_height = y_max - y_min
    mask_width = x_max - x_min
    mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)  # 3-channel mask
    
    return mask, x_min, y_min


# In[ ]:


def reprojection_error(h, src_pts, dst_pts):
    # Convert the 8-parameter vector into a full 3x3 homography matrix
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]  # We fix H33 = 1
    ])
    
    # Apply the homography to src_pts
    src_pts_h = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))  # Make them homogeneous
    projected_pts_h = (H @ src_pts_h.T).T  # Transform the points
    
    # Convert back to Cartesian coordinates
    projected_pts = projected_pts_h[:, :2] / projected_pts_h[:, 2:3]
    
    # Compute the reprojection error
    error = dst_pts[:, :2] - projected_pts
    
    return error.flatten()

def refine_homography_lm(src_pts, dst_pts, initial_H):
    """Library implementation of LM algorithm"""
    # Flatten the initial homography matrix to a vector of 8 parameters
    h0 = initial_H.flatten()[:8]  # We exclude the last element (H33)
    
    # Use LM optimization to minimize reprojection error
    result = least_squares(reprojection_error, h0, args=(src_pts, dst_pts))
    
    # Rebuild the refined homography matrix
    h_optimized = result.x
    refined_H = np.array([
        [h_optimized[0], h_optimized[1], h_optimized[2]],
        [h_optimized[3], h_optimized[4], h_optimized[5]],
        [h_optimized[6], h_optimized[7], 1.0]  # Set H33 to 1
    ])
    
    return refined_H


# In[ ]:


# Function to compute the Jacobian matrix
def compute_jacobian(h, src_pts):
    n = len(src_pts)
    J = np.zeros((2 * n, 8))  # Jacobian will have 2*n rows (x, y residuals) and 8 columns (homography parameters)
    
    for i in range(n):
        x, y = src_pts[i]
        denom = (h[6] * x + h[7] * y + 1)  # Denominator from the homography equation
        J[2*i, 0] = x / denom
        J[2*i, 1] = y / denom
        J[2*i, 2] = 1 / denom
        J[2*i, 6] = -x * (h[0] * x + h[1] * y + h[2]) / denom**2
        J[2*i, 7] = -y * (h[0] * x + h[1] * y + h[2]) / denom**2
        
        J[2*i+1, 3] = x / denom
        J[2*i+1, 4] = y / denom
        J[2*i+1, 5] = 1 / denom
        J[2*i+1, 6] = -x * (h[3] * x + h[4] * y + h[5]) / denom**2
        J[2*i+1, 7] = -y * (h[3] * x + h[4] * y + h[5]) / denom**2

    return J

def refine_homography_lm_own(src_pts, dst_pts, initial_H, max_iter=10000, tol=1e-6, lambda_init=1e-3):
    """Custom implementation of LM algorithm"""
    h = initial_H.flatten()[:8]  # Initial guess for h, ignoring H33
    J = compute_jacobian(h, src_pts)
    JTJ = J.T @ J  # Approximation to the Hessian
    tau= 0.5
    lambda_init= tau*np.max(np.diagonal(JTJ))

    lambda_factor = 2 # Multiplicative factor for lambda_a adjustment
    lambda_a = lambda_init  # Initial lambda_a value (controls the damping)
    
    for i in range(max_iter):
        error = (reprojection_error(h, src_pts, dst_pts))  # Compute the reprojection error
        cost = np.linalg.norm(error)**2
        J = compute_jacobian(h, src_pts)  # Compute the Jacobian matrix
        JTJ = J.T @ J  # Approximation to the Hessian
        JT_error = J.T @ error  # Gradient

        # Update using LM step: (JTJ + lambda_a * I) * Δh = JT_error
        identity = np.eye(JTJ.shape[0])
        update = np.linalg.inv(JTJ + lambda_a * identity) @ JT_error

        # Update h
        new_h = h + update

        # Calculate new error
        new_error = (reprojection_error(new_h, src_pts, dst_pts))
        new_cost = np.linalg.norm(new_error)

        rho= (cost-new_cost)/(np.dot(update, np.dot(J.T, error)) + lambda_a* np.dot(update.T, np.dot(identity, update)))
        lambda_a= lambda_a * max(1/3, 1-(2*rho-1)**3)
        h= new_h
        
        if np.max(update)<1e-13:
            break

    # Rebuild the refined homography matrix
    refined_H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]  # Set H33 to 1
    ])
    
    return refined_H


# In[ ]:


def create_panorama(H_matrices, images):

    # Determine the index of the central image
    center_idx = len(images) // 2
    print(f"Center image index: {center_idx}")
    
    # Extract the central image
    center_image = images[center_idx]
    
    # Remove the center image from the list of images to be processed
    other_images = np.delete(images, center_idx, axis=0)
    
    # Initialize homography list relative to the center image
    H_rel = []
    
    # Compute homographies for images to the left of the center image
    H_left = np.dot(H_matrices[1], H_matrices[0])
    H_rel.append(H_left)
    
    # Add the homography for the image directly adjacent to the center
    H_rel.append(H_matrices[1])
    
    # Compute inverse homographies for images to the right of the center
    H_inv_right = np.linalg.inv(H_matrices[2])
    H_right_combined = np.dot(H_inv_right, np.linalg.inv(H_matrices[3]))
    
    # Append homographies for the right-side images
    H_rel.append(H_inv_right)
    H_rel.append(H_right_combined)
    
    # Generate the mask for the central image and compute necessary offsets
    mask, x_offset, y_offset = create_mask_from_images(center_image, other_images, H_rel)
    
    return mask, H_rel, x_offset, y_offset


# In[ ]:


def get_relative_homography(H_matrices):

    # Initialize homography list relative to the center image
    H_rel = []
    
    # Compute homographies for images to the left of the center image
    H_left = np.dot(H_matrices[1], H_matrices[0])
    H_rel.append(H_left)
    
    # Add the homography for the image directly adjacent to the center
    H_rel.append(H_matrices[1])
    
    # Compute inverse homographies for images to the right of the center
    H_inv_right = np.linalg.inv(H_matrices[2])
    H_right_combined = np.dot(H_inv_right, np.linalg.inv(H_matrices[3]))
    
    # Append homographies for the right-side images
    H_rel.append(H_inv_right)
    H_rel.append(H_right_combined)
    

    
    return H_rel


# In[ ]:


def apply_homography(src_img, H, height, width):
    # Initialize the output image with zeros

    
    warped_img = np.zeros((height, width, src_img.shape[2]), dtype=src_img.dtype)
    
    # Compute the inverse of the homography matrix for backward mapping
    H_inv = np.linalg.inv(H)
    
    epsilon= 0.0001

    # Iterate over every pixel in the output image
    for y in range(height):
        for x in range(width):
            # Create homogeneous coordinate for the current pixel
            dest_coord = np.array([x, y, 1])
            
            # Map the pixel from the output image back to the source image using inverse homography
            src_coord = H_inv @ dest_coord
            src_coord /= (src_coord[2]+epsilon)  # Convert to Cartesian coordinates

            x_src, y_src = src_coord[0], src_coord[1]

            # Check if the mapped source coordinates are within the valid range of the source image
            if 0 <= x_src < src_img.shape[1] and 0 <= y_src < src_img.shape[0]:
                # Perform bilinear interpolation to compute the pixel value
                x1, y1 = int(x_src), int(y_src)
                x2, y2 = min(x1 + 1, src_img.shape[1] - 1), min(y1 + 1, src_img.shape[0] - 1)

                a = x_src - x1
                b = y_src - y1

                # Calculate the interpolated pixel value
                interpolated_value = (
                    (1 - a) * (1 - b) * src_img[y1, x1] +
                    a * (1 - b) * src_img[y1, x2] +
                    (1 - a) * b * src_img[y2, x1] +
                    a * b * src_img[y2, x2]
                )

                # Assign the computed pixel value to the output image
                warped_img[y, x] = interpolated_value

    return warped_img


# In[ ]:


def stitch_images_into_panorama(center_img, images, homographies, mask, x_offset, y_offset, output_dir='stiched_image.png'):
    # Get the size of the panorama from the mask
    panorama_height, panorama_width = mask.shape[:2]
    
    # Initialize an empty panorama
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=center_img.dtype)

    # Offset matrix for translating coordinates based on x_offset and y_offset
    translation_matrix = np.array([
        [1, 0, -x_offset],
        [0, 1, -y_offset],
        [0, 0, 1]
    ], dtype=np.float64)

    # Add the central image directly to the panorama without homography
    panorama[-y_offset:center_img.shape[0] - y_offset, -x_offset:center_img.shape[1] - x_offset] = center_img

    # Loop through each surrounding image and warp it
    for i, img in enumerate(images):
        # Combine the homography with the translation matrix
        homography = np.dot(translation_matrix, homographies[i])
        
        # Warp the current image using the homography and place it in the panorama
        warped_image = apply_homography(img, homography, panorama_height, panorama_width)
        
        # Create a mask for the warped image to only copy valid pixels
        warped_mask = np.sum(warped_image, axis=-1) > 0
        
        # Blend the warped image into the panorama
        if np.all(panorama[warped_mask] == 0):
            # If the pixel value in panorama is 0, replace it with the warped_image pixel value
            panorama[warped_mask] = warped_image[warped_mask]
        else:
            # If the pixel value is non-zero, take the average of the two images
            panorama[warped_mask] = (0*panorama[warped_mask] + 1*warped_image[warped_mask])
    cv2.imwrite(output_dir, panorama)

    return panorama


# In[ ]:


### choose which set of images you want to operate on
# provided = provided with images
# own = author's own images

# set= 'provided'
set= 'own'


# In[ ]:


if set== 'provided':
    img1= cv2.imread('1.jpg')
    img2= cv2.imread('2.jpg')
    img3= cv2.imread('3.jpg')
    img4= cv2.imread('4.jpg')
    img5= cv2.imread('5.jpg')
elif set=='own':
    img1= cv2.imread('o11.jpeg')
    img2= cv2.imread('o22.jpeg')
    img3= cv2.imread('o33.jpeg')
    img4= cv2.imread('o44.jpeg')
    img5= cv2.imread('o55.jpeg')


# In[ ]:


mp12_1, mp12_2= SIFTDetector(img1, img2, f'sift12_{set}.png', 100)
H12, inliers12= ransac_homography(mp12_1, mp12_2)
plot_inliers(img1, img2, mp12_1, mp12_2, inliers12, f'inliers12_{set}.png')


# In[ ]:


mp12_1_in= mp12_1[inliers12==True]
mp12_2_in= mp12_2[inliers12==True]
H12_r= refine_homography_lm(mp12_1_in, mp12_2_in, H12)


# In[ ]:


mp23_1, mp23_2= SIFTDetector(img2, img3, f'sift23_{set}.png', 100)
H23, inliers23= ransac_homography(mp23_1, mp23_2)
plot_inliers(img2, img3, mp23_1, mp23_2, inliers23, f'inliers23_{set}.png')


# In[ ]:


mp23_1_in= mp23_1[inliers23==True]
mp23_2_in= mp23_2[inliers23==True]
H23_r= refine_homography_lm(mp23_1_in, mp23_2_in, H23)


# In[ ]:


mp34_1, mp34_2= SIFTDetector(img3, img4, f'sift34_{set}.png', 100)
H34, inliers34= ransac_homography(mp34_1, mp34_2)
plot_inliers(img3, img4, mp34_1, mp34_2, inliers34, f'inliers34_{set}.png')


# In[ ]:


mp34_1_in= mp34_1[inliers34==True]
mp34_2_in= mp34_2[inliers34==True]
H34_r= refine_homography_lm(mp34_1_in, mp34_2_in, H34)


# In[ ]:


mp45_1, mp45_2= SIFTDetector(img4, img5, f'sift45_{set}.png', 100)
H45, inliers45= ransac_homography(mp45_1, mp45_2)
plot_inliers(img4, img5, mp45_1, mp45_2, inliers45, f'inliers45_{set}.png')


# In[ ]:


mp45_1_in= mp45_1[inliers45==True]
mp45_2_in= mp45_2[inliers45==True]
H45_r= refine_homography_lm(mp45_1_in, mp45_2_in, H45)


# In[ ]:


error_before_lm= np.linalg.norm(reprojection_error(H45.flatten()[:8], mp45_1_in, mp45_2_in))
error_after_lm= np.linalg.norm(reprojection_error(H45_r.flatten()[:8], mp45_1_in, mp45_2_in))


# In[ ]:


error_before_lm


# In[ ]:


error_after_lm


# In[ ]:


images= [img1, img2, img3, img4, img5]
homographies= [H12, H23, H34, H45]
homographies_r= [H12_r, H23_r, H34_r, H45_r]


# #### With only RANSAC

# In[ ]:


H_rel= get_relative_homography(homographies)


# In[ ]:


mask, x_offset, y_offset= create_mask_from_images(img3, [img1, img2, img4, img5], H_rel)


# In[ ]:


panorama= stitch_images_into_panorama(img3, [img1, img2, img4, img5], H_rel, mask, x_offset, y_offset, f'stitchwithRANSAC_{set}.png')


# #### After refining with LM

# In[ ]:


H_rel= get_relative_homography(homographies_r)


# In[ ]:


mask, x_offset, y_offset= create_mask_from_images(img3, [img1, img2, img4, img5], H_rel)


# In[ ]:


panorama= stitch_images_into_panorama(img3, [img1, img2, img4, img5], H_rel, mask, x_offset, y_offset, f'stitchafterRefining_{set}.png')


# In[ ]:


H12_r_own= refine_homography_lm_own(mp12_1_in, mp12_2_in, H12)
H23_r_own= refine_homography_lm_own(mp23_1_in, mp23_2_in, H23)
H34_r_own= refine_homography_lm_own(mp34_1_in, mp34_2_in, H34)
H45_r_own= refine_homography_lm_own(mp45_1_in, mp45_2_in, H45)


# In[ ]:


h_before_refining= [H12, H23, H34, H45]
h_after_refining= [H12_r, H23_r, H34_r, H45_r]
h_after_refining_own= [H12_r_own, H23_r_own, H34_r_own, H45_r_own]
matches= [(mp12_1_in, mp12_2_in), (mp23_1_in, mp23_2_in), (mp34_1_in, mp34_2_in), (mp45_1_in, mp45_2_in)]


# In[ ]:


errors_before_refining= []
errors_after_refining= []
errors_after_refining_own= []

for h1, h2, h3, match in zip(h_before_refining, h_after_refining, h_after_refining_own, matches):
    error_before_refining= np.linalg.norm(reprojection_error(h1.flatten()[:8], match[0], match[1]))
    error_after_refining= np.linalg.norm(reprojection_error(h2.flatten()[:8], match[0], match[1]))
    error_after_refining_own= np.linalg.norm(reprojection_error(h3.flatten()[:8], match[0], match[1]))
    errors_before_refining.append(error_before_refining)
    errors_after_refining.append(error_after_refining)
    errors_after_refining_own.append(error_after_refining_own)


# In[ ]:


errors_before_refining# Create a DataFrame from the three lists
df = pd.DataFrame({
    'Error Before Refining': errors_before_refining,
    'Error After Refining (Library)': errors_after_refining,
    'Error After Refining (Own)': errors_after_refining_own
})


# In[ ]:


df


# In[ ]:




