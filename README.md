# **Panorama Stitching and Homography Optimization**

This repository implements panorama stitching and homography optimization using geometric principles and mathematical understanding. All the components such as homography estimation, RANSAC, and Levenberg-Marquardt (LM) optimization—are implemented from scratch to demonstrate the understanding of the concept.

---

## **Features**

1. **Feature Detection and Matching**:
   - Uses SIFT (Scale-Invariant Feature Transform) for feature detection and matching.
   - Relies on OpenCV for SIFT feature extraction but builds custom pipelines for processing matches.

2. **Homography Estimation**:
   - Implements Direct Linear Transformation (DLT) from scratch to compute homography matrices.
   - Robustly removes outliers using a custom implementation of the RANSAC algorithm.

3. **Homography Refinement**:
   - Refines homographies using both library-based and custom-built Levenberg-Marquardt (LM) optimization techniques

4. **Panorama Stitching**:
   - Computes relative homographies for multiple images using the derived transformations.
   - Implements geometric calculations for image warping and stitching, avoiding reliance on black-box library functions.

5. **Error Analysis**:
   - Evaluates reprojection error before and after refinement to illustrate the accuracy improvement achieved using geometric approaches.


### **Dependencies**
Ensure the following Python libraries are installed:
- `numpy`
- `opencv-python`
- `matplotlib`
- `scipy`
- `pandas`
