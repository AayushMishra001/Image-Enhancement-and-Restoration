# Image Enhancement and Restoration using Python & OpenCV

## Overview

This project implements an image enhancement and restoration pipeline for electronic circuit layout images using Python and OpenCV. The objective is to reduce image degradation while preserving fine structural edges that are important for automated inspection and analysis.

## Features

* Contrast enhancement using CLAHE
* Gaussian blur and Gaussian noise simulation
* Non-Local Means (NLM) denoising
* Bilateral filtering
* Edge detection using:

  * Canny
  * Sobel
  * Laplacian
* Selective edge restoration using Unsharp Masking
* Image quality evaluation using PSNR and SSIM

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-image
* Matplotlib

## Applications

* Digital Image Processing
* Automated Optical Inspection (AOI)
* Electronic Circuit Image Analysis
* Image Restoration

## Repository Contents

* `image_restoration_simple.py` – Main implementation
* `results/` – Sample outputs and processed images
* Input images used for evaluation

## Future Improvements

* Support for additional restoration algorithms
* Deep learning-based denoising methods
* Batch image processing
* Interactive GUI interface
