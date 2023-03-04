# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:14:04 2023

@author: aruppel and chatGTP
"""

import os
import time
import numpy as np
import tifffile
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift



def main(path_BK, path_AK, path_supp_stack, path_pattern, savepath, pixelsize, finterval):
    # Load the TIFF files
    supp_stack = tifffile.imread(path_supp_stack)[0:2, :, :, :]
    pattern = tifffile.imread(path_pattern)[0:2, :, :]
    ak = tifffile.imread(path_AK)
    bk = tifffile.imread(path_BK)[0:2, :, :, :]
    
    
    # Average over the first dimension of supp_stack, AK.tif, and BK.tif
    supp_stack_avg = np.mean(supp_stack, axis=1)
    ak_avg = np.mean(ak[2:4, :, :], axis=0)
    bk_avg = np.mean(bk[:, 3:5, :, :], axis=1)
    
    # Set up variables for saving registered images and displacement vectors
    reg_bk = np.zeros_like(bk_avg)
    reg_supp_stack = np.zeros_like(supp_stack_avg)
    reg_pattern = np.zeros_like(pattern)
    displacement_vectors = np.zeros((bk.shape[0], 2))
    
    # Loop over the first dimension of BK.tif and perform image registration for each image
    print("Starting translation correction")
    for i in range(bk.shape[0]):    
    # for i in range(2):
        
        # set a timer to measure how long the analysis takes for each frame
        t0 = time.perf_counter()
        
        if i == 0:
            # Use the first frame as the reference for registration
            reference = bk_avg[i]
            displacement, _, _ = phase_cross_correlation(reference, ak_avg, upsample_factor=10)
            reg_ak = shift(ak_avg, displacement)
            displacement = 0
        else:
            # Register the current frame to the reference frame using phase cross correlation
            displacement, _, _ = phase_cross_correlation(reference, bk_avg[i], upsample_factor=10)
            displacement_vectors[i] = displacement
            
        # Shift the image and save the registered image
        reg_bk[i] = shift(bk_avg[i], displacement)

        # Apply the same displacement to supp_stack and pattern
        reg_supp_stack[i] = shift(supp_stack_avg[i], displacement)
        reg_pattern[i] = shift(pattern[i], displacement)
        
        t1 = time.perf_counter()
        print("Frame " + str(i) + ": Translation correction took " + str((t1-t0)/60) + " minutes")
    
    # Save the registered images and displacement vectors
    tifffile.imwrite(os.path.join(savepath, 'AK_registered.tif'), 
                     np.mean(reg_ak, axis=0).astype("uint16"),
                     imagej=True,
                     resolution=(1/pixelsize, 1/pixelsize),
                     metadata={
                         'unit': 'um',
                         'axes': 'YX'
                         }
                     )
    tifffile.imwrite(os.path.join(savepath, 'BK_registered.tif'), 
                     np.mean(reg_bk, axis=1).astype("uint16"),
                     imagej=True,
                     resolution=(1/pixelsize, 1/pixelsize),
                     metadata={
                         'unit': 'um',
                         'finterval': finterval,
                         'axes': 'TYX'
                         }
                     )
    tifffile.imwrite(os.path.join(savepath, 'GAP-GFP_registered.tif'), 
                     np.mean(reg_supp_stack, axis=1).astype("uint16"),
                     imagej=True,
                     resolution=(1/pixelsize, 1/pixelsize),
                     metadata={
                         'unit': 'um',
                         'finterval': finterval,
                         'axes': 'TYX'
                         }
                     )
    tifffile.imwrite(os.path.join(savepath, 'pattern_registered.tif'), 
                     reg_pattern.astype("uint16"),
                     imagej=True,
                     resolution=(1/pixelsize, 1/pixelsize),
                     metadata={
                         'unit': 'um',
                         'finterval': finterval,
                         'axes': 'TYX'
                         }
                     )
    

    np.savetxt(os.path.join(savepath, 'displacement_vectors.txt'), displacement_vectors)

if __name__ == "__main__":
    folder = "D:/2023-02-23/position0"
    path_supp_stack = os.path.join(folder, 'GAP-GFP.tif')
    path_pattern = os.path.join(folder, 'pattern.tif')
    path_AK = os.path.join(folder, 'AK.tif')
    path_BK = os.path.join(folder, 'BK.tif')    
    savepath = folder + '/TFM_data/'

    if not os.path.exists(savepath):
        os.mkdir(savepath)
    pixelsize = 0.1
    finterval = 90
    main(path_BK, path_AK, path_supp_stack, path_pattern, savepath, pixelsize, finterval)