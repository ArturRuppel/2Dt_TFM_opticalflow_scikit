# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:14:04 2023

@author: aruppel and chatGTP
"""

import os
import time
import numpy as np
import tifffile
from scipy.ndimage import shift
import SimpleITK as sitk
from skimage.registration import phase_cross_correlation



def main(path_BK, path_AK, path_supp_stack, path_pattern, savepath, pixelsize, finterval):
    # Load the TIFF files
    supp_stack = tifffile.imread(path_supp_stack)[:, :, :, :]
    pattern = tifffile.imread(path_pattern)[:, :, :]
    ak = tifffile.imread(path_AK)
    bk = tifffile.imread(path_BK)[:, :, :, :]
    
    
    # # Average over the first dimension of supp_stack, AK.tif, and BK.tif
    # supp_stack_avg = np.mean(supp_stack, axis=1)
    # ak_avg = np.mean(ak[2:4, :, :], axis=0)
    # bk_avg = np.mean(bk[:, 3:5, :, :], axis=1)
    
    # Set up variables for saving registered images and displacement vectors
    reg_bk = np.zeros_like(bk)
    reg_supp_stack = np.zeros_like(supp_stack)
    reg_pattern = np.zeros_like(pattern)
    displacement_vectors = np.zeros((bk.shape[0], 3))
    
    # set parameters for translation correction
    ###########################################
    print("Starting translation correction")
    parameterMap = sitk.GetDefaultParameterMap('translation')   
    parameterMap['Metric'] = ["AdvancedMattesMutualInformation"]
    parameterMap['ResampleInterpolator'] = ["FinalLinearInterpolator"]
    parameterMap['WriteResultImage'] = ['false']
    parameterMap['NumberOfResolutions'] = ['2']
    parameterMap['MaximumNumberOfIterations'] = ['2000']
    
    for i in range(bk.shape[0]):    
    # for i in range(2):
        
        # set a timer to measure how long the analysis takes for each frame
        t0 = time.perf_counter()
        
        if i == 0:
            # Use the first frame as the reference for registration
            fixed_image = bk[i]
            moving_image = ak    
        else:
            moving_image = bk[i]
        
        # correct xy displacement first
        displacement_xy, _, _ = phase_cross_correlation(np.mean(fixed_image, axis=0), np.mean(moving_image, axis=0), upsample_factor=100)
        moving_image_intermediate = shift(moving_image, [0, displacement_xy[0], displacement_xy[1]])
        
        # convert input arrays into sitk image files
        moving_image_sitk = sitk.GetImageFromArray(moving_image_intermediate)
        fixed_image_sitk = sitk.GetImageFromArray(fixed_image)
        
        # initalize and perform rigid image registration
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed_image_sitk)
        elastixImageFilter.SetMovingImage(moving_image_sitk)
        elastixImageFilter.SetParameterMap(parameterMap)
        elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.Execute()
        
        # get resulting image
        moving_image_translated_sitk = elastixImageFilter.GetResultImage()
        
        # get deformation maps
        transformixImageFilter = sitk.TransformixImageFilter()
        transformParameterMapVector = elastixImageFilter.GetTransformParameterMap()
        transformixImageFilter.SetMovingImage(moving_image_translated_sitk)
        transformixImageFilter.SetTransformParameterMap(transformParameterMapVector)
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.LogToConsoleOff()
        transformixImageFilter.Execute()
        
        # convert sitk image files to numpy arrays   
        deformation_field = sitk.GetArrayFromImage(transformixImageFilter.GetDeformationField())
        moving_image_translated = sitk.GetArrayFromImage(moving_image_translated_sitk)
        
        displacement_vectors[i, :] = [-deformation_field[0, 0, 0, 2], displacement_xy[0], displacement_xy[1]]
        
        if i == 0:
            reg_bk[i] = fixed_image
            reg_ak = shift(ak, displacement_vectors[i, :])
            reg_supp_stack[i] = supp_stack[i]
            reg_pattern[i] = pattern[i]
        else:
            reg_bk[i] = shift(bk[i], displacement_vectors[i, :])
            # Apply the same displacement to supp_stack and pattern
            reg_supp_stack[i] = shift(supp_stack[i], displacement_vectors[i, :])
            reg_pattern[i] = shift(pattern[i], displacement_vectors[i, 1:3])
        
        t1 = time.perf_counter()
        print("Frame " + str(i) + ": Translation correction took " + str((t1-t0)/60) + " minutes")
    
    # Save the registered images and displacement vectors
    tifffile.imwrite(os.path.join(savepath, 'AK_registered.tif'), 
                     reg_ak[2:8, 50:1550, 50:1550].astype("uint16"),
                     imagej=True,
                     resolution=(1/pixelsize, 1/pixelsize),
                     metadata={
                         'unit': 'um',
                         'axes': 'ZYX'
                         }
                     )
    tifffile.imwrite(os.path.join(savepath, 'BK_registered.tif'), 
                     reg_bk[:, 2:8, 50:1550, 50:1550].astype("uint16"),
                     imagej=True,
                     resolution=(1/pixelsize, 1/pixelsize),
                     metadata={
                         'unit': 'um',
                         'finterval': finterval,
                         'axes': 'TZYX'
                         }
                     )
    tifffile.imwrite(os.path.join(savepath, 'GAP-GFP_registered.tif'), 
                     reg_supp_stack[:, 2:8, 50:1550, 50:1550].astype("uint16"),
                     imagej=True,
                     resolution=(1/pixelsize, 1/pixelsize),
                     metadata={
                         'unit': 'um',
                         'finterval': finterval,
                         'axes': 'TZYX'
                         }
                     )
    tifffile.imwrite(os.path.join(savepath, 'pattern_registered.tif'), 
                     reg_pattern[:, 50:1550, 50:1550].astype("uint16"),
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