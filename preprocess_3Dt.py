# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:14:04 2023

@author: aruppel
"""

import os
import time
import numpy as np
import tifffile
from scipy.ndimage import shift
import SimpleITK as sitk
from skimage import exposure
from skimage import restoration
from skimage.registration import phase_cross_correlation
from skimage import img_as_uint, img_as_float
from skimage.filters import gaussian


def remove_background(image, radius=100):
    background = restoration.rolling_ball(image, radius=radius)
    return image - background


def stretch_contrast(image, p2, p98):
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale

def stretch_contrast_relative(image, lower_threshold=0.001, upper_threshold=99.99):
    p2 = np.percentile(image, lower_threshold)
    p98 = np.percentile(image, upper_threshold)
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale

def main(path_BK, path_AK, path_supp_stack, path_pattern, savepath, supp_stack_file_name, pixelsize, finterval, radius=20, lower_threshold=95, sigma_smooth=1):
    # Load the TIFF files
    supp_stack = tifffile.imread(path_supp_stack)[0:5, :, :, :]
    pattern = tifffile.imread(path_pattern)[0:5, :, :]
    ak = tifffile.imread(path_AK)[:, :, :]
    bk = tifffile.imread(path_BK)[0:5, :, :, :]

    # Set up variables for saving preprocessed
    # supp_stack_processed = np.zeros_like(supp_stack)
    # pattern_processed = np.zeros_like(pattern)
    ak_processed = np.zeros_like(ak)
    bk_processed = np.zeros_like(bk)

    # Set up variables for saving registered images and displacement vectors
    ak_registered = np.zeros_like(ak)
    bk_registered = np.zeros_like(bk)
    supp_stack_registered = np.zeros_like(supp_stack)
    pattern_registered = np.zeros_like(pattern)
    displacement_vectors = np.zeros((bk.shape[0], 3))

    print("Removing background, stretching contrast for AK")

    p2 = np.percentile(ak[:, :, :], lower_threshold)
    p98 = np.percentile(ak[:, :, :], 99.9)

    for current_z in np.arange(ak.shape[0]):
        ak_current = ak[current_z, :, :]
        ak_processed_1 = remove_background(ak_current, radius)
        ak_processed_2 = stretch_contrast(ak_processed_1, p2, p98)
        ak_processed_3 = img_as_uint(gaussian(ak_processed_2, sigma=sigma_smooth))
        ak_processed[current_z, :, :] = ak_processed_3

    for frame in np.arange(bk.shape[0]):
        print("Removing background, stretching contrast for BK, frame " + str(frame))
        for current_z in np.arange(bk.shape[1]):
            bk_current = bk[frame, current_z, :, :]
            bk_processed_1 = remove_background(bk_current, radius)
            bk_processed_2 = stretch_contrast(bk_processed_1, p2, p98)
            bk_processed_3 = img_as_uint(gaussian(bk_processed_2, sigma=sigma_smooth))
            bk_processed[frame, current_z, :, :] = bk_processed_3

    # set parameters for translation correction
    ###########################################
    print("Starting translation correction")
    parameterMap = sitk.GetDefaultParameterMap('translation')   
    parameterMap['Metric'] = ["AdvancedMattesMutualInformation"]
    parameterMap['ResampleInterpolator'] = ["FinalLinearInterpolator"]
    parameterMap['WriteResultImage'] = ['false']
    parameterMap['NumberOfResolutions'] = ['2']
    parameterMap['MaximumNumberOfIterations'] = ['2000']
    
    for frame in range(bk.shape[0]):
    # for frame in range(2):
        
        # set a timer to measure how long the analysis takes for each frame
        t0 = time.perf_counter()
        
        if frame == 0:
            # Use the first frame as the reference for registration
            fixed_image = bk_processed[frame]
            moving_image = ak_processed
        else:
            moving_image = bk_processed[frame]
        
        # correct xy displacement with phase_cross_correlation function from sci-kit, because it works better for large displacements than SimpleElastix
        mask = np.mean(moving_image, axis=0) < -1
        center_x = int(mask.shape[0] / 2)
        center_y = int(mask.shape[1] / 2)
        half_width = 500
        mask[center_x-half_width:center_x+half_width, center_y-half_width:center_y+half_width] = True
        
        displacement_xy, _, _ = phase_cross_correlation(np.mean(fixed_image, axis=0)*mask, np.mean(moving_image, axis=0)*mask, upsample_factor=100)
        # moving_image_intermediate = shift(moving_image, [0, displacement_xy[0], displacement_xy[1]])
        #
        # # convert input arrays into sitk image files
        # moving_image_sitk = sitk.GetImageFromArray(moving_image_intermediate)
        # fixed_image_sitk = sitk.GetImageFromArray(fixed_image)
        #
        # # initalize and perform rigid image registration
        # elastixImageFilter = sitk.ElastixImageFilter()
        # elastixImageFilter.SetFixedImage(fixed_image_sitk)
        # elastixImageFilter.SetMovingImage(moving_image_sitk)
        # elastixImageFilter.SetParameterMap(parameterMap)
        # elastixImageFilter.LogToConsoleOff()
        # elastixImageFilter.Execute()
        #
        # # get resulting image
        # moving_image_translated_sitk = elastixImageFilter.GetResultImage()
        #
        # # get deformation maps
        # transformixImageFilter = sitk.TransformixImageFilter()
        # transformParameterMapVector = elastixImageFilter.GetTransformParameterMap()
        # transformixImageFilter.SetMovingImage(moving_image_translated_sitk)
        # transformixImageFilter.SetTransformParameterMap(transformParameterMapVector)
        # transformixImageFilter.ComputeDeformationFieldOn()
        # transformixImageFilter.LogToConsoleOff()
        # transformixImageFilter.Execute()
        #
        # # convert sitk image files to numpy arrays
        # deformation_field = sitk.GetArrayFromImage(transformixImageFilter.GetDeformationField())
        # # moving_image_translated = sitk.GetArrayFromImage(moving_image_translated_sitk)
        #
        # displacement_vectors[frame, :] = [-deformation_field[0, 0, 0, 2], displacement_xy[0], displacement_xy[1]]

        displacement_vectors[frame, :] = [0, displacement_xy[0], displacement_xy[1]]

        if frame == 0:
            bk_registered[frame] = fixed_image
            ak_registered = shift(ak_processed, displacement_vectors[frame, :])
            supp_stack_registered[frame] = supp_stack[frame]
            pattern_registered[frame] = pattern[frame]
        else:
            bk_registered[frame] = shift(bk_processed[frame], displacement_vectors[frame, :])
            # Apply the same displacement to supp_stack and pattern
            supp_stack_registered[frame] = shift(supp_stack[frame], displacement_vectors[frame, :])
            pattern_registered[frame] = shift(pattern[frame], displacement_vectors[frame, 1:3])
        
        t1 = time.perf_counter()
        print("Frame " + str(frame) + ": Translation correction took " + str((t1 - t0) / 60) + " minutes")

    # Average over the z-dimension dimension of supp_stack, AK.tif, and BK.tif
    supp_stack_registered_avg_z = np.mean(supp_stack_registered, axis=1)
    ak_registered_avg_z = np.mean(ak_registered[2:-2, :, :], axis=0)
    bk_registered_avg_z = np.mean(bk_registered[:, 2:-2, :, :], axis=1)

    # Save the registered images and displacement vectors
    tifffile.imwrite(os.path.join(savepath, 'AK_registered_avg_z.tif'),
                     ak_registered_avg_z.astype("uint16"),
                     imagej=True,
                     resolution=(1/pixelsize, 1/pixelsize),
                     metadata={
                         'unit': 'um',
                         'axes': 'YX'
                         }
                     )
    tifffile.imwrite(os.path.join(savepath, 'BK_registered_avg_z.tif'),
                     bk_registered_avg_z.astype("uint16"),
                     imagej=True,
                     resolution=(1/pixelsize, 1/pixelsize),
                     metadata={
                         'unit': 'um',
                         'finterval': finterval,
                         'axes': 'TYX'
                         }
                     )
    tifffile.imwrite(os.path.join(savepath, supp_stack_file_name + "_registered_avg_z.tif"),
                     supp_stack_registered_avg_z.astype("uint16"),
                     imagej=True,
                     resolution=(1/pixelsize, 1/pixelsize),
                     metadata={
                         'unit': 'um',
                         'finterval': finterval,
                         'axes': 'TYX'
                         }
                     )
    tifffile.imwrite(os.path.join(savepath, 'pattern_registered.tif'),
                     pattern_registered.astype("uint16"),
                     imagej=True,
                     resolution=(1/pixelsize, 1/pixelsize),
                     metadata={
                         'unit': 'um',
                         'finterval': finterval,
                         'axes': 'TYX'
                         }
                     )
    

    np.savetxt(os.path.join(savepath, 'displacement_vectors.txt'), displacement_vectors)
