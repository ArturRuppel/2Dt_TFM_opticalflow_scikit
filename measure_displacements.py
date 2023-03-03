"""

@author: Artur Ruppel

"""

# import napari
# import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import time
import tifffile
import SimpleITK as sitk

from skimage import img_as_uint, img_as_float
from skimage import exposure
from skimage import io
from skimage import restoration
from skimage.filters import gaussian
from skimage.transform import downscale_local_mean



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

def displacement_measurement_2D(moving_image, fixed_image, supp_image, downsamplerate, savepath=False):      
    '''This function measures bead displacements in 3D+t movies for Traction Force Microscopy (TFM) applications. It performs translation correction on the 3D images,
    then averages that image over all z-slices and then performs non-rigid image registration on these averaged images. The translation correction is applied to an additional 3D image.
    
    Inputs: 
        moving_image: a 3D image containing the fluorescent markers. This image will be aligned to a reference 3D image
        fixed_image: a 3D reference image containing fluorscent markers. The moving image will be aligned to this image
        supp_image: a 3D image containing the cell. The translation correction of the rigid transformation will also be applied to this image
        savepath (optional): The parameters used for the image registrations will be printed to this path, if specified
        
    Outputs:
        moving_image_translated: 3D image containing the translation corrected image of the fluorescent beads
        supp_image_translated: 3D image containing the translation corrected cell image
        moving_image_morphed: 2D image containing the result of the non-rigid image registration. This image serves as quality control. If everything worked perfectly, it should look the same as the fixed_image.
        deformation_field: a 2D+d array containing the deformation field. The "d" dimension corresponds to the displacements in the different spatial directions: one for x, one for y and one for z.'''
    
        
    # convert input arrays into sitk image files
    moving_image_sitk = sitk.GetImageFromArray(moving_image)
    supp_image_sitk = sitk.GetImageFromArray(supp_image)
    fixed_image_sitk = sitk.GetImageFromArray(fixed_image)
    
    # set a timer to measure how long the analysis takes for each frame
    t0 = time.perf_counter()
    
    # set parameters for translation correction
    ###########################################
    print("Starting translation correction")
    parameterMap = sitk.GetDefaultParameterMap('translation')        
    parameterMap['Metric'] = ["AdvancedMattesMutualInformation"]
    parameterMap['ResampleInterpolator'] = ["FinalLinearInterpolator"]
    parameterMap['WriteResultImage'] = ['false']
    parameterMap['NumberOfResolutions'] = ['3']
    parameterMap['MaximumNumberOfIterations'] = ['2000']
    
    # parameterMap['AutomaticTransformInitialization'] = ["true"]
    # parameterMap['AutomaticTransformInitializationMethod'] = ["CenterOfMass"]
    # parameterMap['NumberOfHistogramBins'] = ["32"]         
    # parameterMap['Metric'] = ["AdvancedNormalizedCorrelation"]
    # parameterMap['MaximumNumberOfSamplingAttempts'] = ['4']
    # parameterMap['MaximumStepLength'] = ['0.1']
    # parameterMap['MovingImagePyramidSchedule'] = ['128 64 32 16 8 8 4 4 2 2 2 1 1 1 1']
    # parameterMap['ImagePyramidSchedule'] = ['16']


    # save translation parameters to file
    if savepath:
        sitk.WriteParameterFile(parameterMap, savepath + 'translation_parameters.txt')

    # initalize and perform rigid image registration
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image_sitk)
    elastixImageFilter.SetMovingImage(moving_image_sitk)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.Execute()
    
    # get resulting image
    moving_image_translated_sitk = elastixImageFilter.GetResultImage()
    
    # get resulting translation parameters and apply to supp stack
    transformParameterMapVector = elastixImageFilter.GetTransformParameterMap()
    supp_image_translated_sitk = sitk.Transformix(supp_image_sitk, transformParameterMapVector)        

    # stop timer to measure how long the analysis takes for each frame
    t1 = time.perf_counter()
    print("Translation correction took " + str((t1-t0)/60) + " minutes")

    # set parameters for non-rigid image registration
    #################################################
    # print("Starting 2D non-rigid registration")
    
    # # average over z-slices
    # moving_image_translated = sitk.GetArrayFromImage(moving_image_translated_sitk)
    # # moving_image_translated_avg_z = np.nanmean(moving_image_translated[2:5, :, :], axis=0)
    # # fixed_image_avg_z = np.nanmean(fixed_image[2:5, :, :], axis=0)
    # moving_image_translated_avg_z = moving_image_translated[2:5, :, :]
    # fixed_image_avg_z = fixed_image[2:5, :, :]
    
    # # convert to sitk format
    # moving_image_translated_avg_z_sitk = sitk.GetImageFromArray(moving_image_translated_avg_z)
    # fixed_image_avg_z_sitk = sitk.GetImageFromArray(fixed_image_avg_z)
    
    # # the parameter file from the previous registration step is reused, only parameters that change have to be defined here
    # parameterMap['Transform'] = ['BSplineTransform']
    # parameterMap['MaximumNumberOfIterations'] = ['2000']
    # parameterMap['FinalGridSpacingInVoxels'] = [str(downsamplerate)]

    # # save non-rigid regisrtation parameters to file
    # if savepath:
    #     sitk.WriteParameterFile(parameterMap, savepath + 'bspline_transform_parameters.txt')
        
    # # initalize and perform non-rigid image registration
    # elastixImageFilter = sitk.ElastixImageFilter()
    # elastixImageFilter.SetFixedImage(fixed_image_avg_z_sitk)
    # elastixImageFilter.SetMovingImage(moving_image_translated_avg_z_sitk)
    # elastixImageFilter.SetParameterMap(parameterMap)
    # elastixImageFilter.LogToConsoleOff()
    # elastixImageFilter.Execute()    
    
    # # get deformation maps
    # transformixImageFilter = sitk.TransformixImageFilter()
    # transformParameterMapVector = elastixImageFilter.GetTransformParameterMap()
    # transformixImageFilter.SetMovingImage(moving_image_translated_avg_z_sitk)
    # transformixImageFilter.SetTransformParameterMap(transformParameterMapVector)
    # transformixImageFilter.ComputeDeformationFieldOn()
    # transformixImageFilter.LogToConsoleOff()
    # transformixImageFilter.Execute()

    # # get resulting image
    # moving_image_morphed_sitk = elastixImageFilter.GetResultImage()   
    
    # # convert sitk image files to numpy arrays   
    # deformation_field = sitk.GetArrayFromImage(transformixImageFilter.GetDeformationField())
    # moving_image_translated = sitk.GetArrayFromImage(moving_image_translated_sitk)
    # supp_image_translated = sitk.GetArrayFromImage(supp_image_translated_sitk)    
    # moving_image_morphed = sitk.GetArrayFromImage(moving_image_morphed_sitk)
    
    # # set negative pixel values to 0 to prevent overflow artifacts
    # moving_image_translated[moving_image_translated<0] = 0
    # supp_image_translated[supp_image_translated<0] = 0
    # moving_image_morphed[moving_image_morphed<0] = 0
    
    # # convert to 16bit uint for saving
    # moving_image_translated = moving_image_translated.astype("uint64")
    # supp_image_translated = supp_image_translated.astype("uint64")
    # moving_image_morphed = moving_image_morphed.astype("uint64")
    
    # # stop timer to measure how long the analysis takes for each frame
    # t2 = time.perf_counter()
    
    # print("Non-rigid registration took " + str((t2-t1)/60) + " minutes")
    
    print("Starting 3D non-rigid registration")
    
    # the parameter file from the previous registration step is reused, only parameters that change have to be defined here
    parameterMap['Transform'] = ['BSplineTransform']
    # parameterMap['NumberOfResolutions'] = ['3']
    # parameterMap['ImagePyramidSchedule'] = ['16']
    parameterMap['MaximumNumberOfIterations'] = ['500']
    parameterMap['FinalGridSpacingInVoxels'] = ['8 8 8']

    # save non-rigid regisrtation parameters to file
    if savepath:
        sitk.WriteParameterFile(parameterMap, savepath + '/bspline_transform_parameters.txt')
        
    # initalize and perform non-rigid image registration
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image_sitk)
    elastixImageFilter.SetMovingImage(moving_image_translated_sitk)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.Execute()    
    
    # get deformation maps
    transformixImageFilter = sitk.TransformixImageFilter()
    transformParameterMapVector = elastixImageFilter.GetTransformParameterMap()
    transformixImageFilter.SetMovingImage(moving_image_translated_sitk)
    transformixImageFilter.SetTransformParameterMap(transformParameterMapVector)
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.Execute()

    # get resulting image
    moving_image_morphed_sitk = elastixImageFilter.GetResultImage()   
    
    # convert sitk image files to numpy arrays   
    deformation_field = sitk.GetArrayFromImage(transformixImageFilter.GetDeformationField())
    moving_image_translated = sitk.GetArrayFromImage(moving_image_translated_sitk)
    supp_image_translated = sitk.GetArrayFromImage(supp_image_translated_sitk)    
    moving_image_morphed = sitk.GetArrayFromImage(moving_image_morphed_sitk)
    
    # set negative pixel values to 0 to prevent overflow artifacts
    moving_image_translated[moving_image_translated<0] = 0
    supp_image_translated[supp_image_translated<0] = 0
    moving_image_morphed[moving_image_morphed<0] = 0
    
    # convert to 16bit uint for saving
    moving_image_translated = moving_image_translated.astype("uint64")
    supp_image_translated = supp_image_translated.astype("uint64")
    moving_image_morphed = moving_image_morphed.astype("uint64")
    
    # stop timer to measure how long the analysis takes for each frame
    t2 = time.perf_counter()

    return moving_image_translated, supp_image_translated, moving_image_morphed, deformation_field


def displacement_measurement_2Dt_main(moving_stack, fixed_3Dimage, supp_stack, savepath, radius, lower_threshold, sigma_smooth, downsamplerate):
    '''This function measures bead displacements in 3D+t movies for Traction Force Microscopy (TFM) applications. It first removes background and adjusts contrast of the input images and then uses rigid and non-rigid image registration methods from the elastix wrapper SITK.
    
    The input arguments are:
        moving_stack: a 3D+t movie containing the fluorescent markers. Every frame in this stack will be aligned to a reference 3D image
        fixed_3Dimage: a 3D reference image containing fluorscent markers. Every frame in the moving stack will be aligned to this image
        supp_stack: a 3D+t movie containing images of the cell. The translation correction of the rigid transformation will also be applied to this stack
        savepath: The path to which outputfiles are going to be saved
        radius: The radius of the rolling ball for background substraction
        lower_threshold: The percentile value below which all pixel values will be set to 0 after background substraction. 
        sigma_smooth: The size of the smoothening window applied after background substraction and contrast adjustment.
        
    This function first removes the background of the bead images using a rolling ball algorithm. It then thresholds the image by setting background pixels to 0 and normalizing the intensity values to the full dynamic range of the image format
    Next, every frame in the moving stack is aligned to the reference image by using translation image registration. The same translation is applied to the frames in the supp_stack.
    Then, all z-slices are averaged.
    Finally, the displacement field is measured with a BsplineTransform method. The parameters used are printed into a text file which is saved into the savepath
    
    Outputs:
        moving_image_translated: 2D+t movie containing a concatenation of the reference image and the translation corrected moving stack 
        supp_image_translated: 2D+t movie containing the translation corrected the cell images
        moving_image_morphed: 2D+t movie containing the result of the non-rigid image registration. This movie serves as quality control. If everything worked perfectly, all frames should look the same.
        deformation_field: a 2D+t+d array containing the deformation field. The "d" dimension corresponds to the displacements in the different spatial directions: one for x, one for y and one for z.
        '''

    moving_stack_processed = np.zeros(moving_stack.shape).astype('uint16')
    moving_stack_morphed = np.zeros(moving_stack.shape).astype('uint16')
    moving_stack_morphed_avg_z = np.nanmean(moving_stack_morphed , axis=1).astype('uint16') # a little weird way of getting rid of the z-dimension, but this allows the code to be extended for 3D application if necessary
    fixed_3Dimage_processed = np.zeros(fixed_3Dimage.shape).astype('uint16')
    supp_stack_processed = np.zeros(supp_stack.shape).astype('uint16')
    
    deformation_field_x = np.zeros(moving_stack_morphed_avg_z.shape)
    deformation_field_y = np.zeros(moving_stack_morphed_avg_z.shape)
    

    print("Removing background, stretching contrast for AK")
    
    p2 = np.percentile(fixed_3Dimage[2, :, :], lower_threshold)
    p98 = np.percentile(fixed_3Dimage[2, :, :], 99.9)
    
    for current_z in np.arange(moving_stack.shape[1]):
        
        fixed_image = fixed_3Dimage[current_z, :, :]
        fixed_image_1 = remove_background(fixed_image, radius)
        
        fixed_image_2 = stretch_contrast(fixed_image_1, p2, p98)
        fixed_image_processed = img_as_uint(gaussian(fixed_image_2, sigma=sigma_smooth))
        
        fixed_3Dimage_processed[current_z, :, :] = fixed_image_processed
        

    for frame in np.arange(moving_stack.shape[0]):
    # for frame in np.arange(2):
        # frame = 49
        print("Removing background, stretching contrast, frame " + str(frame))
        for current_z in np.arange(moving_stack.shape[1]):            
            moving_image_1 = remove_background(moving_stack[frame, current_z, :, :], radius)
            

            moving_image_2 = stretch_contrast(moving_image_1, p2, p98)
            
            moving_image_processed = img_as_uint(gaussian(moving_image_2, sigma=sigma_smooth))   
            # supp_image_processed = stretch_contrast(supp_stack[frame, current_z, :, :], p2_supp, p98_supp)
            supp_image_processed = supp_stack[frame, current_z, :, :]
            
            moving_stack_processed[frame, current_z, :, :] = moving_image_processed
            supp_stack_processed[frame, current_z, :, :] = supp_image_processed
        
        moving_3Dimage_processed = moving_stack_processed[frame, :, :, :]
        supp_3Dimage_processed = supp_stack_processed[frame, :, :, :]
        
        if frame == 0:  # save registration parameters to file if the first frame is being analyzed
            moving_3Dimage_translated, supp_3Dimage_translated, moving_image_morphed, deformation_field = \
                                                    displacement_measurement_2D(moving_3Dimage_processed, fixed_3Dimage_processed, supp_3Dimage_processed, downsamplerate, savepath=savepath)
        else:
            moving_3Dimage_translated, supp_3Dimage_translated, moving_image_morphed, deformation_field = \
                                                    displacement_measurement_2D(moving_3Dimage_processed, fixed_3Dimage_processed, supp_3Dimage_processed, downsamplerate)                                            
        
        
        moving_stack_processed[frame, :, :, :] = moving_3Dimage_translated
        supp_stack_processed[frame, :, :, :] = supp_3Dimage_translated
        
        # moving_stack_morphed_avg_z[frame, :, :] = moving_image_morphed
        moving_stack_morphed[frame, :, :, :] = moving_image_morphed
        
        deformation_field_x[frame, :, :] = deformation_field[2, :, :, 0]
        deformation_field_y[frame, :, :] = deformation_field[2, :, :, 1]        
        

    return moving_stack_processed, moving_stack_morphed, fixed_3Dimage_processed, supp_stack_processed, deformation_field_x, deformation_field_y



def main(path_BK, path_AK, path_supp_stack, savepath, finterval, pixelsize, supp_stack_file_name, radius=20, lower_threshold=95, sigma_smooth=1, downsamplerate=8):

    print("Loading BK")
    BK = io.imread(path_BK)[:, :, :, :]  # / (2 ** 16)
    print("Loading supp stack")
    supp_stack = io.imread(path_supp_stack)[:, :, :, :]  # / (2 ** 16)
    print("Loading AK")
    AK = io.imread(path_AK)[:, :, :] # / (2 ** 16)
    
    BK, BK_morphed, AK, supp_stack, d_x, d_y = displacement_measurement_2Dt_main(BK, AK, supp_stack, savepath, radius, lower_threshold, sigma_smooth, downsamplerate)

    np.save(savepath + "d_x.npy", downscale_local_mean(d_x, (1, 4, 4)))
    np.save(savepath + "d_y.npy", downscale_local_mean(d_y, (1, 4, 4)))
    # np.save(savepath + "d_z.npy", deformation_field_z)

    AK_and_BK = np.concatenate((np.expand_dims(AK, 0), BK), axis=0)
    
    AK_and_BK_avg_z = np.nanmean(AK_and_BK, axis=1).astype('uint16')
    supp_stack_avg_z = np.nanmean(supp_stack, axis=1).astype('uint16')

    tifffile.imwrite(savepath + "/AK_and_BK_processed.tif", 
                      AK_and_BK,
                      imagej=True,
                      resolution=(1/pixelsize, 1/pixelsize),
                      metadata={
                          'unit': 'um',
                          'finterval': finterval,
                          'axes': 'TZYX'
                          }
                      )
    
    tifffile.imwrite(savepath + "/BK_morphed.tif", 
                      BK_morphed,
                      imagej=True,
                      resolution=(1/pixelsize, 1/pixelsize),
                      metadata={
                          'unit': 'um',
                          'finterval': finterval,
                          'axes': 'TZYX'
                          }
                      )
      
    tifffile.imwrite(savepath + "/" + supp_stack_file_name + "_processed.tif", 
                      supp_stack,
                      imagej=True,
                      resolution=(1/pixelsize, 1/pixelsize),
                      metadata={
                          'unit': 'um',
                          'finterval': finterval,
                          'axes': 'TZYX'
                          }
                      )
    
    tifffile.imwrite(savepath + "/" + supp_stack_file_name + "_avg_z.tif", 
                      supp_stack_avg_z,
                      imagej=True,
                      resolution=(1/pixelsize, 1/pixelsize),
                      metadata={
                          'unit': 'um',
                          'finterval': finterval,
                          'axes': 'TYX'
                          }
                      )
    
    tifffile.imwrite(savepath + "/AK_and_BK_avg_z.tif", 
                      AK_and_BK_avg_z,
                      imagej=True,
                      resolution=(1/pixelsize, 1/pixelsize),
                      metadata={
                          'unit': 'um',
                          'finterval': finterval,
                          'axes': 'TYX'
                          }
                      )


   


# %% dragonfly
if __name__ == "__main__":
    no_stacks = 7               # number of stacks to be analyzed
    finterval = 150             # interval between frames in s
    # pixelsize = 0.1           # size of a pixel in the bead image in µm
    pixelsize = 0.217           # size of a pixel in the bead image in µm
    # pixelsize = 0.325           # size of a pixel in the bead image in µm
    cropwidth = 0               # cropping out a centered window of this size. Set to 0 if no cropping desired
    radius = 20                 # radius of rolling ball for background substraction
    lower_threshold = 50        # lower threshold percentile for bead stack
    sigma_smooth = 1            # radius of gaussian smoothing window


    # path = "D:/2022-10-21 dragonfly induced mesoderm GAP-GFP TFM 3kPa/position"
    path = "D:/2023-01-13 olympus SR induced mesoderm GAP-GFP TFM 6kPa/position"

    # for position in np.arange(no_stacks):
    for position in [2]:

        # position = 1
        path_BK = path + str(position) + "/BK.tif"
        path_AK = path + str(position) + "/AK.tif"
        path_supp_stack = path + str(position) + "/GAP-GFP.tif"""
        savepath = path + str(position) + "/TFM_data/"

        if not os.path.exists(savepath):
            os.mkdir(savepath)
        main(path_BK, path_AK, path_supp_stack, savepath, finterval, pixelsize, radius=radius, lower_threshold=lower_threshold, sigma_smooth=sigma_smooth, downsamplerate=8)

# %% opto microscope
       
# if __name__ == "__main__":
#     no_stacks = 10  # number of stacks to be analyzed
#     no_frames = 3  # number of frames in a stack to be analyzed
#     pixelsize = 0.275 * 1e-6  # size of a pixel in the bead image in m
#     radius = 6
#     th = 80
#     sigma_smooth = 1
#     dmax = 0.5  # max displacement for maps in micron
#     pmax = 0.1  # max traction for maps in kPa
#     sigma_max = 5   # max stress for maps in mN/m
#     downsamplerate = 4  # final forcemaps will have a resolution of the image divided by this number
#     forcemap_pixelsize = pixelsize * downsamplerate  # size of a pixel in the force maps
#     E = 3000  # rigidity of the hydrogel in Pa
#     nu = 0.5  # poisson ratio of the hydrogel
#     alpha = 5 * 1e-18  # regularization parameter for force calculation.
#     cropwidth = 0     # cropping out a centered window of this size. Set to 0 if no cropping desired
    


#     path = "D:/2022-09-16 - NHS-AA 3kPa migration/position"

#     for position in np.arange(no_stacks):
#         position = 1
#         path_BK = path + str(position) + "/BK.tif"
#         path_AK = path + str(position) + "/AK.tif"
#         path_BF = path + str(position) + "/BF.tif"
#         savepath = path + str(position) + "/"

#         if not os.path.exists(savepath):
#             os.mkdir(savepath)
#         main(path_BK, path_AK, savepath, no_frames, pixelsize, radius=radius, th=th, sigma_smooth=sigma_smooth, dmax=dmax, pmax=pmax, sigma_max=sigma_max, downsamplerate=downsamplerate,
#              E=E, nu=nu, alpha=alpha, cropwidth=cropwidth)
