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
from skimage import io
from skimage.registration import optical_flow_tvl1

from skimage.transform import downscale_local_mean, warp


def displacement_measurement_2D(moving_image, fixed_image):
    nr, nc = fixed_image.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    # d_y, d_x = optical_flow_ilk(moving_image, fixed_image, radius=40, gaussian=True)
    d_y, d_x = optical_flow_tvl1(fixed_image, moving_image, attachment=10, prefilter=True)
    # d_y, d_x = optical_flow_tvl1(fixed_image, moving_image, tol=0.000000001, num_warp=8, num_iter=150, attachment=50, prefilter=True)
    # d_y, d_x = optical_flow_ilk(moving_image_translated, fixed_image, radius=10, gaussian=True, num_warp=20)

    moving_image_deformed = warp(moving_image, np.array([row_coords + d_y, col_coords + d_x]), mode="constant")

    return d_x, d_y, moving_image_deformed


def displacement_measurement_2Dt_main(moving_stack, fixed_image, downsamplerate):
    """
    This function measures bead displacements in 2D+t movies for Traction Force Microscopy (TFM) applications by using the optical flow algorithm implemented in sci-kit.

    The input arguments are:
        moving_stack: a 2D+t movie containing the fluorescent markers. Displacements in every frame in this stack will be measured with reference to a reference 2D image
        fixed_image: a 2D reference image containing fluorscent markers. Displacements in every frame in this stack will be measured with reference to this image
        downsamplerate: The resolution of the final deformation field will be reduced by this factor through local averaging of pixels

    Outputs:
        moving_image_morphed: 2D+t movie containing the image after applying the deformation obtained from the optical flow algorithm. This movie serves as quality control. If everything worked perfectly, all frames should like the reference image.
        d_x: The x-compoment of the deformation field
        d_y: The y-compoment of the deformation field
    """

    moving_stack_morphed = np.zeros_like(moving_stack).astype('uint16')

    d_x = np.zeros_like(moving_stack).astype('float32')
    d_y = np.zeros_like(moving_stack).astype('float32')

    print("Starting displacement measurement")

    for frame in np.arange(moving_stack.shape[0]):
        t0 = time.perf_counter()
        d_x_current, d_y_current, moving_image_morphed = displacement_measurement_2D(moving_stack[frame, :, :], fixed_image)

        moving_stack_morphed[frame, :, :] = moving_image_morphed

        d_x[frame, :, :] = d_x_current
        d_y[frame, :, :] = d_y_current

        # import napari
        # napari.view_image(d_x[frame, :, :])
        # napari.view_image(np.stack((moving_stack[frame, :, :], moving_image_morphed * 2 ** 16, fixed_image)))

        t1 = time.perf_counter()
        print("Frame " + str(frame) + ": Displacement measurement took " + str((t1-t0)/60) + " minutes")

    d_x = downscale_local_mean(d_x, (1, downsamplerate, downsamplerate))
    d_y = downscale_local_mean(d_y, (1, downsamplerate, downsamplerate))

    return moving_stack_morphed, d_x, d_y



def main(path_BK, path_AK, savepath, finterval, pixelsize, downsamplerate=8):
    print("Loading BK")
    bk = io.imread(path_BK)[:, :, :]  # / (2 ** 16)
    print("Loading AK")
    ak = io.imread(path_AK)[:, :]  # / (2 ** 16)

    bk_morphed, d_x, d_y = displacement_measurement_2Dt_main(bk, ak, downsamplerate)

    np.save(savepath + "d_x.npy", d_x)
    np.save(savepath + "d_y.npy", d_y)

    tifffile.imwrite(savepath + "/BK_morphed.tif",
                     bk_morphed,
                     imagej=True,
                     resolution=(1 / pixelsize, 1 / pixelsize),
                     metadata={
                         'unit': 'um',
                         'finterval': finterval,
                         'axes': 'TYX'
                     }
                     )
