"""

@author: Artur Ruppel

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2
from skimage import exposure
from skimage import img_as_float, img_as_uint, img_as_ubyte
from skimage import io
from skimage import restoration
from skimage.filters import gaussian
from skimage.morphology import reconstruction
from skimage.registration import optical_flow_tvl1, phase_cross_correlation, optical_flow_ilk
from skimage.transform import EuclideanTransform
from skimage.transform import warp, downscale_local_mean
from skimage.feature import blob_dog, blob_log, blob_doh

from scipy.interpolate import interp2d
from scipy import optimize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib as mpl
from skimage.morphology import dilation, closing, disk, erosion
# from MSM_functions import prepare_forces, grid_setup, FEM_simulation
from plot_movies_functions import *

DPI = 300




def main(path_supp_stack, supp_stack_file_name, savepath, forcemap_pixelsize, dmax=2, pmax=2):
    
    forcemap_pixelsize *= 1e-6
    print("Loading image stack")
    supp_stack = io.imread(path_supp_stack)[:, :, :]  # / (2 ** 16)
    # load data
    d_x = np.load(savepath + "d_x.npy") * forcemap_pixelsize * 1e6
    d_y = np.load(savepath + "d_y.npy") * forcemap_pixelsize * 1e6
    t_x = np.load(savepath + "t_x.npy")    
    t_y = np.load(savepath + "t_y.npy")  
    no_frames = d_x.shape[0]

    print("Making plots")
    # make plots of displacement fields
    for frame in np.arange(no_frames):
        make_displacement_plots(d_x[frame, :, :], d_y[frame, :, :], dmax, savepath + "displacement" + str(frame) + ".png", forcemap_pixelsize, frame=frame)
        make_traction_plots(t_x[frame, :, :], t_y[frame, :, :], pmax, savepath + "traction" + str(frame), forcemap_pixelsize, frame=frame)
        plot_image_with_forces(2*img_as_float(supp_stack[frame, :, :]), t_x[frame, :, :], t_y[frame, :, :], pmax, savepath + supp_stack_file_name + "_forces" + str(frame), forcemap_pixelsize, frame=frame)


    make_movies_from_images(savepath, "displacement", no_frames)
    make_movies_from_images(savepath, "traction", no_frames)
    make_movies_from_images(savepath, supp_stack_file_name + "_forces", no_frames)
    # make_movies_from_images(savepath, "xx-Stress", no_frames)
    # make_movies_from_images(savepath, "yy-Stress", no_frames)
    # make_movies_from_images(savepath, "avg-Stress", no_frames)

    # strain_energy_density = 0.5 * (t_x * d_x + t_y * d_y) * forcemap_pixelsize ** 2
    # strain_energy = np.nansum(strain_energy_density, axis=(0, 1))

    # center = int(strain_energy_density.shape[0] / 2)
    # strain_energy_left = np.nansum(strain_energy_density[:, 0:center, :], axis=(0, 1))
    # strain_energy_right = np.nansum(strain_energy_density[:, center:-1, :], axis=(0, 1))

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))  # create figure and axes
    # ax.plot(strain_energy[0:frame + 1] * 1e15)
    # plt.title("Strain energy [fJ]")
    # plt.xlim(0, no_frames)
    # plt.ylim(0, 20)
    # plt.xlabel("time [min]")
    # plt.savefig(savepath + "strain_energy.png", dpi=DPI, bbox_inches="tight")



# %% dragonfly
if __name__ == "__main__":
    no_stacks = 7  # number of stacks to be analyzed
    pixelsize = 0.217 * 1e-6  # size of a pixel in the bead image in m
    # pixelsize = 0.325 * 1e-6  # size of a pixel in the bead image in m
    downsamplerate = 1  # final forcemaps will have a resolution of the image divided by this number
    dmax = 3  # max displacement for maps in micron
    pmax = 1  # max traction for maps in kPa


    path = "D:/2022-11-02 olympus SR induced mesoderm GAP-GFP TFM 3kPa/sample1/position"
    # path = "D:/demo2/position"


    for position in np.arange(no_stacks):
    # position = 1
        path_supp_stack = path + str(position) + "/TFM_data/GAP-GFP_avg_z.tif"
        savepath = path + str(position) + "/TFM_data/"

        if not os.path.exists(savepath):
            os.mkdir(savepath)
        main(path_supp_stack, savepath, downsamplerate*pixelsize, dmax=dmax, pmax=pmax)

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
