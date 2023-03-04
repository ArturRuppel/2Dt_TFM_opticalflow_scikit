# -*- coding: utf-8 -*-
import os
import numpy as np
from preprocess_3Dt import main as preprocess_3Dt
from preprocess_2Dt import main as preprocess_2Dt
from measure_displacements import main as measure_displacements
from calculate_forces import main as calculate_forces
# from apply_MSM import main as apply_MSM
from make_TFM_movies import main as make_TFM_movies

# disp
no_stacks = 6               # number of stacks to be analyzed
finterval = 90              # interval between frames in s
pixelsize = 0.1           # size of a pixel in the bead image in µm
# pixelsize = 0.217           # size of a pixel in the bead image in µm
# pixelsize = 0.325           # size of a pixel in the bead image in µm
downsamplerate = 4          # final forcemaps will have a resolution of the image divided by this number
radius = 20                 # radius of rolling ball for background substraction
lower_threshold = 70        # lower threshold percentile for bead stack
sigma_smooth = 1            # radius of gaussian smoothing window
dmax = 0.4                    # max displacement for maps in micron
pmax = 1                    # max traction for maps in kPa
lambda_max = 600            # max line tension in N/m
E = 6000                   # rigidity of the hydrogel in Pa
nu = 0.5                    # poisson ratio of the hydrogel
alpha = 1 * 1e-19           # regularization parameter for force calculation


# path = "D:/2022-10-21 dragonfly induced mesoderm GAP-GFP TFM 3kPa/position"
path = "D:/2023-02-23/position"
supp_stack_file_name = "GAP-GFP"

# Using the repr() function to convert the string values to their string representation
s_finterval = repr(finterval)
s_pixelsize = repr(pixelsize)
s_forcemap_pixelsize = repr(downsamplerate * pixelsize)
s_radius = repr(radius)
s_lower_threshold = repr(lower_threshold)
s_sigma_smooth = repr(sigma_smooth)
s_E = repr(E)
s_nu = repr(nu)
s_alpha = repr(alpha)

# Writing parameters to a .txt file
with open(path.replace("position", "TFM_parameters.txt"), "w") as file:
    # Writing the parameters to a text file
    file.write("Frame interval = " + s_finterval + " s \n" + "Pixel size = " + s_pixelsize + " µm \n" + "Forcemap pixel size = " + s_forcemap_pixelsize + " µm \n" +
               "Rolling ball radius for background substraction = " + s_radius + " pixel \n" + "Lower percentile threshold for contrast adjustment = " + s_lower_threshold + "\n" + 
               "Size of gaussian smoothing window = " + s_sigma_smooth + " pixel \n" + "Young's Modulus of TFM gel = " + s_E + " Pa \n" + 
               "Poisson's ratio of TFM gel = " + s_nu + "\n" + "Regularization parameter = " + s_alpha)

    # Closing the file
    file.close

# for position in np.arange(no_stacks):
for position in [0]:

    # position = 1
    path_BK = path + str(position) + "/BK.tif"
    path_AK = path + str(position) + "/AK.tif"
    path_supp_stack = path + str(position) + "/" + supp_stack_file_name + ".tif"
    path_pattern = path + str(position) + "/pattern.tif"

    path_BK_registered = path + str(position) + "/preprocessed_images/BK_registered_avg_z.tif"
    path_AK_registered = path + str(position) + "/preprocessed_images/AK_registered_avg_z.tif"
    path_supp_stack_registered = path + str(position) + "/preprocessed_images/" + supp_stack_file_name + "_registered_avg_z.tif"
    path_pattern_registered = path + str(position) + "/preprocessed_images/pattern_registered.tif"

    savepath_preprocessing = path + str(position) + "/preprocessed_images/"
    savepath_TFM = path + str(position) + "/TFM_data/"

    if not os.path.exists(savepath_preprocessing):
        os.mkdir(savepath_preprocessing)

    if not os.path.exists(savepath_TFM):
        os.mkdir(savepath_TFM)


    preprocess_3Dt(path_BK, path_AK, path_supp_stack, path_pattern, savepath_preprocessing, supp_stack_file_name, pixelsize, finterval, radius=radius, lower_threshold=lower_threshold, sigma_smooth=sigma_smooth)
    measure_displacements(path_BK_registered, path_AK_registered, savepath_TFM, finterval, pixelsize, downsamplerate=downsamplerate)
    calculate_forces(savepath_TFM, pixelsize, dmax=dmax, pmax=pmax, downsamplerate=downsamplerate, E=E, nu=nu, alpha=alpha)
    make_TFM_movies(path_supp_stack_registered, supp_stack_file_name, savepath_TFM, pixelsize, dmax=dmax, pmax=pmax)