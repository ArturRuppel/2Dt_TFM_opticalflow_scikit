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
from skimage.morphology import reconstruction, dilation
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
from pyTFM.grid_setup_solids_py import prepare_forces
from pyTFM.grid_setup_solids_py import grid_setup, FEM_simulation
from pyTFM.grid_setup_solids_py import find_borders
from pyTFM.stress_functions import lineTension
from pyTFM.plotting import plot_continuous_boundary_stresses
# from pyTFM.stress_functions import *
# from pyTFM.utilities_TFM import make_random_discrete_color_range, join_dictionary
from plot_movies_functions import *

DPI = 300




def main(path_mask_stack, path_mask_borders_stack, savepath, forcemap_pixelsize, sigma_max=0.5, dmax=2, pmax=2, lambda_max=300):
    print("Loading image stack")
    mask_all = io.imread(path_mask_stack)[:, :, :]  # / (2 ** 16)
    mask_borders_all = io.imread(path_mask_borders_stack)[:, :, :]
    # load data
    u = np.load(savepath + "u.npy")    
    v = np.load(savepath + "v.npy")
    t_x = np.load(savepath + "t_x.npy")    
    t_y = np.load(savepath + "t_y.npy")  


    # calculate cell/tissue stresses (MSM)
    sigma_xx = np.zeros(t_x.shape)
    sigma_yy = np.zeros(t_x.shape)

    for frame in range(no_frames):
        # frame=3
        mask = dilation(mask_all[frame, :, :], disk(10)) > 0
        mask_borders = dilation(mask_borders_all[frame, :, :], disk(2))

        print("Calculating stress for frame: " + str(frame))
        f_x, f_y = prepare_forces(t_x[:, :, frame], t_y[:, :, frame], forcemap_pixelsize * 1e6, mask)
        # f = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
    
        # construct FEM grid
        nodes, elements, loads, mats = grid_setup(mask, -f_x, -f_y, sigma=0.5)
        # performing FEM analysis
        UG_sol, stress_tensor = FEM_simulation(nodes, elements, loads, mats, mask, verbose=False)
    
        sigma_xx[:, :, frame] = stress_tensor[:, :, 0, 0] / forcemap_pixelsize
        sigma_yy[:, :, frame] = stress_tensor[:, :, 1, 1] / forcemap_pixelsize
        
        borders = find_borders(mask_borders, t_x.shape)
        # we can for example get the number of cells from the "borders" object
        n_cells = borders.n_cells # 8
        # calculating the line tension along the cell borders
        lt, min_v, max_v = lineTension(borders.lines_splines, borders.line_lengths, stress_tensor, pixel_length=forcemap_pixelsize)
        # lt is a nested dictionary. The first key is the id of a cell border.
        # For each cell border the line tension vectors ("t_vecs"), the normal
        # and shear component of the line tension ("t_shear") and the normal
        # vectors of the cell border ("n_vecs") are calculated at a large number of points.
        
        # average norm of the line tension. Only borders not at colony edge are used
        lt_vecs = np.concatenate([lt[l_id]["t_vecs"] for l_id in lt.keys() if l_id not in borders.edge_lines])
        avg_line_tension = np.mean(np.linalg.norm(lt_vecs, axis=1)) # 0.00569 N/m
        
        # average normal component of the line tension
        lt_normal = np.concatenate([lt[l_id]["t_normal"] for l_id in lt.keys() if l_id not in borders.edge_lines])
        avg_normal_line_tension = np.mean(np.abs(lt_normal)) # 0.00566 N/m,
        
        

        # plotting the line tension
        fig3, ax = plot_continuous_boundary_stresses([borders.inter_shape, borders.edge_lines, lt, 0, lambda_max], cbar_style="outside")
        plt.savefig(savepath + "linetension_" + str(frame), dpi=DPI, bbox_inches="tight")
    
        make_stress_plots(sigma_xx[:, :, frame], sigma_max, savepath + "xx-Stress_" + str(frame), "xx-Stress", forcemap_pixelsize, frame=frame)
        make_stress_plots(sigma_yy[:, :, frame], sigma_max, savepath + "yy-Stress_" + str(frame), "yy-Stress", forcemap_pixelsize, frame=frame)
        make_stress_plots((sigma_yy[:, :, frame] + sigma_xx[:, :, frame]) / 2, sigma_max, savepath +
                          "avg-Stress_" + str(frame), "Avg. normal stress", forcemap_pixelsize, frame=frame)
    
    np.save(savepath + "sigma_xx.npy", sigma_xx)
    np.save(savepath + "sigma_yy.npy", sigma_yy)

    make_movies_from_images(savepath, "linetension_", no_frames)
    make_movies_from_images(savepath, "xx-Stress_", no_frames)
    make_movies_from_images(savepath, "yy-Stress_", no_frames)
    make_movies_from_images(savepath, "avg-Stress_", no_frames)

    # strain_energy_density = 0.5 * (t_x * u + t_y * v) * forcemap_pixelsize ** 2
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
    no_stacks = 1  # number of stacks to be analyzed
    no_frames = 20  # number of frames in a stack to be analyzed
    pixelsize = 0.325 * 1e-6  # size of a pixel in the bead image in m
    downsamplerate = 1  # final forcemaps will have a resolution of the image divided by this number
    dmax = 1  # max displacement for maps in micron
    pmax = 0.1  # max traction for maps in kPa
    lambda_max = 600    # max line tension in N/m


    path = "D:/2022-10-21 TFM dragonfly induced mesoderm GAP-GFP 3kPa/position"


    for cell in np.arange(no_stacks):
        # cell = 6
        savepath = path + str(cell) + "/"
        path_mask_stack = path + str(cell) + "/mask.tif"
        path_mask_borders_stack = path + str(cell) + "/mask_borders.tif"


        if not os.path.exists(savepath):
            os.mkdir(savepath)
        main(path_mask_stack, path_mask_borders_stack, savepath, downsamplerate*pixelsize, dmax=dmax, pmax=pmax, lambda_max=lambda_max)
        


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
