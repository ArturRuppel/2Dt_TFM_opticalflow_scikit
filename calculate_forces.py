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




def calculate_traction_stresses(u, v, E, nu, pixelsize, alpha):
    '''This function takes a displacement field u and v, a gel rigidity E, it's poisson ratio nu, the size of a pixel in '''
    M, N = u.shape

    # pad displacement map with zeros until it has a shape of 2^n by 2^n because fourier transform is faster
    n = 2
    while (2 ** n < M) or (2 ** n < N):
        n = n + 1

    M2 = 2 ** n
    N2 = M2

    u_padded = np.zeros((M2, N2))
    v_padded = np.zeros((M2, N2))

    u_padded[:u.shape[0], :u.shape[1]] = u
    v_padded[:v.shape[0], :v.shape[1]] = v

    u_fft = fft2(u_padded)
    v_fft = fft2(v_padded)

    # remove component related to translation
    u_fft[0, 0] = 0
    v_fft[0, 0] = 0

    Kx1 = (2 * np.pi / pixelsize) / N2 * np.arange(int(N2 / 2))
    Kx2 = -(2 * np.pi / pixelsize) / N2 * (N2 - np.arange(int(N2 / 2), N2))
    Ky1 = (2 * np.pi / pixelsize) / M2 * np.arange(int(M2 / 2))
    Ky2 = -(2 * np.pi / pixelsize) / M2 * (M2 - np.arange(int(M2 / 2), M2))

    Kx = np.concatenate((Kx1, Kx2))
    Ky = np.concatenate((Ky1, Ky2))

    kx, ky = np.meshgrid(Kx, Ky)
    k = np.sqrt(kx ** 2 + ky ** 2)
    t_xt = np.zeros((M2, N2), dtype=complex)
    t_yt = np.zeros((M2, N2), dtype=complex)

    for i in np.arange(M2):
        for j in np.arange(N2):
            if i == M2 / 2 or j == N2 / 2:  # Nyquist frequency
                Gt = np.zeros((2, 2))
                Gt[0, 0] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * ky[i, j] ** 2)
                Gt[1, 1] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * kx[i, j] ** 2)

                a = (Gt.T * Gt + alpha * np.eye(2)) ** -1 * Gt.T
                a[np.isnan(a)] = 0
                b = (u_fft[i, j], v_fft[i, j])
                Tt = np.dot(a, b)
                t_xt[i, j] = Tt[0]
                t_yt[i, j] = Tt[1]

            elif ~((i == 1) and (j == 1)):
                Gt = np.zeros((2, 2))
                Gt[0, 0] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * ky[i, j] ** 2)
                Gt[1, 1] = 2 * (1 + nu) / (E * k[i, j] ** 3) * ((1 - nu) * k[i, j] ** 2 + nu * kx[i, j] ** 2)
                Gt[0, 1] = - nu * kx[i, j] * ky[i, j]
                Gt[1, 0] = - nu * kx[i, j] * ky[i, j]

                a = (Gt.T * Gt + alpha * np.eye(2)) ** -1 * Gt.T
                a[np.isnan(a)] = 0
                b = (u_fft[i, j], v_fft[i, j])
                Tt = np.dot(a, b)
                t_xt[i, j] = Tt[0]
                t_yt[i, j] = Tt[1]

    t_x = ifft2(t_xt)
    t_y = ifft2(t_yt)
    traction_x = np.real(t_x)
    traction_y = np.real(t_y)

    return traction_x[0:M, 0:N], traction_y[0:M, 0:N]





def main(savepath, pixelsize, radius=10, th=10, sigma_smooth=1, dmax=2, pmax=2, sigma_max=5, downsamplerate=8,
         E=19960, nu=0.5, alpha=1*1e-19, cropwidth=0):


    forcemap_pixelsize = pixelsize * downsamplerate * 1e-6
    # mask = supp_stack[0, :, :] > -1
    

    # convert to boolean
    # mask = mask > 0

    # imageio.imwrite(savepath + "mask.tif", img_as_uint(mask))

    d_x = np.load(savepath + "d_x.npy") * forcemap_pixelsize
    d_y = np.load(savepath + "d_y.npy") * forcemap_pixelsize
    no_frames = d_x.shape[0]

    # make plots of displacement fields
    # for frame in np.arange(no_frames):           
    #     d_x_current = d_x[frame, :, :]
    #     d_y_current = d_y[frame, :, :]
        
        # make_displacement_plots(d_x_current, d_y_current, dmax, savepath + "displacement_" + str(frame) + ".png", forcemap_pixelsize, frame=frame)


    # calculate forces and make force plots
    t_x = np.zeros(d_x.shape)
    t_y = np.zeros(d_y.shape)
    for frame in range(no_frames):
        print("Calculating traction forces for frame: " + str(frame))
        t_x_current, t_y_current = calculate_traction_stresses(d_x[frame, :, :], d_y[frame, :, :], E, nu, forcemap_pixelsize, alpha)

        # t_x_current *= mask
        # t_y_current *= mask

        # make_traction_plots(t_x_current, t_y_current, pmax, savepath + "traction_" + str(frame), forcemap_pixelsize, frame=frame)

        t_x[frame, :, :] = t_x_current
        t_y[frame, :, :] = t_y_current

    np.save(savepath + "t_x.npy", t_x)
    np.save(savepath + "t_y.npy", t_y)

    # make_movies_from_images(savepath, "displacement_", no_frames)
    # make_movies_from_images(savepath, "traction_", no_frames)


    # strain_energy_density = 0.5 * (t_x * d_x + t_y * d_y) * forcemap_pixelsize ** 2
    # strain_energy = np.nansum(strain_energy_density, axis=(0, 1))


    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))  # create figure and axes
    # ax.plot(strain_energy[0:frame + 1] * 1e15)
    # plt.title("Strain energy [fJ]")
    # plt.xlim(0, no_frames)
    # plt.ylim(0, 20)
    # plt.xlabel("time [min]")
    # plt.savefig(savepath + "strain_energy.png", dpi=DPI, bbox_inches="tight")



# %%
if __name__ == "__main__":
    no_stacks = 7  # number of stacks to be analyzed
    pixelsize = 0.217  # size of a pixel in the bead image in m
    # pixelsize = 0.325           # size of a pixel in the bead image in µm
    dmax = 3  # max displacement for maps in micron
    pmax = 1  # max traction for maps in kPa
    downsamplerate = 1  # final forcemaps will have a resolution of the image divided by this number
    E = 30000  # rigidity of the hydrogel in Pa
    nu = 0.5  # poisson ratio of the hydrogel
    alpha = 1 * 1e-18  # regularization parameter for force calculation.

    # path = "D:/2022-10-21 dragonfly induced mesoderm GAP-GFP TFM 3kPa/position"
    path = "D:/2022-11-23  GEL TOO STIFF olympus SR induced mesoderm GAP-GFP TFM 30kPa/position"


    # for position in np.arange(no_stacks):
    for position in [2, 3, 4, 5, 6]:


            # position = 1
        # position = 1
        # path_supp_stack = path + str(position) + "/TFM_data/GAP-GFP_avg_z.tif"
        savepath = path + str(position) + "/TFM_data/"
    
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        main(savepath, pixelsize, dmax=dmax, pmax=pmax, downsamplerate=downsamplerate, E=E, nu=nu, alpha=alpha)

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
