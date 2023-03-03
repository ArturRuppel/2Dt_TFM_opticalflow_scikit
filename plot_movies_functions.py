"""

@author: Artur Ruppel

"""

from moviepy.editor import *
import moviepy.video.io.ImageSequenceClip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

DPI = 300

def make_movies_from_images(folder, filename, no_frames, fps=10):
    image_files = []
    for frame in range(no_frames):
        image_files.append(folder + filename + str(frame) + ".png")

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_gif(folder + filename + ".gif", fps=fps)

    for img in image_files:
        os.remove(img)


def plot_registration_images(moving_image_morphed, fixed_image, ax, pixelsize, frame=0):
    O = np.zeros(fixed_image.shape)
    fixed_image_RGB = np.stack((O, fixed_image / 2, fixed_image), axis=2)
    moving_image_morphed_RGB = np.stack((moving_image_morphed, moving_image_morphed / 2, O), axis=2)
    ax.imshow(1 - fixed_image_RGB - moving_image_morphed_RGB, origin="upper")
    ax.set_axis_off()

    x_end = fixed_image.shape[0]

    # add time label
    xpos = x_end / 30
    ypos = x_end - x_end / 12

    if frame < 10:
        plt.text(xpos, ypos, str(frame * 2.5) + "min", color="black", backgroundcolor="white")
    else:
        plt.text(xpos, ypos, str(frame * 2.5) + "min", color="black", backgroundcolor="white")

    # draw a bar of 10 micron width
    plt.text(xpos, 2 * xpos, "10 µm", color="black", backgroundcolor="white")
    rect = mpl.patches.Rectangle((2.5 * xpos, xpos / 10), 1e-5 / pixelsize, 1e-6 / pixelsize, edgecolor='black', facecolor="black")
    ax.add_patch(rect)

def make_displacement_plots(d_x, d_y, dmax, savepath, forcemap_pixelsize, frame=0):
    dmax = dmax  # in m
    axtitle = "micron"  # unit of colorbar
    suptitle = "Displacement"  # title of plot
    x_end = d_x.shape[1]  # create x- and y-axis for plotting maps
    y_end = d_x.shape[0]
    n_arrows = 60 # about n_arrows will be plotted per line
    n = int(x_end / n_arrows)  # every nth arrow will be plotted
    if n == 0:
        n = 1
    extent = [0, x_end, 0, y_end]
    xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))  # create mesh for vectorplot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.7, 3))  # create figure and axes
    plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
    # ******************************************************************************************************************************************
    d_x_plot = d_x
    d_y_plot = d_y
    # w_plot = w * 1e6  # convert to micron

    displacement = np.sqrt(d_x_plot ** 2 + d_y_plot ** 2)
    # displacement = w_plot

    im = ax.imshow(displacement, cmap="inferno", interpolation="bilinear", extent=extent,
                   vmin=0, vmax=dmax, aspect="auto", origin="upper")
    ax.quiver(xq[::n, ::n], yq[::-n, ::-n], d_x_plot[::n, ::n], -d_y_plot[::n, ::n], 
              angles="xy", scale=10 * dmax, units="width", color="r")           #-n to invert the orientation of the meshgrid according to y and -v_plot to invert the y-coordinates as well

    ax.axis("off")
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

    # add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_title(axtitle)

    # add title
    plt.suptitle(suptitle, y=0.95, x=0.44)

    # add time label
    xpos = x_end / 30
    ypos = x_end - x_end / 12

    if frame < 10:
        plt.text(xpos, ypos, str(frame * 2.5) + "min", color="white")
    else:
        plt.text(xpos, ypos, str(frame * 2.5) + "min", color="white")

    # draw a bar of 10 micron width
    plt.text(xpos, 2 * xpos, "10 µm", color="white")
    rect = mpl.patches.Rectangle((2.5 * xpos, xpos), 1e-5 / forcemap_pixelsize, 1e-6 / forcemap_pixelsize, edgecolor='white', facecolor="white")
    ax.add_patch(rect)

    # # draw stimulation region
    # if frame >= 20 and frame < 30:
    #     rect = mpl.patches.Rectangle((15, 15), 30, 82, edgecolor='cyan', facecolor='none', lw=3)
    #     ax.add_patch(rect)

    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")


def make_traction_plots(t_x, t_y, pmax, savepath, forcemap_pixelsize, frame=0):
    pmax = pmax  # in Pa
    axtitle = "kPa"  # unit of colorbar
    suptitle = "Traction"  # title of plot
    x_end = np.shape(t_x)[1]  # create x- and y-axis for plotting maps
    y_end = np.shape(t_x)[0]
    n_arrows = 60  # about n_arrows will be plotted per line
    n = int(x_end / n_arrows)  # every nth arrow will be plotted
    if n == 0:
        n = 1
    extent = [0, x_end, 0, y_end]
    xq, yq = np.meshgrid(np.linspace(0, extent[1], x_end), np.linspace(0, extent[3], y_end))  # create mesh for vectorplot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.7, 3))  # create figure and axes
    plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
    # ******************************************************************************************************************************************
    t_x_plot = t_x * 1e-3  # convert to kPa
    t_y_plot = t_y * 1e-3  # convert to kPa
    t = np.sqrt(t_x_plot ** 2 + t_y_plot ** 2)

    im = ax.imshow(t, cmap="turbo", interpolation="bilinear", extent=extent,
                   vmin=0, vmax=pmax, aspect="auto", origin="upper")
    ax.quiver(xq[::n, ::n], yq[::-n, ::-n], t_x_plot[::n, ::n], -t_y_plot[::n, ::n],
              angles="xy", scale=10 * pmax, units="width", color="r")       #-n to invert the orientation of the meshgrid according to y and -v_plot to invert the y-coordinates as well

    ax.axis("off")
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

    # add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_title(axtitle)

    # add title
    plt.suptitle(suptitle, y=0.95, x=0.44)

    # add time label
    xpos = x_end / 30
    ypos = x_end - x_end / 12

    if frame < 10:
        plt.text(xpos, ypos, str(frame * 2.5) + "min", color="white")
    else:
        plt.text(xpos, ypos, str(frame * 2.5) + "min", color="white")

    # draw a bar of 10 micron width
    plt.text(xpos, 2 * xpos, "10 µm", color="white")
    rect = mpl.patches.Rectangle((2.5 * xpos, xpos), 1e-5 / forcemap_pixelsize, 1e-6 / forcemap_pixelsize, edgecolor='white',
                                 facecolor="white")
    ax.add_patch(rect)

    # draw stimulation region
    # if frame >= 20 and frame < 30:
    #     rect = mpl.patches.Rectangle((15, 15), 30, 82, edgecolor='cyan', facecolor='none', lw=3)
    #     ax.add_patch(rect)

    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")


def make_stress_plots(sigma, sigma_max, savepath, title, forcemap_pixelsize, frame=0):
    axtitle = "mN/m"  # unit of colorbar
    suptitle = title  # title of plot
    x_end = sigma.shape[1]  # create x- and y-axis for plotting maps
    y_end = sigma.shape[0]
    extent = [0, x_end, 0, y_end]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.7, 3))  # create figure and axes
    plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots
    # ******************************************************************************************************************************************
    sigma_plot = sigma * 1e3  # convert to mN/m

    im = ax.imshow(sigma_plot, cmap="cividis", interpolation="bilinear", extent=extent,
                   vmin=0, vmax=sigma_max, aspect="auto", origin="upper")

    ax.axis("off")
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

    # add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_title(axtitle)

    # add title
    plt.suptitle(suptitle, y=0.95, x=0.44)

    # add time label
    xpos = x_end / 30
    ypos = x_end - x_end / 12

    if frame < 10:
        plt.text(xpos, ypos, str(frame * 2.5) + "min", color="white")
    else:
        plt.text(xpos, ypos, str(frame * 2.5) + "min", color="white")

    # draw a bar of 10 micron width
    plt.text(xpos, 2 * xpos, "10 µm", color="white")
    rect = mpl.patches.Rectangle((2.5 * xpos, xpos), 1e-5 / forcemap_pixelsize, 1e-6 / forcemap_pixelsize, edgecolor='white',
                                 facecolor="white")
    ax.add_patch(rect)

    # draw stimulation region
    # if frame >= 20 and frame < 30:
    #     rect = mpl.patches.Rectangle((15, 15), 30, 82, edgecolor='cyan', facecolor='none', lw=3)
    #     ax.add_patch(rect)

    plt.savefig(savepath, dpi=DPI, bbox_inches="tight")


def plot_image_with_forces(image, t_x, t_y, pmax, savepath, pixelsize, frame=0):
    # set up plot parameters
    axtitle = 'kPa'  # unit of colorbar
    extent = [0, t_x.shape[1], 0, t_y.shape[0]]
    xq, yq = np.meshgrid(np.linspace(0, extent[1], t_x.shape[1]), np.linspace(0, extent[3], t_x.shape[0]))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.7, 3))  # create figure and axes
    plt.subplots_adjust(wspace=0.02, hspace=-0.06)  # adjust space in between plots

    n_arrows = 40 # about n_arrows will be plotted per line
    n = int(t_x.shape[1] / n_arrows)  # every nth arrow will be plotted
    if n == 0:
        n = 1

    x = xq[::n, ::n].flatten()
    y = yq[::-n, ::-n].flatten()

    t = np.sqrt(t_x ** 2 + t_y ** 2)

    t_x_flat = t_x[::n, ::n].flatten() * 1e-3
    t_y_flat = -t_y[::n, ::n].flatten() * 1e-3
    t_flat = t[::n, ::n].flatten() * 1e-3

    norm = mpl.colors.Normalize()
    norm.autoscale([0, pmax])
    colormap = mpl.cm.turbo
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    # plot actin image
    ax.imshow(image, cmap=plt.get_cmap("Greys"), interpolation="bilinear", extent=extent, origin="upper")

    # plot forces
    ax.quiver(x, y, t_x_flat, t_y_flat, angles='xy', scale_units='xy', scale=0.06 * pmax, color=colormap(norm(t_flat)), width=0.002)

    # ax.set_xlim([-0.1 * extent[1], 1.1 * extent[1]])
    # ax.set_ylim([-0.1 * extent[3], 1.1 * extent[3]])

    ax.axis('off')
    aspectratio = 1.0
    ratio_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(ratio_default * aspectratio)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_title(axtitle)

    x_end = t_x.shape[0]

    # add time label
    xpos = x_end / 30
    ypos = x_end - x_end / 12

    if frame < 10:
        plt.text(xpos, ypos, str(frame * 2.5) + "min", color="black", backgroundcolor="white")
    else:
        plt.text(xpos, ypos, str(frame * 2.5) + "min", color="black", backgroundcolor="white")

    # draw a bar of 10 micron width
    plt.text(xpos, 2 * xpos, "10 µm", color="black", backgroundcolor="white")
    rect = mpl.patches.Rectangle((2.5 * xpos, xpos / 10), 1e-5 / pixelsize, 1e-6 / pixelsize, edgecolor='black', facecolor="black")
    ax.add_patch(rect)

    # draw stimulation region
    # if frame >= 20 and frame < 30:
    #     rect = mpl.patches.Rectangle((15, 15), 30, 82, edgecolor='cyan', facecolor='none', lw=3)
    #     ax.add_patch(rect)

    fig.savefig(savepath, dpi=DPI, bbox_inches="tight")
    plt.close()

    return sm
