"""
Implement utilities using GPU

Michael Chen   mchen0405@berkeley.edu
David Ren      david.ren@berkeley.edu
November 22, 2017
"""

import numpy as np
import arrayfire as af
import skimage.io as skio
import scipy.io as sio
import contexttimer
import tkinter
import matplotlib.pyplot as plt
from math import factorial
from scipy.ndimage.filters import uniform_filter
from tkinter.filedialog import askdirectory
from matplotlib.widgets import Slider
from os import listdir, path
import sys
from opticaltomography import settings

np_float_datatype   = settings.np_float_datatype
af_complex_datatype = settings.af_complex_datatype
MAX_DIM = 512*512*512 if settings.bit == 32 else 512*512*256



def show3DStack(image_3d, axis = 2, cmap = "gray", clim = (0, 1), extent = (0, 1, 0, 1)):
    if axis == 0:
        image  = lambda index: image_3d[index, :, :]
    elif axis == 1:
        image  = lambda index: image_3d[:, index, :]
    else:
        image  = lambda index: image_3d[:, :, index]

    current_idx= 0
    _, ax      = plt.subplots(1, 1, figsize=(6.5, 5))
    plt.subplots_adjust(left=0.15, bottom=0.15)
    fig        = ax.imshow(image(current_idx), cmap = cmap,  clim = clim, extent = extent)
    ax.set_title("layer: " + str(current_idx))
    plt.colorbar(fig, ax=ax)
    plt.axis('off')
    ax_slider  = plt.axes([0.15, 0.1, 0.65, 0.03])
    slider_obj = Slider(ax_slider, "layer", 0, image_3d.shape[axis]-1, valinit=current_idx, valfmt='%d')
    def update_image(index):
        global current_idx
        index       = int(index)
        current_idx = index
        ax.set_title("layer: " + str(index))
        fig.set_data(image(index))
    def arrow_key(event):
        global current_idx
        if event.key == "left":
            if current_idx-1 >=0:
                current_idx -= 1
        elif event.key == "right":
            if current_idx+1 < image_3d.shape[axis]:
                current_idx += 1
        slider_obj.set_val(current_idx)
    slider_obj.on_changed(update_image)
    plt.gcf().canvas.mpl_connect("key_release_event", arrow_key)
    plt.show()
    return slider_obj

def compare3DStack(stack_1, stack_2, axis = 2, cmap = "gray", clim = (0, 1), extent = (0, 1, 0, 1) , colorbar = True):
    assert stack_1.shape == stack_2.shape, "shape of two input stacks should be the same!"

    if axis == 0:
        image_1  = lambda index: stack_1[index, :, :]
        image_2  = lambda index: stack_2[index, :, :]
    elif axis == 1:
        image_1  = lambda index: stack_1[:, index, :]
        image_2  = lambda index: stack_2[:, index, :]
    else:
        image_1  = lambda index: stack_1[:, :, index]
        image_2  = lambda index: stack_2[:, :, index]

    current_idx  = 0
    _, ax        = plt.subplots(1, 2, figsize=(9, 5), sharex = 'all', sharey = 'all')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    fig_1        = ax[0].imshow(image_1(current_idx), cmap = cmap,  clim = clim, extent = extent)
    ax[0].axis("off")
    ax[0].set_title("stack 1, layer: " + str(current_idx))
    plt.colorbar(fig_1, ax = ax[0])
    fig_2        = ax[1].imshow(image_2(current_idx), cmap = cmap,  clim = clim, extent = extent)
    ax[1].axis("off")
    ax[1].set_title("stack 2, layer: " + str(current_idx))
    plt.colorbar(fig_2, ax = ax[1])
    ax_slider    = plt.axes([0.10, 0.05, 0.65, 0.03])
    slider_obj   = Slider(ax_slider, 'layer', 0, stack_1.shape[axis]-1, valinit=current_idx, valfmt='%d')
    def update_image(index):
        global current_idx
        index       = int(index)
        current_idx = index
        ax[0].set_title("stack 1, layer: " + str(index))
        fig_1.set_data(image_1(index))
        ax[1].set_title("stack 2, layer: " + str(index))
        fig_2.set_data(image_2(index))
    def arrow_key(event):
        global current_idx
        if event.key == "left":
            if current_idx-1 >=0:
                current_idx -= 1
        elif event.key == "right":
            if current_idx+1 < stack_1.shape[axis]:
                current_idx += 1
        slider_obj.set_val(current_idx)
    slider_obj.on_changed(update_image)
    plt.gcf().canvas.mpl_connect("key_release_event", arrow_key)
    plt.show()
    return slider_obj

def calculateNumericalGradient(func, x, point, delta = 1e-4):
    function_value_0  = func(x)
    x_shift           = x.copy()
    x_shift[point]   += delta
    function_value_re = func(x_shift)
    grad_re           = (function_value_re - function_value_0)/delta

    x_shift           = x.copy()
    x_shift[point]   += 1.0j* delta
    function_value_im = func(x_shift)
    grad_im           = (function_value_im - function_value_0)/delta
    gradient          = 0.5*(grad_re + 1.0j*grad_im)

    return gradient

def cart2Pol(x, y):
    rho          = (x * af.conjg(x) + y * af.conjg(y))**0.5
    theta        = af.atan2(af.real(y), af.real(x)).as_type(af_complex_datatype)
    return rho, theta

def genZernikeAberration(shape, pixel_size, NA, wavelength, z_coeff = [1], z_index_list = [0]):
    assert len(z_coeff) == len(z_index_list), "number of coefficients does not match with number of zernike indices!"

    pupil             = genPupil(shape, pixel_size, NA, wavelength)
    fxlin             = genGrid(shape[1], 1/pixel_size/shape[1], flag_shift = True)
    fylin             = genGrid(shape[0], 1/pixel_size/shape[0], flag_shift = True)
    fxlin             = af.tile(fxlin.T, shape[0], 1)
    fylin             = af.tile(fylin, 1, shape[1])
    rho, theta        = cart2Pol(fxlin, fylin)
    rho[:, :]        /= NA/wavelength

    def zernikePolynomial(z_index):
        n                    = int(np.ceil((-3.0 + np.sqrt(9+8*z_index))/2.0))
        m                    = 2*z_index - n*(n+2)
        normalization_coeff  = np.sqrt(2 * (n+1)) if abs(m) > 0 else np.sqrt(n+1)
        azimuthal_function   = af.sin(abs(m)*theta) if m < 0 else af.cos(abs(m)*theta)
        zernike_poly         = af.constant(0.0, shape[0], shape[1], dtype = af_complex_datatype)
        for k in range((n-abs(m))//2+1):
            zernike_poly[:, :]  += ((-1)**k * factorial(n-k))/ \
                                    (factorial(k)*factorial(0.5*(n+m)-k)*factorial(0.5*(n-m)-k))\
                                    * rho**(n-2*k)

        return normalization_coeff * zernike_poly * azimuthal_function

    for z_coeff_index, z_index in enumerate(z_index_list):
        zernike_poly = zernikePolynomial(z_index)

        if z_coeff_index == 0:
            zernike_aberration = z_coeff.ravel()[z_coeff_index] * zernike_poly
        else:
            zernike_aberration[:, :] += z_coeff.ravel()[z_coeff_index] * zernike_poly

    return zernike_aberration * pupil

def genPupil(shape, pixel_size, NA, wavelength):
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin        = genGrid(shape[1], 1/pixel_size/shape[1], flag_shift = True)
    fylin        = genGrid(shape[0], 1/pixel_size/shape[0], flag_shift = True)
    fxlin        = af.tile(fxlin.T, shape[0], 1)
    fylin        = af.tile(fylin, 1, shape[1])

    pupil_radius = NA/wavelength
    pupil        = (fxlin**2 + fylin**2 <= pupil_radius**2).as_type(af_complex_datatype)
    return pupil

def propKernel(shape, pixel_size, wavelength, prop_distance, NA = None, RI = 1.0, band_limited=True):
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin        = genGrid(shape[1], 1/pixel_size/shape[1], flag_shift = True)
    fylin        = genGrid(shape[0], 1/pixel_size/shape[0], flag_shift = True)
    fxlin        = af.tile(fxlin.T, shape[0], 1)
    fylin        = af.tile(fylin, 1, shape[1])

    if band_limited:
        assert NA is not None, "need to provide numerical aperture of the system!"
        Pcrop    = genPupil(shape, pixel_size, NA, wavelength)
    else:
        Pcrop    = 1.0

    prop_kernel  = Pcrop * af.exp(1.0j * 2.0 * np.pi * abs(prop_distance) * Pcrop *\
                                  ((RI/wavelength)**2 - fxlin**2 - fylin**2)**0.5)
    prop_kernel  = af.conjg(prop_kernel) if prop_distance < 0 else prop_kernel
    return prop_kernel

def genGrid(size, dx, flag_shift = False):
    """
    This function generates 1D Fourier grid, and is centered at the middle of the array
    Inputs:
        size    - length of the array
        dx      - pixel size
    Optional parameters:
        flag_shift - flag indicating whether the final array is circularly shifted
                     should be false when computing real space coordinates
                     should be true when computing Fourier coordinates
    Outputs:
        xlin       - 1D Fourier grid

    """
    xlin = (af.range(size) - size//2) * dx
    if flag_shift:
        xlin = af.shift(xlin, -1 * size//2)
    return xlin.as_type(af_complex_datatype)

class EmptyClass:
    def __init__(self):
        pass
    def forward(self, field, zernike_coeffs=None, x_shift=None, y_shift=None, **kwargs):
        return field
    def adjoint(self, field, zernike_coeffs=None, x_shift=None, y_shift=None, **kwargs):
        return field

class ImageCrop:
    """
    Class for image cropping
    """
    def __init__(self, shape, pad = True, pad_size = None, **kwargs):
        """
        shape:          shape of object (y,x,z)
        pixel_size:     pixel size of the system
        pad:            boolean variable to pad the reconstruction
        pad_size:       if pad is true, default pad_size is shape//2. Takes a tuple, pad size in dimensions (y, x)
        """
        self.shape          = shape
        self.pad            = pad
        if self.pad:
            self.pad_size       = pad_size
            if self.pad_size == None:
                self.pad_size   = (self.shape[0]//4, self.shape[1]//4)
            self.row_crop   = slice(self.pad_size[0], self.shape[0] - self.pad_size[0])
            self.col_crop   = slice(self.pad_size[1], self.shape[1] - self.pad_size[1])
        else:
            self.row_crop   = slice(0, self.shape[0])
            self.col_crop   = slice(0, self.shape[1])

    def forward(self, field):
        return field[self.row_crop, self.col_crop]

    def adjoint(self, residual):
        if len(residual.shape) > 2:
            field_pad = af.constant(0.0, self.shape[0], self.shape[1], residual.shape[2], dtype = af_complex_datatype)
            for z_idx in range(field_pad.shape[2]):
                field_pad[self.row_crop, self.col_crop, z_idx] = residual[:, :, z_idx] + 0.0j
        else:
            field_pad = af.constant(0.0, self.shape[0], self.shape[1], dtype = af_complex_datatype)
            field_pad[self.row_crop, self.col_crop] = residual + 0.0j
        return field_pad

