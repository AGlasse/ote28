#!/usr/bin/env python
import math

from matplotlib import style, pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import figure
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from ote28_globals import Globals as Globals
import numpy as np
import math


class Ote28Plot:

    def __init__(self):
        rcParams['figure.figsize'] = [8., 5.]
        plt.rcParams.update({'font.size': 12})
        return

    def get_cv3_plot_parameters(self, code):
        """ Returns matplotlib colour and linestyle codes based on cv3 data id. """
        colours = {'F': 'green', 'M': 'blue', 'P': 'red'}
        side = code[0]
        f_mm = float(code[1])
        colour = colours[side]
        lw = 2.0 if f_mm < 1.0 else 1.0
        ls = 'solid' if f_mm < 1.0 else 'dashed'
        return colour, lw, ls

    def display(self, image, title, **kwargs):
        skip = kwargs.get('skip', False)
        if skip:
            print("!! Display of image '{:s}' has been skipped".format(title))
            return

        stars = kwargs.get('stars', [])
        units = kwargs.get('units', 'arb. units')
        is_log = kwargs.get('log', False)
        xmax, ymax = image.shape
        xlim = kwargs.get('xlim', [0, xmax - 1])
        ylim = kwargs.get('ylim', [0, ymax - 1])
        xmin, xmax, ymin, ymax = xlim[0], xlim[1], ylim[0], ylim[1]

        img = image[xlim[0]:xlim[1] + 1, ylim[0]:ylim[1] + 1]
        vmed, vstd = np.nanmedian(img), np.nanstd(img)
        vmin, vmax = np.nanmin(img), np.nanmax(img)

        vmin = kwargs.get('vmin', vmin)
        vmax = kwargs.get('vmax', vmax)
        if 'sigma_cut' in kwargs:
            sc = kwargs.get('sigma_cut', 3.0)
            vmin, vmax = vmed - sc * vstd, vmed + sc * vstd

        if is_log:
            vmin, vmax = math.log10(vmin), math.log10(vmax)
            img = np.abs(img)
            img = np.log10(img)
            units = "log10(abs(val) / {:s})".format(units)

        fig, ax = plt.subplots()
        plt.xlim(xmin - 1, xmax + 1)
        plt.ylim(ymin - 1, xmax + 1)
        fig = plt.imshow(img,
                         extent=(xmin - 0.5, xmax + 0.5, ymin - 0.5, ymax + 0.5),
                         interpolation='nearest', cmap='binary',  # cmap='hot'
                         vmin=vmin, vmax=vmax, origin='lower')
        cbar = plt.colorbar()
        cbar.set_label(units)

        xc, yc = [], []
        for star in stars:
            xc.append(star['xcentroid'])
            yc.append(star['ycentroid'])
        plt.plot(xc, yc, color='r', marker='o', fillstyle='none', ls='none')

#        for star in stars:
#            xc, yc = star[1], star[2]
#            plt.scatter([xc], [yc], lw=0, s=30, color='r', marker='o', fillstyle='none')
        plt.title(title, fontsize='medium')
        plt.show()
        return

    def histogram(self, img, title, **kwargs):
        skip = kwargs.get('skip', False)
        if skip:
            print("!! Histogram '{:s}' has been skipped".format(title))
            return

        bins = kwargs.get('bins', 5)
        histtype = kwargs.get('histtype', 'step')
        vlim = kwargs.get('xlim', (0.7, 0.9))

        fig, ax = plt.subplots()
        flat_img = img.flatten()
        fig = plt.hist(flat_img, bins=bins, histtype=histtype)
        plt.xlim(vlim[0], vlim[1])
        plt.xlabel('Background signal (Mjy/sr/pixel)')
        plt.ylabel('Pixel count')
        plt.title(title, fontsize='medium')
        plt.show()
        return

    def plot_eef_list(self, eevr_list, **kwargs):
        title = kwargs.get('title', '')
        skip = kwargs.get('skip', False)
        if skip:
            print("!! EE v r plot '{:s}' has been skipped".format(title))
            return

        osample = kwargs.get('osample', 1.0)    # Oversample factor (=4 for CDP PSFs)
        r_max = kwargs.get('r_max', 20.0)
        lc_list = kwargs.get('lc_list', None)
        ls_list = kwargs.get('ls_list', None)
        lw_list = kwargs.get('lw_list', None)
        ref_ee = kwargs.get('ref_ee', (False, None, None))

        show_ref_rad, ee_ref, ee_ref_std = ref_ee

        plt.xlabel('log10(Radius / pixel)')
        plt.ylabel('Encircled energy fraction')
        plt.title(title)

        xtick_lin_vals = np.array([0.5, 1, 2, 5, 10, r_max/ osample])
        xtick_vals = np.log10(xtick_lin_vals)
        plt.xticks(xtick_vals, xtick_lin_vals)
        plt.xlim(np.log10([0.5, r_max/ osample]))
        n_obs = len(eevr_list)
        for j in range(0, n_obs):

            lc = 'grey' if lc_list is None else lc_list[j]
            ls = 'solid' if ls_list is None else ls_list[j]
            lw = 1.5 if lw_list is None else lw_list[j]
            _, base_radii, ees, _ = eevr_list[j]
            radii = base_radii / osample
            rl = np.log10(radii)
            n_profiles, n_radii = ees.shape
            for i in range(0, n_profiles):
                plt.plot(rl, ees[i, :], color=lc, ls=ls, lw=lw)
        if show_ref_rad:
            x = math.log10(Globals.ref_radius)
            plt.errorbar([x], [ee_ref], yerr=ee_ref_std, fmt='ok')
            text = "{:5.3f}+-{:5.3f} ".format(ee_ref, ee_ref_std)
            plt.text(x, ee_ref - 0.05, text, color='black', va='top')
        plt.show()
        return

    def plot_profile_list(self, profile_list, xlim, **kwargs):
        title = kwargs.get('title', '')
        skip = kwargs.get('skip', False)
        if skip:
            print("!! Profile plot '{:s}' has been skipped".format(title))
            return

        best_fit = kwargs.get('best_fit', None)
        normalise = kwargs.get('normalise', False)
        lc_list = kwargs.get('lc_list', None)
        ls_list = kwargs.get('ls_list', None)
        lw_list = kwargs.get('lw_list', None)

        plt.xlabel('pixel')
        plt.ylabel('profile')
        plt.title(title)
        plt.xlim(xlim)

        for j, profile in enumerate(profile_list):
            lc = 'grey' if lc_list is None else lc_list[j]
            ls = 'solid' if ls_list is None else ls_list[j]
            lw = 1.5 if lw_list is None else lw_list[j]

            identifier, ui_vals, uf_vals, p, params = profile
            star, fit, covar = params
            amp, fwhm, xpk = fit
            if normalise:
                pmax = np.max(p)
                p = p / pmax
                amp /= pmax
            plt.plot(uf_vals, p, color=lc, ls=ls, lw=lw, marker='o', fillstyle='none')

        if best_fit is not None:
            fwhm, fwhm_err = best_fit
            text = "FWHM = {:5.3f}+-{:5.3f} pix".format(fwhm, fwhm_err)
            plt.text(1.5, 0.5, text, color='black', va='top')
            xf = np.arange(uf_vals[0], uf_vals[-1], 0.1)
            yf = Globals.Gauss(xf, 1.0, fwhm, 0.0)
            plt.plot(xf, yf, color='black', lw=2.0, ls='dotted')
        plt.show()
        return

    def plot_cv3_eevfocus(self, foci, eevr_list):
        plt.title('CV3 EE (at r = 1.9 pixels) v focus')
        plt.xlabel('OSIM focus / mm')
        plt.ylabel('EE')
        ee_ave_refs = []
        for eevr in eevr_list:
            name, _, _, ee_refs = eevr
            ee_ave_ref = np.mean(ee_refs)   # Average EE ref values for all stars
            ee_ave_refs.append(ee_ave_ref)
        plt.plot(foci, ee_ave_refs, marker='o', fillstyle='none', ls='none')
        plt.show()
        return

    def plot_cv3_fwhmvfocus(self, foci, profile_list, title):
        fwhms = []
        for profile in profile_list:
            _, _, _, _, params = profile
            _, fit, _ = params
            fwhm = fit[1]
            fwhms.append(fwhm)

        plt.title(title)
        plt.xlabel('OSIM focus / mm')
        plt.ylabel('FWHM / pixels')
        plt.plot(foci, fwhms, marker='o', fillstyle='none', ls='none')
        plt.show()
        return
