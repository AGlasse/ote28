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
            print("!! Plot '{:s}' has been skipped".format(title))
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

        osample = kwargs.get('osample', 1.0)    # Oversample factor (=4 for CDP PSFs)
        r_max = kwargs.get('r_max', 20.0)
        lc_list = kwargs.get('lc_list', None)
        ls_list = kwargs.get('ls_list', None)
        lw_list = kwargs.get('lw_list', None)
        show_ref_rad = kwargs.get('show_ref_rad', True)

        title = kwargs.get('title', '')

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
#            eef_mean = None
#            if plot_average:
#                eef_mean = np.squeeze(np.mean(eevr_list, axis=0), axis=0)
#                eef_std = np.squeeze(np.std(eevr_list, axis=0), axis=0)
#                plt.plot(rl, eef_mean, color='blue', ls=ls, lw=1.5)
#                ee_sample = np.interp(r_sample, radii, eef_mean)
#                ee_sample_std = np.interp(r_sample, radii, eef_std)
#                lrs = math.log10(r_sample)
#                plt.scatter(lrs, ee_sample, color='blue', marker='o')
#                text = "EE @ {:3.1f} pix = {:4.3f} +- {:4.3f}".format(r_sample, ee_sample, ee_sample_std)
#                plt.text(lrs, 0.95*ee_sample, text, color='blue', va='top')
        if show_ref_rad:
            x = math.log10(Globals.ref_radius)
            plt.plot([x, x], [0.0, 1.0], color='red', ls='dotted', lw=1.0)
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
            _, _, _, fit = profile
            _, fwhm, _ = fit[0]
            fwhms.append(fwhm)

        plt.title(title)
        plt.xlabel('OSIM focus / mm')
        plt.ylabel('FWHM / pixels')
        plt.plot(foci, fwhms, marker='o', fillstyle='none', ls='none')
        plt.show()
        return



    def plot_profile_list(self, profile_list, xlim, **kwargs):

        fit_profile = kwargs.get('fit_profile', None)
        normalise = kwargs.get('normalise', False)
        title = kwargs.get('title', '')
        lc_list = kwargs.get('lc_list', ['blue'])
        ls_list = kwargs.get('ls_list', ['solid'])
        lw_list = kwargs.get('lw_list', [1.5])

        plt.xlabel('pixel')
        plt.ylabel('profile')
        plt.title(title)
        plt.xlim(xlim)

        for j, profile in enumerate(profile_list):
            lc = 'grey' if lc_list is None else lc_list[j]
            ls = 'solid' if ls_list is None else ls_list[j]
            lw = 1.5 if lw_list is None else lw_list[j]

            identifier, u_vals, p, params = profile
            fit, covar = params
            amp, fwhm, xpk = fit
            if normalise:
                pmax = np.max(p)
                p = p / pmax
                amp /= pmax

            plt.plot(u_vals, p, color=lc, ls=ls, lw=lw, marker='o', fillstyle='none')
            if profile is fit_profile:
                text = "FWHM = {:5.3f} pix".format(fwhm)
                plt.text(3.0, 0.25, text, color=lc, va='top')
                xf = np.arange(u_vals[0], u_vals[-1], 0.1)
                yf = Globals.Gauss(xf, amp, fwhm, xpk)
                plt.plot(xf, yf, color=lc, lw=1.5, ls='dotted')
        plt.show()
        return

    def plot_cv3_eefs(self, eef_list, code_list, f0rad, f0ee):
        import math

        fig, ax = plt.subplots()
        plt.xlabel('log10(Radius / pixel)')
        plt.ylabel('Encircled energy fraction')

        xtick_lin_vals = np.array([0.1, 1, 2, 5, 10, 20])
        xtick_vals = np.log10(xtick_lin_vals)
        plt.xticks(xtick_vals, xtick_lin_vals)
        plt.xlim(np.log10([0.1, 20.0]))

        n_cv3 = len(code_list)
        i = 0
        for radii, eef in eef_list:

            rl = np.log10(radii)
            colour, lw, ls = 'grey', 1.0, 'solid'
            if i < n_cv3:
                colour, lw, ls = get_cv3_plot_parameters(code_list[i])
            if i > n_cv3:  # Kludge to suppress plotting out of focus data
                plt.plot(rl, eef, color=colour, ls=ls, lw=lw)
            i += 1

        radii, eef = eef_list[0]  # Plot CV3 in focus on top
        rl = np.log10(radii)
        colour, lw, ls = get_cv3_plot_parameters(code_list[0])
        plt.plot(rl, eef, color=colour, lw=lw, ls=ls)
        f0radl = math.log10(f0rad)
        plt.plot([f0radl], [f0ee], color=colour, marker='o')
        plt.text(math.log10(2.1), f0ee, '61.8 % @ r=1.9 pix.', color=colour, va='top', ha='left')
        plt.show()
        return
