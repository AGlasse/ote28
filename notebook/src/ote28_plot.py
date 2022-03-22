#!/usr/bin/env python
import math

from matplotlib import style, pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import figure
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from ote28_globals import Ote28Globals as Globals
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
            return

        units = kwargs.get('units', 'arb. units')
        is_log = kwargs.get('log', False)
        xmax, ymax = image.shape
        xlim = kwargs.get('xlim', [0, xmax - 1])
        ylim = kwargs.get('ylim', [0, ymax - 1])

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

#        print(title)
#        print("Image median = {:10.3f}, stdev = {:10.3f}".format(vmed, vstd))
#        print("vmin, vmax = {:10.3f}{:10.3f}".format(vmin, vmax))

        stars = kwargs.get('stars', [])
        fig, ax = plt.subplots()
        xmin, xmax, ymin, ymax = xlim[0], xlim[1], ylim[0], ylim[1]
        fig = plt.imshow(img,
                         extent=(xmin - 0.5, xmax + 0.5, ymin - 0.5, ymax + 0.5),
                         interpolation='nearest', cmap='binary',  # cmap='hot'
                         vmin=vmin, vmax=vmax, origin='lower')
        cbar = plt.colorbar()
        cbar.set_label(units)
        for star in stars:
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.scatter([star[1]], [star[2]], lw=0, s=4, color='r', marker='o')
        plt.title(title, fontsize='medium')
        plt.show()
        return

    def histogram(self, img, title, **kwargs):
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

    def plot_eef_pair(self, r_eef_list, **kwargs):
        title = kwargs.get('title', '')

        plt.xlabel('log10(Radius / pixel)')
        plt.ylabel('Encircled energy fraction')
        plt.title(title)
        xtick_lin_vals = np.array([0.1, 1, 2, 5, 10, 20.0, 50.0, 100.0])
        xtick_vals = np.log10(xtick_lin_vals)
        plt.xticks(xtick_vals, xtick_lin_vals)
        plt.xlim(np.log10([0.1, 100.0]))

        colours = ['blue', 'red']
        scale = [1.000, 1.000]
        lws = ['solid', 'solid']
        for i, r_eef in enumerate(r_eef_list):
            radii, eef = r_eef
            y = eef * scale[i]
            x = np.log10(radii)
            colour, lw, ls = colours[i], 1.0, lws[i]
            plt.plot(x, y, color=colour, ls=ls, lw=lw)
        plt.show()
        return

    def plot_eef_list(self, radii, eefs, **kwargs):

        plot_average = kwargs.get('plot_average', False)
        r_sample = kwargs.get('r_sample', 1.9)
        r_max = kwargs.get('r_max', 20.0)
        lc_list = kwargs.get('lc_list', ['blue'])
        ls_list = kwargs.get('ls_list', ['solid'])
        lw_list = kwargs.get('lw_list', [1.5])

        title = kwargs.get('title', '')

        plt.xlabel('log10(Radius / pixel)')
        plt.ylabel('Encircled energy fraction')
        plt.title(title)

        xtick_lin_vals = np.array([0.5, 1, 2, 5, 10, r_max])
        xtick_vals = np.log10(xtick_lin_vals)
        plt.xticks(xtick_vals, xtick_lin_vals)
        plt.xlim(np.log10([0.5, r_max]))

        n_obs, n_profiles, n_radii = eefs.shape
        rl = np.log10(radii)
        for j in range(0, n_obs):
            lc = 'grey' if lc_list is None else lc_list[j]
            ls = 'solid' if ls_list is None else ls_list[j]
            lw = 1.5 if lw_list is None else lw_list[j]
            for i in range(0, n_profiles):
                plt.plot(rl, eefs[j, i], color=lc, ls=ls, lw=lw)
            eef_mean = None
            if plot_average:
                eef_mean = np.squeeze(np.mean(eefs, axis=0), axis=0)
                eef_std = np.squeeze(np.std(eefs, axis=0), axis=0)
                plt.plot(rl, eef_mean, color='blue', ls=ls, lw=1.5)
                ee_sample = np.interp(r_sample, radii, eef_mean)
                ee_sample_std = np.interp(r_sample, radii, eef_std)
                lrs = math.log10(r_sample)
                plt.scatter(lrs, ee_sample, color='blue', marker='o')
                text = "EE @ {:3.1f} pix = {:4.3f} +- {:4.3f}".format(r_sample, ee_sample, ee_sample_std)
                plt.text(lrs, 0.95*ee_sample, text, color='blue', va='top')

        plt.show()
        return eef_mean

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
