#!/usr/bin/env python
import numpy as np
import photutils
from photutils import psf, aperture_photometry
from photutils import Background2D, MedianBackground, CircularAperture, RectangularAperture
from astropy.stats import SigmaClip
from scipy.optimize import curve_fit
from ote28_globals import Globals as Globals


class Profiler:

    def __init__(self):
        return

    def normalise(self, profile, normal):
        if normal != 'none':
            if normal == 'peak':
                peak_val = np.max(profile)
                profile /= peak_val
            if normal == 'sum':
                sum_val = np.sum(profile)
                profile /= sum_val
        return profile

    def combine_profiles(self, profiles):
        """ Combine multiple equal length profiles into a single x ordered profile. """
        nprofiles = len(profiles)
        nvals_profile = len(profiles[0][0])
        nvals = nvals_profile * nprofiles
        cx, cy = np.zeros(nvals), np.zeros(nvals)
        for p, profile in enumerate(profiles):
            identifier, x, y, params = profile
            fit, covar = params
            amp, sig, phase = fit
            i = nvals_profile * p
            cx[i: i+nvals_profile] = x[0: nvals_profile] - phase
            cy[i: i+nvals_profile] = y[0: nvals_profile]
        idx_sorted = np.argsort(cx)
        scx = cx[idx_sorted]
        scy = cy[idx_sorted]
        fit, covar = curve_fit(Globals.Gauss, scx, scy)
        params = fit, covar
        group, axis, fit_type, image_id = identifier
        com_identifier = group, axis, fit_type, 'combined'
        return com_identifier, scx, scy, params

    def find_profiles(self, identifier, ui_vals, v_coadd, image, stars, profile_list, **kwargs):
        program_id, obs_id, axis, normal = identifier
        delta_u = ui_vals[1] - ui_vals[0]
        verbose = kwargs.get('verbose', False)
        u_sample = kwargs.get('u_sample', delta_u)
        method = kwargs.get('method', 'enslitted')

        n_pts = len(ui_vals)
        fmt = "{:10s}{:10s}{:10s}{:12s}"
        print("Extracting along {:s} profiles ".format(axis))
        print(fmt.format('Star ID', 'Col', 'Row', 'Flux'))
        for star in stars:
            ident, xcen, ycen, flux = star['id'], star['xcentroid'], star['ycentroid'], star['flux']
            print('{:>10d}{:10.3f}{:10.3f}{:12.1f}'.format(ident, xcen, ycen, flux))
            u_cen = round(ycen)
            phase = ycen - u_cen
            if axis == 'row':
                u_cen = round(xcen)
                phase = xcen - u_cen
#            us = np.add(ui_vals, u_cen)
            p = np.zeros(n_pts)     # Profile Y values
            if method == 'cut':
                xc, yc, hu, hv = int(xcen), int(ycen), int(n_pts/2.0), int(v_coadd/2.0)
                if axis == 'row':
                    x1 = xc - hu
                    x2 = xc + hu
                    y1 = yc - hv
                    y2 = yc + hv
                    subim = image[y1:y2, x1:x2]
                    p = np.sum(subim, 0)
                else:
                    x1 = xc - hv
                    x2 = xc + hv
                    y1 = yc - hu
                    y2 = yc + hu
                    subim = image[y1:y2, x1:x2]
                    p = np.sum(subim, 1)
            if method == 'enslitted':
                for i in range(0, n_pts):
                    u = ui_vals[i] + u_cen
                    k = 0.5     # Pixel image[0][0]'s centre is at index 0.5, 0.5
                    if axis == 'row':
                        ap_pos = u + k, ycen + k
                        aperture = RectangularAperture(ap_pos, w=u_sample, h=v_coadd)
                    else:
                        ap_pos = xcen + k, u + k
                        aperture = RectangularAperture(ap_pos, w=v_coadd, h=u_sample)
                    p[i] = self.exact_rectangular(image, aperture)
            p = self.normalise(p, normal)
            fit, covar = curve_fit(Globals.Gauss, ui_vals, p)
            amp, fwhm, u_line_centre = fit
            uf_vals = ui_vals - u_line_centre
            fit[1] = abs(fit[1])
            params = star, fit, covar
            profile = identifier, ui_vals, uf_vals, p, params
            profile_list.append(profile)
        return profile_list

    def print_profile_list(self, direction, profile_list, best_fit):
        print()
        first_profile = True
        for profile in profile_list:
            identifier, ui_vals, uf_vals, p, params = profile
            star, fit, covar = params
            program_id, obs_id, axis, normal = identifier
            if first_profile:
                fmt = "Program {:s} - along {:s} direction)"
                print(fmt.format(program_id, direction))
                fmt = "{:>15s},{:>6s},{:>15s},{:>12s},{:>12s}"
                print(fmt.format('Observation', 'Star', 'Phot-Peak', 'Fit-Amp', 'Fit-FWHM'))
                fmt = "{:>15s},{:6d},{:15.1f},{:12.3f},{:12.3f}"
                first_profile = False
            print(fmt.format(obs_id, star['id'], star['phot_sig'], fit[0], fit[1]))
        fmt = "{:>51s},{:12.3f} +-{:7.3f}"
        print(fmt.format('Best fit FWHM for all lines', best_fit[0], best_fit[1]))
        return

    def find_profile_best_fit(self, profile_list):
        """ Calculate the best fit FWHM and its error for a list of profiles.
        """
        n_profiles = len(profile_list)
        fwhms = np.zeros(n_profiles)
        for j, profile in enumerate(profile_list):
            identifier, xi, xf, y, params = profile
            group, image_id, axis, fit_type = identifier
            star, fit, covar = params
            amp, fwhm, phase = fit
            print("{:10s}{:10s}{:10.2f}{:10.3f}{:10.3f}".format(group, image_id, amp, fwhm, phase))
            fwhms[j] = fwhm
        best_fit = np.mean(fwhms), np.std(fwhms)
        return best_fit, fwhms

    def exact_rectangular(self, image, aperture):
        cen = aperture.positions        #[0]
        w = aperture.w
        h = aperture.h
        x1 = cen[0] - w / 2.0
        x2 = x1 + w
        y1 = cen[1] - h / 2.0
        y2 = y1 + h
        c1, c2, r1, r2 = int(x1), int(x2), int(y1), int(y2)
        nr = r2 - r1 + 1       # Number of rows in subarray
        nc = c2 - c1 + 1
        wts = np.ones((nr, nc))
        im = image[r1:r1+nr, c1:c1+nc]

        fc1 = 1. - (x1 - c1)
        fc2 = x2 - c2
        if nc == 1:
            fc = fc1 + fc2 - 1.
            wts[:,0] *= fc
        else:
            wts[:,0] *= fc1
            wts[:,nc-1] *= fc2

        fr1 = 1. - (y1 - r1)
        fr2 = y2 - r2
        if nr == 1:
            fr = fr1 + fr2 - 1.
            wts[0,:] *= fr
        else:
            wts[0,:] *= fr1
            wts[nr-1,:] *= fr2

        wtim = np.multiply(im, wts)
        f = np.sum(wtim)
        return f
