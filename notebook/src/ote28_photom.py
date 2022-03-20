#!/usr/bin/env python
import numpy as np
import photutils
from photutils import psf, aperture_photometry
from photutils import Background2D, MedianBackground, CircularAperture, RectangularAperture
from astropy.stats import SigmaClip
from scipy.optimize import curve_fit
from ote28_globals import Ote28Globals as Globals


class Ote28Photom:

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

    def find_profiles(self, identifier, u_vals, v_coadd, image, stars, profile_list, **kwargs):
        group, code, axis, normal = identifier
        delta_u = u_vals[1] - u_vals[0]
        u_sample = kwargs.get('u_sample', delta_u)
        method = kwargs.get('method', 'enslitted')

        n_pts = len(u_vals)
        fmt = "{:10s}{:10s}{:10s}{:12s}"
        print("Extracting along {:s} profiles ".format(axis))
        print(fmt.format('Star ID', 'Col', 'Row', 'Flux'))
        for star in stars:
            ident, xcen, ycen, flux = star['id'], star['x_0'], star['y_0'], star['flux_0']
            print('{:>10d}{:10.3f}{:10.3f}{:12.1f}'.format(ident, xcen, ycen, flux))
            u_cen = round(xcen) if axis == 'row' else round(ycen)       # Nearest pixel to peak
            us = np.add(u_vals, u_cen)
            profile = np.zeros(n_pts)
            if method == 'cut':
                xc, yc, hu, hv = int(xcen), int(ycen), int(n_pts/2.0), int(v_coadd/2.0)
                if axis == 'row':
                    x1 = xc - hu
                    x2 = xc + hu
                    y1 = yc - hv
                    y2 = yc + hv
                    subim = image[y1:y2, x1:x2]
                    profile = np.sum(subim, 0)
                else:
                    x1 = xc - hv
                    x2 = xc + hv
                    y1 = yc - hu
                    y2 = yc + hu
                    subim = image[y1:y2, x1:x2]
                    profile = np.sum(subim, 1)
            if method == 'enslitted':
                for i in range(0, n_pts):
                    u = us[i]
                    k = 0.5     # image[0][0]'s centroid is at 0.5, 0.5
                    if axis == 'row':
                        ap_pos = u + k, ycen + k
                        aperture = RectangularAperture(ap_pos, w=u_sample, h=v_coadd)
                    else:
                        ap_pos = xcen + k, u + k
                        aperture = RectangularAperture(ap_pos, w=v_coadd, h=u_sample)
                    profile[i] = self.exact_rectangular(image, aperture)
            profile = self.normalise(profile, normal)
            fit, covar = curve_fit(Globals.Gauss, u_vals, profile)
            fit[1] = abs(fit[1])
            params = fit, covar
            profile_list.append((identifier, u_vals, profile, params))
        return profile_list

    def find_sources(self, img, bkg_sigma):
        """ Find all point sources in an image with flux above sf_threshold x bkg_sigma

        :param img:
        :param bkg_sigma:
        :return:
        """
        sf_threshhold = 3
        filter_fwhm = 2.00
        str = "Searching for stars above {:3.1f} sigma background noise ".format(sf_threshhold)
        str += " using a filter FWHM = {:5.2f}".format(filter_fwhm)
        print(str)
        dsf = photutils.DAOStarFinder(threshold=sf_threshhold * bkg_sigma, fwhm=filter_fwhm)
        found_stars = dsf(img)
        found_stars['xcentroid'].name = 'x_0'
        found_stars['ycentroid'].name = 'y_0'
        found_stars['flux'].name = 'flux_0'
        print("- {:d} sources found".format(len(found_stars)))
        return found_stars

    def select_bright_stars(self, stars, **kwargs):
        # Clip bounds, xmin, xmax, ymin, ymax
        def_field = 550, 950, 50, 950
        field = kwargs.get('field', def_field)
        threshold = kwargs.get('threshold', 70.0)

        x1, x2, y1, y2 = field
        str = "Selecting stars with flux > {:9.1f}'.format(threshold))"
        str += " and in region x={:5.0f}-{:5.0f}, y={:5.0f}-{:5.0f}".format(x1, x2, y1, y2)
        bstars = []
        for star in stars:
            x, y = star['x_0'], star['y_0']
            noclip = (x1 < x < x2) and (y1 < y < y2)
            if (star['flux_0'] > threshold) and noclip:
                bstars.append(star)
        return bstars

    def make_bkg(self, img):
        sig = 3.0
        sigma_clip = SigmaClip(sigma=sig)  # , iters=10)
        str = "Estimating background by finding median value in 50 x 50 boxes, "
        str += "averaged over 3x3 boxes and {:3.1f} sigma clipped ".format(sig)
        print(str)
        bkg_estimator = MedianBackground()
        bkg = Background2D(img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        bkg_sigma = bkg.background_rms_median
        print("Background median ={:10.1f} MJy/sterad".format(bkg.background_median))
        print("Background RMS    ={:10.1f} MJy/sterad".format(bkg_sigma))
        return bkg, bkg_sigma

    def find_eefs(self, radii, image, stars):
        n_radii = len(radii)
        n_stars = len(stars)
        eefs = np.zeros((n_stars, n_radii))
        for i, star in enumerate(stars):
            ident, x, y, flux = star['id'], star['x_0'], star['y_0'], star['flux_0']
            print('{:>5d}{:10.3f}{:10.3f}{:12.1f}'.format(ident, x, y, flux))
            centroid = x, y

            ee = np.zeros(n_radii)
            for j, r in enumerate(radii):
                aperture = CircularAperture(centroid, r)
                phot = aperture_photometry(image, aperture)
                ee[j] = phot['aperture_sum']

            eefs[i, :] = ee / ee[-1]  # Normalise to last ee point in profile
        return eefs

    def find_eeradii(self, eef_list, code_list, radref):
        """ Find the encircled energy at a specific radius (pixels) using linear
        interpolation
        """
        eerefs = np.zeros(len(code_list))
        j = 0
        for radii, eef in eef_list:
            iz = np.where(radii > radref)
            i = iz[0][0]
            factor = (eef[i] - eef[i-1]) / (radii[i] - radii[i-1])
            eeref = eef[i-1] + (radref - radii[i-1]) * factor
            print(j, code_list[j], radref, factor, eeref)
            eerefs[j] = eeref
            j += 1
        return eerefs

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
