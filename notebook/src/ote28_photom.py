#!/usr/bin/env python
import numpy as np
import photutils
from photutils import psf, aperture_photometry
from photutils import Background2D, MedianBackground, CircularAperture, RectangularAperture
from astropy.stats import SigmaClip
from scipy.optimize import curve_fit
from ote28_globals import Globals as Globals


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

    def scale_eeds(self, eevr, **kwargs):
        rscale = kwargs.get('rscale', None)
        eescale = kwargs.get('eescale', None)
        if eescale is not None:
            eevr_out = eevr[0], eevr[1], eescale * eevr[2], eescale * eevr[3]
        if rscale is not None:
            eevr_out = eevr[0], eevr[1] * rscale, eevr[2], eevr[3]
        return eevr_out

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
            ident, xcen, ycen, flux = star['id'], star['xcentroid'], star['ycentroid'], star['flux']
            print('{:>10d}{:10.3f}{:10.3f}{:12.1f}'.format(ident, xcen, ycen, flux))
            u_cen = round(xcen) if axis == 'row' else round(ycen)       # Nearest pixel to peak
            us = np.add(u_vals, u_cen)
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
                    u = us[i]
                    k = 0.5     # image[0][0]'s centroid is at 0.5, 0.5
                    if axis == 'row':
                        ap_pos = u + k, ycen + k
                        aperture = RectangularAperture(ap_pos, w=u_sample, h=v_coadd)
                    else:
                        ap_pos = xcen + k, u + k
                        aperture = RectangularAperture(ap_pos, w=v_coadd, h=u_sample)
                    p[i] = self.exact_rectangular(image, aperture)
            p = self.normalise(p, normal)
            fit, covar = curve_fit(Globals.Gauss, u_vals, p)
            fit[1] = abs(fit[1])
            params = fit, covar
            profile = identifier, u_vals, p, params
            profile_list.append(profile)
        return profile_list

    def find_sources(self, img, bkg_sigma, sd_threshhold, sd_fwhm):
        """ Find all point sources in an image with flux above bkg_sigma x sd_threshold
        where bkd_sigma is the background uncertainty per pixel, and sd_threshold is the
        factor by which the mean source flux per pixel when the image is convolved with
        a Gaussian of sd_fwhm.

        :param img:
        :param bkg_sigma:
        :return:
        """
        threshold = sd_threshhold * bkg_sigma
        dsf = photutils.DAOStarFinder(threshold=threshold, fwhm=sd_fwhm)
        found_stars = dsf(img)
        # The dsf flux parameter is calculated as the peak density in the convolved image divided
        # by the detection threshold. We scale this to give the integrated photometric signal 'phot_sig'
        flux_param = found_stars['flux']
        n_pixels = found_stars['npix']
        phot_sig = flux_param * bkg_sigma * n_pixels
        found_stars['phot_sig'] = phot_sig
#        print("- {:d} sources found".format(len(found_stars)))
        return found_stars

    def print_stars(self, stars):
        print("Found {:d} targets".format(len(stars)))
        fmt = "{:>10s}{:>10s}{:>10s}{:>12s}{:>12s}"
        print(fmt.format('ID', 'xcen', 'ycen', 'sig_aper', 'sig_pix'))
        fmt = "{:10d}{:10.3f}{:10.3f}{:12.1f}{:12.1f}"
        for star in stars:
            id = star['id']
            xcen, ycen = star['xcentroid'], star['ycentroid']
            sig_aper, phot_err = star['phot_sig'], 0.0
            n_pix = star['npix']
            sig_pix = sig_aper / n_pix
            print(fmt.format(id, xcen, ycen, sig_aper, sig_pix))
        return

    def select_bright_stars(self, stars, **kwargs):
        # Clip bounds, xmin, xmax, ymin, ymax
        default_field = 550, 950, 50, 950
        field = kwargs.get('field', default_field)
        threshold = kwargs.get('threshold', 70.0)

        x1, x2, y1, y2 = field
        str = "Selecting stars with flux > {:9.1f}".format(threshold)
        str += " in region x={:5.0f}-{:5.0f}, y={:5.0f}-{:5.0f}".format(x1, x2, y1, y2)
        print(str)
        bstars = []
        for star in stars:
            x, y = star['xcentroid'], star['ycentroid']
            noclip = (x1 < x < x2) and (y1 < y < y2)
            if (star['phot_sig'] > threshold) and noclip:
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

    def find_eefs(self, radii, observation):
        """ Calculate the EE curve of growth for a list of stars. """
        name, image, stars = observation
        n_radii = len(radii)
        n_stars = len(stars)
        ee_refs = np.zeros(n_stars)         # Calculate reference EE for each star
        ees = np.zeros((n_stars, n_radii))
        for i, star in enumerate(stars):
            ident, x, y, flux = star['id'], star['xcentroid'], star['ycentroid'], star['flux']
            print('{:>5d}{:10.3f}{:10.3f}{:12.1f}'.format(ident, x, y, flux))
            centroid = x, y

            ee = np.zeros(n_radii)
            for j, r in enumerate(radii):
                aperture = CircularAperture(centroid, r)
                phot = aperture_photometry(image, aperture)
                ee[j] = phot['aperture_sum']

            ee_norm = ee[-1]
            ee = ee / ee_norm  # Normalise to last ee point in profile
            ee_refs[i] = self.interpolate(radii, ee, Globals.ref_radius)
            ees[i, :] = ee
        return name, radii, ees, ee_refs

    def average_ee_list(self, eevr_list):
        """ Find mean and stdev of a list of EE(r) profiles.  It is assumed that
        all profiles are sampled on the same radius values
        """
        ee_list, ee_refs = [], []
        for eevr in eevr_list:
            name, radii, ees, ee_ref = eevr
            ee_list.extend(ees)
            ee_refs.extend(ee_ref)
        ee_array = np.array(ee_list)
        ee_ave = np.expand_dims(np.mean(ee_array, axis=0), axis=0)
        ee_ref_ave = np.mean(np.array(ee_refs))
        eevr_ave = name + 'ave', radii, ee_ave, ee_ref_ave
        ee_std = np.expand_dims(np.std(ee_array, axis=0), axis=0)
        ee_ref_std = np.std(np.array(ee_refs))
        eevr_std = name + 'err', radii, ee_std, ee_ref_std
        return eevr_ave, eevr_std

    def interpolate(self, radii, ee, ref_radius):
        """ Find the encircled energy at a specific radius (pixels) using linear
        interpolation
        """
        idxs = np.where(radii > ref_radius)
        idx = idxs[0][0]
        dr = radii[idx] - radii[idx-1]
        dr_ref = ref_radius - radii[idx - 1]
        dee_dr = (ee[idx] - ee[idx-1]) / dr
        eeref = ee[idx-1] + dr_ref * dee_dr
        return eeref


    def find_ee_at_radius(self, eevr_list, ref_radius):
        """ Find the encircled energy at a specific radius (pixels) using linear
        interpolation
        """
        eerefs = []
        for j, eevr in enumerate(eevr_list):
            eeref_star = []
            _, rad, ees, ee_ref = eevr
            idxs = np.where(rad > ref_radius)
            idx = idxs[0][0]
            dr = rad[idx] - rad[idx-1]
            dr_ref = ref_radius - rad[idx - 1]
            n_stars, n_samples = ees.shape
            for i in range(0, n_stars):
                ee = ees[i]
                dee_dr = (ee[idx] - ee[idx-1]) / dr
                eeref = ee[idx-1] + dr_ref * dee_dr
                eeref_star.append(eeref)
            eeref_mean = np.mean(np.array(eeref_star))
            print("Mean EE = {:8.3f}".format(eeref_mean))
            eerefs.append(eeref_star)
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
