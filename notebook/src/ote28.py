import os
import glob
import sys
#
import scipy.ndimage
import numpy as np

from ote28_plot import Ote28Plot
from ote28_photom import Ote28Photom
from ote28_globals import Globals

from matplotlib import style, pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import figure
from matplotlib.backends.backend_pdf import PdfPages
#import chart_studio.plotly as py
from mpl_toolkits.mplot3d import Axes3D
#from bokeh.plotting import figure, show, output_file
#
from astropy.coordinates import match_coordinates_sky
from astropy.io import fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.visualization import LogStretch, LinearStretch, PercentileInterval, ManualInterval
from astropy.io import ascii
import astropy.units as u
from astropy import wcs
from astropy.modeling import models
#
# --------------------------------------------------------------------
# input/output parameters --------------------------------------------
# --------------------------------------------------------------------

# set the location of the simulation files **REQUIRED**


# set the filter for example analysis. Options are:
# F560W, F770W, F1000W, F1130W, F1280W, F1500W, F1800W, F2100W, F2550W
ima_filter = 'F560W'

# set filters to analyze and produce output files. Options are:
# F560W, F770W, F1000W, F1130W, F1280W, F1500W, F1800W, F2100W, F2550W
ima_filters_out = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W']

# output file directory (will be created if it doesn't exist)
output_dir = 'output/'

# --------------------------------------------------------------------
# global analysis parameters -----------------------------------------
# --------------------------------------------------------------------

# minimum flux density of catalog stars to be used (modify as needed). Currently set to a low value.
flux_min = 1.0e-10

filter_fwhm = {"F560W", 2.00}
filter='F560W'

plot = Ote28Plot()
photom = Ote28Photom()
globals = Globals()

rcParams['figure.figsize'] = [8., 5.]
plt.rcParams.update({'font.size': 12})

process_cv3 = False
if process_cv3:
    dataset = 'CV3'
    cv3_dir = '../data/cv3_data/'
    cv3_prog = 'MIRM042MRF-'
    cv3_list = [('F0', '0-6026132013_1_493_SE_2016-01-26T13h27m18', '0B-6026131649_1_493_SE_2016-01-26T13h23m38'),
                ('M2', 'M2-6026131344_1_493_SE_2016-01-26T13h21m48', 'M2B-6026131028_1_493_SE_2016-01-26T13h17m18'),
                ('M4', 'M4-6026130723_1_493_SE_2016-01-26T13h15m28', 'M4B-6026130412_1_493_SE_2016-01-26T13h10m58'),
                ('M6', 'M6-6026130058_1_493_SE_2016-01-26T13h09m08', 'M6B-6026125734_1_493_SE_2016-01-26T13h07m28'),
                ('M8', 'M8-6026125408_1_493_SE_2016-01-26T13h01m18', 'M8B-6026125042_1_493_SE_2016-01-26T12h58m08'),
                ('P2', 'P2-6026133128_1_493_SE_2016-01-26T13h37m48', 'P2B-6026132331_1_493_SE_2016-01-26T13h35m28'),
                ('P4', 'P4-6026133742_1_493_SE_2016-01-26T13h44m18', 'P4B-6026133431_1_493_SE_2016-01-26T13h41m38'),
                ('P6', 'P6-6026134415_1_493_SE_2016-01-26T13h50m58', 'P6B-6026134104_1_493_SE_2016-01-26T13h47m58'),
                ('P8', 'P8-6026135056_1_493_SE_2016-01-26T13h58m28', 'P8B-6026134735_1_493_SE_2016-01-26T13h58m38')]
    cv3_ls_dict = {'F':'solid', 'M':'dotted', 'P':'dashed'}
    cv3_foc_sign = {'F':0.0, 'M':-1.0, 'P':1.0}
    cv3_lc_dict = {'0':'black', '2':'blue', '4':'green', '6':'brown', '8':'red'}
    cv3_ls_list, cv3_lc_list, cv3_lw_list = [], [], []
    foci = []
    for cv3 in cv3_list:
        ls_tag = cv3[0][0:1]
        cv3_ls_list.append(cv3_ls_dict[ls_tag])
        lc_tag = cv3[0][1:2]
        focus = float(lc_tag) * cv3_foc_sign[ls_tag]
        foci.append(focus)
        cv3_lc_list.append(cv3_lc_dict[lc_tag])
        lw = 1.5 if lc_tag == '0' else 1.0
        cv3_lw_list.append(lw)

    cv3_post = '_LVL2.fits'
    n_cv3 = len(cv3_list)
    row_profile_list, col_profile_list, eevr_list = [], [], []
    code_list = []

    # Define sampling arrays, can be different for CV3 and OTE-28.2 data sets
    u_sample = 1.0      # Sample psf once per pixel to avoid steps
    u_start = 0.0       # Offset from centroid
    u_radius = 16.0     # Maximum radial size of aperture
    v_coadd = 15.0      # Number of pixels to coadd orthogonal to profile

    u_vals = np.arange(u_start - u_radius, u_start + u_radius, u_sample)

    r_start, r_max, r_sample = 0.1, 16.0, 0.1
    radii = np.arange(r_start, r_max, r_sample)

    sd_threshhold = 10       # Star detection threshold, multiple of background noise level
    sd_fwhm = 2.00           # Star detector spatial filter FWHM in pixels
    str = "Searching for stars above {:3.1f} sigma background noise ".format(sd_threshhold)
    str += " using a filter FWHM = {:5.2f}".format(sd_fwhm)
    print(str)
    make_smoothed_image = True
    for obs_tag in cv3_list:    #[0:1]:
        code = obs_tag[0]
        code_list.append(code)
        cv3_img_file = cv3_dir + cv3_prog + obs_tag[1] + cv3_post
        cv3_bgd_file = cv3_dir + cv3_prog + obs_tag[2] + cv3_post
        print('Analysing image {:s}'.format(cv3_prog + obs_tag[1]))

        hdu_list = fits.open(cv3_img_file)
        hdu = hdu_list[0]
        img, hdr = hdu.data[0], hdu.header
        hdu_list = fits.open(cv3_bgd_file)
        hdu = hdu_list[0]
        bgd, hdr = hdu.data[0], hdu.header
        image = np.subtract(img, bgd)

        out_folder = '../output/'
        path = out_folder + 'cv3_ref_' + code
        fits.writeto(path, image, header=None, overwrite=True)
        title = 'CV3 point source ' + code
        plot.display(image, title,
                     vmin=-0.1, vmax=1000.0, log=False,
                     xlim=[507, 515], ylim=[507, 515],
                     units='DN/s', skip=True)
        im_smooth = scipy.ndimage.gaussian_filter(image, 2.0)
        if make_smoothed_image:
            im_smooth_stack = np.array(im_smooth)
            make_smoothed_image = False
        else:
            im_smooth_stack += im_smooth
        plot.display(im_smooth, title,
                     vmin=-0.1, vmax=1.0, log=False,
                     xlim=[440, 580], ylim=[440, 580],
                     units='DN/s', skip=True)

        # Find profiles for all bright stars and add to list
        field = 480, 540, 480, 540      # x0, x1, y0, y1

        bkg_sigma = np.nanstd(image[field[0]:field[1], field[2]:field[3]])
        stars = photom.find_sources(image, bkg_sigma, sd_threshhold, sd_fwhm)
        photom.print_stars(stars)
        stars = photom.select_bright_stars(stars, threshold=1.0, field=field)  # Filter bright stars >100 pix from edge of field

        row_identifier = 'CV3', code, 'row', 'sum'  # group, image_id, axis, fit_type
        row_profile_list = photom.find_profiles(row_identifier, u_vals, v_coadd, image, stars, row_profile_list)

        col_identifier = 'CV3', code, 'col', 'sum'  # group, axis, fit_type, image_id
        col_profile_list = photom.find_profiles(col_identifier, u_vals, v_coadd, image, stars, col_profile_list,
                                                axis='col', normal='sum')
        eevr = photom.find_eefs(radii, (code, image, stars))
        eevr_list.append(eevr)

    im_smooth_stack = im_smooth_stack / n_cv3
    plot.display(im_smooth_stack, "CV3 images, smoothed and stacked",
                 vmin=-0.1, vmax=1.0, log=False,
                 xlim=[440, 580], ylim=[440, 580],
                 units='DN/s', skip=True)

#    ee_refs = photom.find_ee_at_radius(eevr_list, Globals.ref_radius)
    for eevr in eevr_list:
        name, _, _, ee_refs = eevr
        fmt = "{:s} - EE values at r= {:6.2f}, "
        ee_text = fmt.format(name, Globals.ref_radius)
        for ee_ref in ee_refs:
            ee_text += "{:8.3f}, ".format(ee_ref)
        print(ee_text)

    plot.plot_cv3_eevfocus(foci, eevr_list)
    plot.plot_cv3_fwhmvfocus(foci, row_profile_list, 'CV3 Along Row FWHM v focus')
    plot.plot_cv3_fwhmvfocus(foci, col_profile_list, 'CV3 Along Column FWHM v focus')

    plot.plot_eef_list(eevr_list,
                       title='CV3 EED at 0,2,4..8 mm defocus',
                       show_ref_rad=True,
                       lc_list=cv3_lc_list, ls_list=cv3_ls_list, lw_list=cv3_lw_list)

    plot.plot_profile_list(row_profile_list,
                           fit_profile=row_profile_list[0],
                           xlim=[-5.0, 5.0],              # Plot limits
                           normalise=True,
                           title='CV3 row profiles at 0,2,4..8 mm defocus',
                           lc_list=cv3_lc_list, ls_list=cv3_ls_list, lw_list=cv3_lw_list)


    plot.plot_profile_list(col_profile_list,
                           fit_profile=col_profile_list[0],
                           xlim=[-5.0, 5.0],              # Plot limits
                           normalise=True,
                           title='CV3 column profiles at 0,2,4..8 mm defocus',
                           lc_list=cv3_lc_list, ls_list=cv3_ls_list, lw_list=cv3_lw_list)

process_ote28_eed = True
process_ote28_fwhm = True
process_ote28 = process_ote28_eed or process_ote28_fwhm

if process_ote28:
    datasets = ['ote28_sim_mar21', 'ote28_sim_sep21', 'flight']
    dataset = datasets[1]
    print("Analysing {:s} dataset - reading image files..".format(dataset))
    image_list = []  # List of all sub-images in dataset

    simulation_dir = '../data/' + dataset
    obs_pre = '/jw01464005001_01101_000'
    obs_post = '_mirimage_cal.fits'
    obs_tags = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    rcParams['figure.figsize'] = [8., 5.]
    plt.rcParams.update({'font.size': 11})

    xlim = [500, 900]
    ylim = [100, 900]

    u_sample = 1.0          # Sample psf once per pixel to avoid steps
    u_start = 0.0           # Offset from centroid
    u_radius = 20.0         # Maximum radial size of aperture
    u_vals = np.arange(u_start - u_radius, u_start + u_radius, u_sample)
    v_coadd = 15.0          # Number of pixels to coadd orthogonal to profile

    r_start, r_max, r_sample = 0.1, 16.0, 0.1
    radii = np.arange(r_start, r_max, r_sample)

    all_col_fwhms, all_row_fwhms, all_eeds = [], [], []
    sim_row_profile_list, sim_col_profile_list, = [], []
    n_bright = 0
    observations = []
    for obs_tag in obs_tags[0:3]:
        file = simulation_dir + obs_pre + obs_tag + obs_post
        print(file)
        hdu_list = fits.open(file)
        hdu = hdu_list[1]
        img, hdr = hdu_list[1].data, hdu_list[1].header
        dq = hdu_list[2].data
        img[np.where(dq == 262660)] = np.nan
        img[np.where(dq == 262656)] = np.nan
        img[np.where(dq == 2359812)] = np.nan
        img[np.where(dq == 2359808)] = np.nan
        title = file.split('/')[2] + ' - obs ' + obs_tag
        plot.display(img, title, vmin=0.0, vmax=2.0, skip=True)
        image_list.append(img)

    image_stack = np.array(image_list)
    bgd_stack = np.nanmedian(image_stack, axis=0)
    title = file.split('/')[2] + ' - stacked bgd'
    plot.display(bgd_stack, title, vmin=0.0, vmax=2.0, skip=False)
    bgd_sample_region = bgd_stack[450:550, 450:550]
    plot.histogram(bgd_sample_region, title, bins=50)
    bkg_sigma = np.nanstd(bgd_sample_region)
    bkg_mean = np.nanmean(bgd_sample_region)
    fmt = "Mean stacked background = {:10.3f} +- {:10.3f}"
    print(fmt.format(bkg_mean, bkg_sigma))
    sd_threshold = 10.0         # N sigma above background
    sd_fwhm = 2.0
    str = "Searching for stars above {:3.1f} sigma background noise,".format(sd_threshold)
    str += " using a Gaussian kernel, fwhm = {:5.2f}".format(sd_fwhm)
    print(str)
    for i, image in enumerate(image_list):
        obs = image - bgd_stack
        obs_tag = obs_tags[i]

        print("Processing observation {:s}".format(obs_tag))
        stars = photom.find_sources(obs, bkg_sigma, sd_threshold, sd_fwhm)
        print("Found {:d} stars".format(len(stars)))

        stars = photom.select_bright_stars(stars)  # Filter bright stars >100 pix from edge of field
        photom.print_stars(stars)
        observation = obs_tag, obs, stars
        observations.append(observation)
        title = 'obs ' + obs_tag
        plot.display(obs, title, vmin=0.0, vmax=2.0, stars=stars, skip=True)

if process_ote28_eed:
    all_eefs, all_stars = None, []
    # Read in a reference image which can be used to normalise the measured EE(r) profile.
    ref_image_list = [('CDP PSF', 'MIRI_FM_MIRIMAGE_F560W_PSF_07.02.00.fits', 0.8),
                      ('CDP PSF OOF', 'MIRI_FM_MIRIMAGE_F560W_PSF-OOF_07.00.00.fits', 0.9)]
    lc_refs = ['green', 'red']
    sd_threshold = 10.0  # N sigma above background
    sd_fwhm = 2.0
    str = "Searching for stars above {:3.1f} sigma background noise,".format(sd_threshold)
    str += " using a Gaussian kernel, fwhm = {:5.2f}".format(sd_fwhm)
    bkg_sigma = .02

    eevr_refs = []
    for ref_image_tag, ref_image_name, eescale in ref_image_list:
        print("Processing reference image {:s}".format(ref_image_name))
        ref_image_file = '../data/' + 'cruciform_sim/' + ref_image_name
        hdu_list = fits.open(ref_image_file)
        ref_image = hdu_list[1].data[0]

        ref_stars = photom.find_sources(ref_image, bkg_sigma, sd_threshold, sd_fwhm)
        photom.print_stars(ref_stars)
        title = "Reference image ({:s})".format(ref_image_tag)
        plot.display(ref_image, title, vmin=0.0, vmax=0.001, skip=True)
        ref_radii = np.arange(r_start, 240.0, 0.1)
        eevr_ref = photom.find_eefs(ref_radii, (ref_image_tag, ref_image, ref_stars))
        eevr_ref = photom.scale_eeds(eevr_ref, rscale=0.25)
        eevr_ref = photom.scale_eeds(eevr_ref, eescale=eescale)
        eevr_refs.append(eevr_ref)
    title = "Reference (CDP PSF) EE v r - F560W"
    plot.plot_eef_list(eevr_refs,
                       show_ref_rad=True, title=title,
                       r_max=50.0, lc_list=lc_refs)

    eevr_all = []
    lc_all, lw_all = [], []
    print("Analysing encircled energy in OTE-28 images")
    eescale = 0.8
    print("- EE(r) will be scaled by {:5.2f}".format(eescale))
    print("{:12s},{:10s},{:14s}".format('Obs', 'n_stars', 'EE(r=1.9 pix)'))
    for observation in observations:
#        obs_tag, image, stars = observation
        n_stars = len(stars)
        eevr = photom.find_eefs(radii, observation)
        eevr = photom.scale_eeds(eevr, eescale=eescale)
        eevr_all.append(eevr)
        lc_all.append('grey')
        lw_all.append(0.5)

    eevr_ote, eevr_ote_std = photom.average_ee_list(eevr_all)
    name_ote, radii, ee_ote, ee_ote_ref = eevr_ote
    eevr_ote_ref = photom.interpolate(radii, ee_ote[0], Globals.ref_radius)
    name_ote_std, radii, ee_ote_std, ee_ote_std_ref = eevr_ote_std
    ee_ote_ref_std = photom.interpolate(radii, ee_ote_std[0], Globals.ref_radius)
    fmt = "OTE-28 EE(r = 1.9 pixels ={:10.3f} +-{:10.3f}"
    print(fmt.format(ee_ote_ref, ee_ote_ref_std))

    eevr_refs.append(eevr_ote)
    lc_refs.append('black')
    lw_refs = [1.0, 1.0, 1.5]
    plot.plot_eef_list(eevr_refs,
                       show_ref_rad=True,
                       title='OTE-28 EE(r)',
                       r_max=50.0,
                       ref_eed=(Globals.ref_radius, ee_ote_ref, ee_ote_ref_std),
                       lc_list=lc_refs, lw_list=lw_refs)

if process_ote28_fwhm:
    for observation in observations:
        obs_tag, image, bstars = observation
        n_bright += len(bstars)
        plot.display(image, "OTE-28.2 simulated", fmin=-0.01, fmax=10.0,
                     stars=bstars, xlim=xlim, ylim=ylim, skip=False)
        print(np.nanmin(image), np.nanmax(image))

        # Find profiles for all bright stars and add to list
        row_identifier = 'OTE28', obs_tag, 'row', 'sum'  # group, image_id, axis, fit_type
        sim_row_profile_list = photom.find_profiles(row_identifier,
                                                    u_vals, v_coadd, image,
                                                    bstars, sim_row_profile_list,
                                                    method='enslitted')
        col_identifier = 'OTE28', obs_tag, 'col', 'sum'  # group, image_id, axis, fit_type
        sim_col_profile_list = photom.find_profiles(col_identifier,
                                                    u_vals, v_coadd, image,
                                                    bstars, sim_col_profile_list,
                                                    method='enslitted')

        n_profiles = len(sim_row_profile_list)
        fwhms = np.zeros(n_profiles)
        for j, profile in enumerate(sim_row_profile_list):
            identifier, x, y, params = profile
            group, image_id, axis, fit_type = identifier
            fit, covar = params
            amp, fwhm, phase = fit
            print("{:10s}{:10s}{:10.2f}{:10.3f}{:10.3f}".format(group, image_id, amp, fwhm, phase))
            fwhms[j] = fwhm
            all_row_fwhms.append(fwhm)
        mean_fwhm = np.mean(fwhms)
        sig_fwhm = np.std(fwhms)
        print("{:10s}{:10s}{:10.2f}{:10.3f}{:10.3f}".format(group, 'Row ave.', 0.00, mean_fwhm, sig_fwhm))
        print()
        for j, profile in enumerate(sim_col_profile_list):
            identifier, x, y, params = profile
            group, image_id, axis, fit_type = identifier
            fit, covar = params
            amp, fwhm, phase = fit
            print("{:10s}{:10s}{:10.2f}{:10.3f}{:10.3f}".format(group, image_id, amp, fwhm, phase))
            fwhms[j] = fwhm
            all_col_fwhms.append(fwhm)
#            plot.plot_profile(profile, obs_list, axis, plot_fit=True, is_cv3=False)
        mean_fwhm = np.mean(fwhms)
        sig_fwhm = np.std(fwhms)
        print("{:10s}{:10s}{:10.2f}{:10.3f}{:10.3f}".format(group, 'Col ave.', 0.00, mean_fwhm, sig_fwhm))

    col_fwhms = np.array(all_col_fwhms)
    mean_all_col_fwhm = np.mean(col_fwhms)
    sig_all_col_fwhm = np.std(col_fwhms)
    row_fwhms = np.array(all_row_fwhms)
    mean_all_row_fwhm = np.mean(row_fwhms)
    sig_all_row_fwhm = np.std(row_fwhms)
    print()
    print("Dataset = {:s}".format(dataset))
    fmt = "Along {:s} FWHM averaged over all observations = {:10.3f} +- {:10.3f}"
    print(fmt.format('column', mean_all_col_fwhm, sig_all_col_fwhm))
    print(fmt.format('row', mean_all_row_fwhm, sig_all_row_fwhm))

process_crux = False
if process_crux:

    simulation_dir = '../data/cruciform_sim'
    obs_pre = '/o0_6_0_'
    obs_post = '.fits'
    obs_tags = ['psf', 'psf_oof']
    rcParams['figure.figsize'] = [8., 5.]
    plt.rcParams.update({'font.size': 11})

    xlim = [500, 800]
    ylim = [500, 800]

    u_sample = 1.0          # Sample psf once per pixel to avoid steps
    u_start = 0.0           # Offset from centroid
    u_radius = 20.0         # Maximum radial size of aperture
    u_vals = np.arange(u_start - u_radius, u_start + u_radius, u_sample)
    v_coadd = 15.0          # Number of pixels to coadd orthogonal to profile

    r_start, r_max, r_sample = 0.1, 16.0, 0.1
    radii = np.arange(r_start, r_max, r_sample)

    sim_row_profile_list, sim_col_profile_list, sim_eef_list = [], [], []
    n_bright = 0
    for obs_tag in obs_tags[0:2]:
        file = simulation_dir + obs_pre + obs_tag + obs_post
        print(file)
        hdu_list = fits.open(file)
        hdu = hdu_list[1]
        img, hdr = hdu_list[1].data, hdu_list[1].header
        dq = hdu_list[2].data
        img[np.where(dq == 262660)] = np.nan
        img[np.where(dq == 262656)] = np.nan
        img[np.where(dq == 2359812)] = np.nan
        img[np.where(dq == 2359808)] = np.nan
        plot.display(img, file, fmin=-0.01, fmax=0.2, xlim=xlim, ylim=ylim)
        bkg_obj, bkg_sigma = photom.make_bkg(img)  # Make background image
        bkg = bkg_obj.background
        plot.display(bkg, 'Background')
        image = img - bkg
        stars = photom.find_sources(image, bkg_sigma)
        bstars = photom.select_bright_stars(stars,
                                            threshold=2000.,
                                            field=(600, 760, 400, 560))  # Filter bright stars >100 pix from edge of field
        n_bright += len(bstars)
    #    plot.display(image, "OTE-28.2 simulated", fmin=-0.01, fmax=0.03,
    #                 stars=bstars, xlim=xlim, ylim=ylim, skip=False)
    #    print(np.nanmin(image), np.nanmax(image))

        # Find profiles for all bright stars and add to list
        row_identifier = 'Sim', obs_tag, 'row', 'sum'  # group, image_id, axis, fit_type
        sim_row_profile_list = photom.find_profiles(row_identifier, u_vals, v_coadd,
                                                    image, bstars, sim_row_profile_list)
        col_identifier = 'Sim', obs_tag, 'col', 'sum'  # group, image_id, axis, fit_type
        sim_col_profile_list = photom.find_profiles(col_identifier, u_vals, v_coadd,
                                                    image, bstars, sim_col_profile_list)

        n_profiles = len(sim_row_profile_list)
        fwhms = np.zeros(n_profiles)
        fmt = "{:10s}{:10s}{:10s}{:10s}{:10s}"
        print(fmt.format('group', 'image_id', 'amplitude', 'fwhm', 'phase'))
        for i, profile in enumerate(sim_row_profile_list):
            identifier, x, y, params = profile
            group, image_id, axis, fit_type = identifier
            fit, covar = params
            amp, fwhm, phase = fit
            print("{:10s}{:10s}{:10.2f}{:10.3f}{:10.3f}".format(group, image_id, amp, fwhm, phase))
            fwhms[i] = fwhm
            plot.plot_profile(profile, obs_tags, axis, plot_fit=True, is_cv3=False)
        mean_fwhm = np.mean(fwhms)
        sig_fwhm = np.std(fwhms)
        print("{:10s}{:10s}{:10.2f}{:10.3f}{:10.3f}".format(group, 'Row ave.', 0.00, mean_fwhm, sig_fwhm))
        print()
        for i, profile in enumerate(sim_col_profile_list):
            identifier, x, y, params = profile
            group, image_id, axis, fit_type = identifier
            fit, covar = params
            amp, fwhm, phase = fit
            print("{:10s}{:10s}{:10.2f}{:10.3f}{:10.3f}".format(group, image_id, amp, fwhm, phase))
            fwhms[i] = fwhm
#            plot.plot_profile(profile, obs_list, axis, plot_fit=True, is_cv3=False)
        mean_fwhm = np.mean(fwhms)
        sig_fwhm = np.std(fwhms)
        print("{:10s}{:10s}{:10.2f}{:10.3f}{:10.3f}".format(group, 'Col ave.', 0.00, mean_fwhm, sig_fwhm))

print('Done')






