#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:58:29 2022

@author: awatkins

Unit tests for the fakes.reduction_pipeline module functions.
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def test_hdulist():
    '''
    Verifies that reduction_pipeline.makeHduList() does, in fact, make an
    HDUlist, with a header and data, by checking the input against the
    output.
    '''
    from fakes.reduction_pipeline import makeHduList
    # Blank image
    im = np.zeros((100, 100))
    # Header
    head = fits.Header({'TEST': 1})
    hduList = makeHduList(im, head)

    assert np.array_equal(im, hduList[0].data)
    assert hduList[0].header['TEST'] == 1


def test_blankImage():
    '''
    Tests the reduction_pipeline.makeBlankImage() function, to ensure the
    image is blank and the WCS info in the header is implemented correctly.
    '''
    from fakes.reduction_pipeline import makeBlankImage
    ra = 180
    dec = 45
    size = 100
    blankHdu = makeBlankImage(ra, dec, size)

    want = blankHdu[0].data != 0
    assert len(want[want]) == 0
    assert blankHdu[0].header['CRVAL1'] == 180
    assert blankHdu[0].header['CRVAL2'] == 45
    assert np.isclose(blankHdu[0].header['PC1_1'], -0.168/3600, 0.00001)


def test_binImage():
    '''
    Tests that the output binned image has the right value in one pixel,
    that it has the right dimensions, and that the WCS info has been scaled
    appropriately as well.
    Also checks that
    '''
    from fakes.reduction_pipeline import binImage
    from fakes.reduction_pipeline import makeBlankImage
    mn = 100
    std = 3
    size = 1000
    block = 10  # Bin factor
    xTest = 405
    yTest = 405
    rng = np.random.default_rng(12345)  # For adding noise to an image

    imHdu = makeBlankImage(180, 45, size, 1.5)
    w = WCS(imHdu[0].header)
    coo = w.pixel_to_world(xTest, yTest)
    raTest = coo.ra.value
    decTest = coo.dec.value
    noise = rng.normal(mn, std, size=(size, size))
    imHdu[0].data += noise  # Image with just random noise
    imSec = imHdu[0].data[:block, :block]  # One pixel in the binned image

    bnHdu = binImage(imHdu, block)
    bnW = WCS(bnHdu[0].header)
    bnCoo = bnW.pixel_to_world(xTest//block, yTest//block)
    bnRa = bnCoo.ra.value
    bnDec = bnCoo.dec.value

    assert bnHdu[0].data.shape == (size/block, size/block)
    assert np.isclose(np.median(imSec), bnHdu[0].data[0, 0], 0.00001)
    assert np.isclose(raTest, bnRa, 0.00042)  # 1.5 arcsec, in degrees
    assert np.isclose(decTest, bnDec, 0.00042)


def test_maskToLimit():
    '''
    Tests that all pixels in an image with a given value and lower are
    masked out, and that all the rest are not masked.
    '''
    from fakes.reduction_pipeline import makeHduList
    from fakes.reduction_pipeline import maskToLimit
    val = 10
    zp = 27
    scale = 0.168
    # Masking everything brighter than 5 cts, because floating point nonsense
    sb = -2.5*np.log10(val-5) + zp + 2.5*np.log10(scale**2)

    im = makeHduList(np.zeros((100, 100)))
    im[0].data[40:60, 40:60] = val
    mask = np.zeros((20, 20)) + 1
    want = im[0].data == 0
    maskImage = maskToLimit(im, sb, zp, scale)

    assert np.array_equal(maskImage[0].data[40:60, 40:60], mask)
    assert np.array_equal(want, maskImage[0].data == 0)


def test_legendreSkySub():
    '''
    Tests that the residuals of a sky-subtracted image have the correct
    Poissonian standard deviation, assuming a strong residual structure will
    bias this.
    We do it this way because the actual fit parameters are apparently highly
    unstable compared to the inputs even for solutions that reproduce the
    shape of the input very well, possibly due to the forced inclusion of
    high-order terms.
    '''
    import warnings
    from astropy.utils.exceptions import AstropyUserWarning
    import fakes.reduction_pipeline as rp
    from fakes.insert_fakes import ImageBuilder

    # Creating image with a model polynomial sky
    dimX = 500
    dimY = 500
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {'c0_0': 1000, 'c1_0': 0.2, 'c0_1': 2e-4, 'c1_1': 5e-8}
    noise = None
    polyDeg = 1

    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)
    im.makeModelSky(polyDeg)
    hduList = rp.makeHduList(im.image.array+im.sky, header=im.w.to_header())

    # Suppress the warning about linearity
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='Model is linear in parameters',
                                category=AstropyUserWarning)
        modelSky = rp.legendreSkySub(polyDeg, hduList,
                                     bnFac=9, maskVal=np.nan, full=False)

    residuals = im.sky - modelSky

    # Do the residuals have the right standard deviation?
    # If too high, a plane is still present.
    assert np.isclose(np.std(residuals), np.sqrt(1000), atol=1)


def test_coadd():
    '''
    Tests that images with specific offsets from a central coordinate are
    appropriately coadded into a blank reference image.
    '''
    from fakes.insert_fakes import ImageBuilder
    from fakes.reduction_pipeline import makeHduList
    from fakes.reduction_pipeline import coaddImages

    def mk_im(val):
        im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict)
        hdu = makeHduList(im.image.array+val, header=im.w.to_header())
        return hdu

    dimX = 250
    dimY = 500
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {}
    size = 1000

    # Creating three flat images to coadd
    im1 = mk_im(5)
    im2 = mk_im(10)
    im3 = mk_im(3)
    imDict = {0: im1,
              1: im2,
              2: im3}

    # How much in x and y each is offset from the reference image center
    # This works because coaddition works in pixel space, no celestial
    offsets = [(-375, 250), (-250, 0), (375, -250)]

    coaddHdu = coaddImages(raCen, decCen, size, imDict, offsets, pxScale)
    coaddIm = coaddHdu[0].data

    # Checking finally that the pixel values are correct in the correct areas
    assert np.all(coaddIm[0: 500, 750:] == 3)
    assert np.all(coaddIm[500: 750, 125: 250] == np.median([10, 5]))


def test_coaddSubtraction():
    '''
    Tests that coadd is correctly aligned with and subtracted from a single
    fake exposure.
    '''
    from fakes.insert_fakes import ImageBuilder
    from fakes.reduction_pipeline import makeHduList
    from fakes.reduction_pipeline import coaddImages
    from fakes.reduction_pipeline import coaddSubtraction

    dimX = 250
    dimY = 125
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {}
    size = 500

    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict)
    hdu = makeHduList(im.image.array+5, header=im.w.to_header())
    imDict = {0: hdu}
    offsets = [(-125, 125)]
    coaddHdu = coaddImages(raCen, decCen, size, imDict, offsets, pxScale)

    diffImage = coaddSubtraction(hdu, offsets[0], coaddHdu)
    assert ~np.all(diffImage[0].data)
