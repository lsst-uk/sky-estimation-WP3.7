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
