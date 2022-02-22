#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:23:01 2022

@author: awatkins

Unit tests for insert_fakes.py's two major classes.
"""
import galsim
import astropy
import numpy as np


def test_fakes_instance():
    '''
    Tests ImageBuilder class by creating an instance and verifying that the
    input parameters match those associated with the class instance.
    Also tests ImageBuilder.makeImage(), which runs by default.
    '''
    from fakes.insert_fakes import ImageBuilder

    dimX = 500
    dimY = 500
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {}
    noise = None

    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)

    assert im.dimX == dimX
    assert im.dimY == dimY
    assert im.raCen == raCen
    assert im.decCen == decCen
    assert im.polyDict == polyDict
    assert np.array_equal(im.image.array, np.zeros((dimX, dimY)))


def test_draw_instance():
    '''
    Tests DrawModels class by creating an instance and verifying that the
    input parameters match those associated with the class instance.
    '''
    from fakes.insert_fakes import DrawModels

    ra = 180
    dec = 45
    image = galsim.Image(10, 10)

    d = DrawModels(ra, dec, image)

    assert d.ra == ra
    assert d.dec == dec
    assert d.image == image


def test_createWcs():
    '''
    Tests that usage of ImageBuilder.createWcs() function generates sensible
    output.  Tests that coordinate transformations are correct.
    '''
    from fakes.insert_fakes import ImageBuilder

    dimX = 500
    dimY = 600
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {}
    noise = None

    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)

    im.createWcs(pxScale)
    cen = im.w.wcs_world2pix(raCen, decCen, 1)

    assert type(im.w) == astropy.wcs.wcs.WCS
    assert np.isclose(cen[0], dimX/2, 0.00001)
    assert np.isclose(cen[1], dimY/2, 0.00001)


def test_polyDict_noRandom():
    '''
    Tests the output of ImageBuilder.makePolyDict() without randomized
    displacement from the input values
    '''
    from fakes.insert_fakes import ImageBuilder
    dimX = 500
    dimY = 600
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {'c0_0': 1000,
                'c1_0': 0.1}
    noise = None

    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)

    im.makePolyDict()

    assert im.polyDict['c0_0'] == 1000
    assert im.polyDict['c1_0'] == 0.1
    assert im.polyDict['c1_1'] == 0


def test_polyDict_random():
    '''
    Tests the output of ImageBuilder.makePolyDict() with randomized
    displacement from the input values
    '''
    all_keys = ['c0_0', 'c0_1', 'c0_2',
                'c1_0', 'c1_1', 'c1_2',
                'c2_0', 'c2_1', 'c2_2']
    from fakes.insert_fakes import ImageBuilder
    dimX = 500
    dimY = 600
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {'c0_0': 1000,
                'c1_0': 0.1}
    polyDictKeep = {'c0_0': 1000,
                    'c1_0': 0.1}
    noise = None

    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)

    fD = 0.05
    random = True
    seed = 12345
    im.makePolyDict(fD, random, seed)

    rng = np.random.default_rng(seed)

    for key in all_keys:
        if key in polyDictKeep.keys():
            assert im.polyDict[key] == polyDictKeep[key] \
                + rng.normal(0, polyDictKeep[key] * fD)
        else:
            assert im.polyDict[key] == 0


def test_makeModelSky():
    '''
    Tests the output of ImageBuilder.makeModelSky() by creating comparing the
    Poisson noise in the model vs. another built the same way
    '''
    from fakes.insert_fakes import ImageBuilder
    dimX = 500
    dimY = 600
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {'c0_0': 1000,
                'c1_0': 0.1}
    noise = None
    seed = 12345
    polyDeg = 0

    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)
    im.makeModelSky(polyDeg, seed=seed)

    # Creating a noiseless sky to test against
    rng = np.random.default_rng(seed)
    sky = np.zeros((im.dimY, im.dimX)) + im.polyDict['c0_0']
    fish = rng.poisson(sky)
    fish_amp = sky - fish

    assert np.array_equal(im.sky-fish_amp, sky)


def test_ditheredCoordinates():
    '''
    Tests the function which creates arrays of dithered coordinates by
    checking that the coordinates do, in fact, land within ditherStep
    +/- tol of the input center.
    '''
    from fakes.insert_fakes import ImageBuilder
    dimX = 500
    dimY = 600
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {'c0_0': 1000,
                'c1_0': 0.1}
    noise = None
    ditherStep = 100
    tol = 0.1
    n = 500  # Need a large number of tests

    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)
    centers = im.ditheredCoordinates(ditherStep, tol, n)
    cens_deg = [galsim.CelestialCoord(i[0]*galsim.degrees, i[1]*galsim.degrees)
                for i in centers]
    cens_xy = [im.image.wcs.toImage(i) for i in cens_deg]
    for cen in cens_xy:
        assert (int(cen.x) <= dimX/2 + ditherStep + dimX*tol)
        assert (int(cen.x) >= dimX/2 - ditherStep - dimX*tol)
        assert (int(cen.y) <= dimY/2 + ditherStep + dimY*tol)
        assert (int(cen.y) >= dimY/2 - ditherStep - dimY*tol)


def test_convertCoords():
    '''
    Tests DrawModels.convertCoords()
    '''
    from fakes.insert_fakes import ImageBuilder
    from fakes.insert_fakes import DrawModels
    dimX = 500
    dimY = 500
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {'c0_0': 1000,
                'c1_0': 0.1}
    noise = None

    # Needs an image on which to draw the models
    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)

    # Position to use to test the function
    ra = 180
    dec = 45  # Just use the image center

    mod = DrawModels(ra, dec, im.image)
    image_pos, __, __ = mod.convertCoords()

    # Checks it's found the right pixel
    # The other two values are linked to image_pos
    assert np.isclose(image_pos.x, 250, 0.00001)
    assert np.isclose(image_pos.y, 250, 0.00001)


def test_drawPSF():
    '''
    Draws a fake star on the image at a given coordinate, then checks the
    total flux within a circular aperture at that coordinate against the
    expectation.  Tests location and flux simultaneously this way.
    '''
    def circular_aperture(cenX, cenY, maxR, image):
        x = np.arange(1, image.shape[1]+1)
        y = x.reshape(-1, 1)
        rad = np.sqrt((x - cenX)**2 + (y - cenY)**2)
        want = (rad <= maxR)
        return np.sum(image[want])

    from fakes.insert_fakes import ImageBuilder
    from fakes.insert_fakes import DrawModels
    dimX = 500
    dimY = 500
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {'c0_0': 1000,
                'c1_0': 0.1}
    noise = None
    beta = 3
    fwhm = 1
    mag = 10
    zp = 27
    totFlux = 10**(-0.4*(mag-zp))  # Magnitude to flux

    # Needs an image on which to draw the models
    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)

    # Where to draw the star
    star_x = 350
    star_y = 400
    coo = galsim.PositionD(star_x, star_y)  # Off-center somewhere
    coo_deg = im.image.wcs.toWorld(coo)
    ra = coo_deg.ra.deg
    dec = coo_deg.dec.deg

    star = DrawModels(ra, dec, im.image)
    star.drawPsf(beta, fwhm, mag, zp)

    circFlux = circular_aperture(star_x, star_y, 20*fwhm, star.image.array)
    # Doesn't perfectly preserve flux, so just checking it's within 1%
    assert np.isclose(circFlux/totFlux, 1, 0.01)


def test_drawSersic():
    '''
    Draws a model galaxy with a high inclination, then checks the total
    magnitude within an elliptical aperture of the same shape.  Tests
    simultaneously the galaxy position, flux, and orientation on the image.
    '''
    def elliptical_aperture(cenX, cenY, maxR, pa, ell, image):
        # NOTE: pa here is misaligned from how Galsim interprets it by 90 deg
        # Please excuse all the nonsense to correct for that.  I don't feel
        # like rewriting this to be cleaner.
        pa = 90 - pa
        x = np.arange(1, image.shape[1]+1)
        y = x.reshape(-1, 1)
        xEll = -(x - cenX)*np.cos(np.radians(pa+90)) + \
            (y - cenY)*np.sin(np.radians(pa+90))
        yEll = (x - cenX)*np.sin(np.radians(pa+90)) + \
            (y - cenY)*np.cos(np.radians(pa+90))
        rad = np.sqrt(xEll**2 + (yEll/(1-ell))**2)
        want = (rad <= maxR)
        return np.sum(image[want])

    from fakes.insert_fakes import ImageBuilder
    from fakes.insert_fakes import DrawModels
    dimX = 500
    dimY = 500
    raCen = 180
    decCen = 45
    pxScale = 1
    polyDict = {'c0_0': 1000,
                'c1_0': 0.1}
    noise = None
    beta = 3
    fwhm = 0
    mag = 10
    zp = 27
    totFlux = 10**(-0.4*(mag-zp))  # Magnitude to flux
    pa = 73
    ell = 0.8  # Very thin, to ensure aperture shape is fine
    rEff = 4
    n = 1
    stampWidth = 150

    # Needs an image on which to draw the models
    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)

    # Where to draw the galaxy
    gal_x = 350
    gal_y = 350
    coo = galsim.PositionD(gal_x, gal_y)  # Off-center somewhere
    coo_deg = im.image.wcs.toWorld(coo)
    ra = coo_deg.ra.deg
    dec = coo_deg.dec.deg

    gal = DrawModels(ra, dec, im.image)
    gal.drawSersic(n, rEff, 1-ell, pa, mag, beta, fwhm, stampWidth,
                   zp)

    ellFlux = elliptical_aperture(gal_x, gal_y, 20*rEff, pa, ell,
                                  gal.image.array)
    # Doesn't perfectly preserve flux, so just checking it's within 1%
    assert np.isclose(ellFlux/totFlux, 1, 0.01)
