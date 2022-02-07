#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:23:01 2022

@author: awatkins

Unit tests for insert_fakes.py
"""
import galsim
import astropy
import numpy as np
import pytest


def test_fakes_instance():
    '''
    Tests ImageBuilder class by creating an instance and verifying that the
    input parameters match those associated with the class instance.
    '''
    from fakes.insert_fakes import ImageBuilder

    dimX = 500
    dimY = 500
    raCen = 180
    decCen = 90
    pxScale = 1
    polyDict = {}
    noise = None

    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)

    assert im.dimX == dimX
    assert im.dimY == dimY
    assert im.raCen == raCen
    assert im.decCen == decCen
    assert im.polyDict == polyDict


def test_draw_instance():
    '''
    Tests DrawModels class by creating an instance and verifying that the
    input parameters match those associated with the class instance.
    '''
    from fakes.insert_fakes import DrawModels

    ra = 180
    dec = 90
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
    decCen = 90
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
    decCen = 90
    pxScale = 1
    polyDict = {'c0_0': 1000,
                'c1_0': 0.1}
    noise = None

    im = ImageBuilder(dimX, dimY, raCen, decCen, pxScale, polyDict, noise)

    fracDisplacement = 0.05
    im.makePolyDict(fracDisplacement)

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
    decCen = 90
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

