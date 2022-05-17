#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:37:37 2022

@author: awatkins

Unit test for fakes.sersic module functions.
"""
import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    'test, expected',
    [(4, 7.669),
     (1, 1.678)]
    )
def test_bnn(test, expected):
    '''
    Ensures that some common-case Sersic index values return the correct
    value of b_n.
    '''
    from fakes.sersic import bnn
    frac = 0.5
    npt.assert_almost_equal(bnn(test, frac), expected, decimal=3)


@pytest.mark.parametrize(
    'mag, rad, expected',
    [(10, 2, 13.501),
     (15, 5, 20.490),
     (20, 10, 26.995)]
    )
def test_getMuEffAv(mag, rad, expected):
    '''
    Checks for a range of by-hand calculations that correct surface brightness
    is returned for three sets of magnitude and half-light radius.
    '''
    from fakes.sersic import getMuEffAv
    npt.assert_almost_equal(getMuEffAv(mag, rad), expected, decimal=3)


@pytest.mark.parametrize(
    'mag, rad, n, expected',
    [(10, 2, 1, 14.199),
     (15, 5, 2, 21.526),
     (20, 10, 4, 28.388)]
    )
def test_getMuEff(mag, rad, n, expected):
    '''
    Checks for a range of by-hand calculations that the correct effective
    surface brightness is returned for a range of magnitude, half-light radius,
    and Sersic index.
    '''
    from fakes.sersic import getMuEff
    npt.assert_almost_equal(getMuEff(mag, rad, n), expected, decimal=3)


@pytest.mark.parametrize(
    'mag, rad, n, expected',
    [(10, 2, 1, 12.377),
     (15, 5, 2, 17.539),
     (20, 10, 4, 20.062)]
    )
def test_getMu0(mag, rad, n, expected):
    '''
    Checks for a range of by-hand calculations that the correct central
    surface brightness is returned for a range of magnitude, half-light radius,
    and Sersic index.
    '''
    from fakes.sersic import getMu0
    npt.assert_almost_equal(getMu0(mag, rad, n), expected, decimal=3)


@pytest.mark.parametrize(
    'mag, rad, n, expected',
    [(10, 2, 1, 24.8),
     (15, 5, 2, 95.9),
     (20, 10, 4, 103.6)]
    )
def test_sbLimWidth(mag, rad, n, expected):
    '''
    Checks isophotal radii for a set of test cases, using axial ratio 1 and
    for an isophote of 35 mag/arcsec^2.
    '''
    from fakes.sersic import sbLimWidth
    axRat = 1
    muLim = 35
    # Losing decimal precision due to rounding propagation....
    npt.assert_almost_equal(sbLimWidth(mag, rad, n, axRat, muLim), expected,
                            decimal=1)
