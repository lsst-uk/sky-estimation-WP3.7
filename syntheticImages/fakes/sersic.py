import numpy as np
from scipy.special import gamma, gammaincinv


def bnn(n, frac):
    '''
    Derives Sersic bn parameter for given light fraction
        Parameters
        ----------
        n : `float`
            Sersic index
        frac : `float`
            Light fraction, e.g. 0.5

        Returns
        -------
        bN : `float`
            Value of bn for given n and light fraction
    '''
    assert n > 0, 'Sersic index must be positive and non-zero'
    assert (frac > 0) & (frac <= 1), \
        'Fraction must be between 0 and 1'

    bN = gammaincinv(2*n, frac)

    return bN


def getMuEffAv(mag, rEff):
    '''
    Derives average surface brightness within 1 effective radius
        Parameters
        ----------
        mag : `float`
            Total magnitude
        rEff : `float`
            Half-light (effective) radius in arcseconds

        Returns
        -------
        muEffAv : `float`
            Average surface brightness within rEff
    '''
    assert rEff > 0, 'Half-light radius must be positive and non-zero'
    muEffAv = mag + 2.5*np.log10(2*np.pi*rEff**2)

    return muEffAv


def getMuEff(mag, rEff, n):
    '''
    Derives surface brightness at the effective radius
        Parameters
        ----------
        mag : `float`
            Total magnitude
        rEff : `float`
            Half-light (effective) radius in arcseconds
        n : `float`
            Sersic index

        Returns
        -------
        muEff : `float`
            Surface brightness at the effective radius
    '''
    assert rEff > 0, 'Half-light radius must be positive and non-zero'
    assert n > 0, 'Sersic index must be positive and non-zero'
    fn = (n*np.exp(bnn(n, 0.5))*gamma(2*n))/(bnn(n, 0.5)**(2*n))
    muEffAv = getMuEffAv(mag, rEff)
    muEff = muEffAv + 2.5*np.log10(fn)

    return muEff


def getMu0(mag, rEff, n):
    '''
    Derives central surface brightness

        Parameters
        ----------
        mag : `float`
            Total magnitude
        rEff : `float`
            Half-light (effective) radius in arcseconds
        n : `float`
            Sersic index

        Returns
        -------
        mu0 : `numpy.ndarray`
            Central surface brightness in units of mag/arcsec^2
    '''
    assert rEff > 0, 'Half-light radius must be positive and non-zero'
    assert n > 0, 'Sersic index must be positive and non-zero'
    b = bnn(n, 0.5)
    lTot = 10**(-0.4*mag)
    i0 = (lTot*(b**(2*n))) / (gamma(2*n)*2*n*np.pi*rEff**2)
    mu0 = -2.5*np.log10(i0)

    return mu0


def getSersicRadProf(mag, rEff, n, maxR, pxScale):
    '''
    Derives surface brightness profile for given parameters

        Parameters
        ----------
        mag : `float`
            Total magnitude
        rEff : `float`
            Half-light (effective) radius in arcseconds
        n : `float`
            Sersic index
        maxR : `float`
            Maximum radius of model profile, in pixels
        pxScale : `float`
            Pixel scale to convert to arcseconds

        Returns
        -------
        rad : `numpy.array`
            Radius array in arcseconds
        muR : `numpy.array`
            Surface brightness array out to maxR
    '''
    assert rEff > 0, 'Half-light radius must be positive and non-zero'
    assert n > 0, 'Sersic index must be positive and non-zero'
    assert maxR > 0, 'Maximum radius must be positive and non-zero'
    assert pxScale > 0, 'Pixel scale must be positive and non-zero'

    muEff = getMuEff(mag, rEff, n)
    bn = bnn(n, 0.5)

    rad = np.arange(0, maxR+1, 1)*pxScale
    radPart = (rad/rEff)**(1/n) - 1
    muR = muEff + ((2.5*bn)/np.log(10))*radPart

    return rad, muR


def sbLimWidth(mag, rEff, n, axRat, sbLim):
    '''
    Derives an isophotal radius for a projected Sersic profile, which can be
    used to determine an appropriate stamp width

        Parameters
        ----------
        mag : `float'
            Total magnitude
        rEff : `float'
            Half-light (effective) radius in arcseconds
        n : `float`
            Sersic index
        axRat : `float`
            Projected axis ratio (values between 0 and 1)
        sbLim : `float`
            Desired surface brightness at which to truncate the model
        Returns
        -------
        rIso : `int`
            Width of the stamp that limits the model to sbLim in surface
            brightness, in arcseconds

    NOTE: current implementation is a bit crude, as it assumes the model PA
    is multiples of 90 degrees.  Will underestimate if it's between those.
    '''
    assert rEff > 0, 'Half-light radius must be positive and non-zero'
    assert n > 0, 'Sersic index must be positive and non-zero'
    assert (axRat > 0) & (axRat <= 1), \
        'Axial ratio must have a value between 0 and 1'
    bn = bnn(n, 0.5)
    muEff = getMuEff(mag, rEff, n)  # Face-on
    _a = np.log(10)/(2.5*bn)
    _b = muEff + 2.5*np.log10(axRat)  # Projecting the profile
    rIso = rEff*((sbLim - _b)*_a + 1)**n

    return rIso
