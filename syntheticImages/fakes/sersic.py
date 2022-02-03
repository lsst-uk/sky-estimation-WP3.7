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
        gammaincinv(2*n, frac) : `float`
            Value of bn for given n and light fraction
    '''
    return gammaincinv(2*n, frac)


def getMuEff(mag, rEff, n):
    '''
    Derives effective surface brightnesses for a given
    set of model parameters
        Parameters
        ----------
        mag : `float`
            Model magnitude
        rEff : `float`
            Model effective radius in arcsec
        n : `float`
            Model Sersic index

        Returns
        -------
        muEff : `float`
            Surface brightness at the effective radius
        muEffAv : `float`
            Mean surface brightness within 1Reff
    '''
    fn = (n*np.exp(bnn(n, 0.5))*gamma(2*n))/(bnn(n, 0.5)**(2*n))
    muEffAv = mag + 2.5*np.log10(2*np.pi*rEff**2)
    muEff = muEffAv + 2.5*np.log10(fn)

    return muEff, muEffAv


def getMu0(mag, rEff, n, magZp=21.0967):
    '''
    Derives central surface brightness for a given
    set of model parameters

        Parameters
        ----------
        mag : `float`
            Model magnitude
        rEff : `float`
            Model effective radius in arcsec
        n : `float`
            Model Sersic index
        magZp : `float`
            Photometric zeropoint of mag
            Default is for the S4G AB magnitudes

        Returns
        -------
        mu0 : `numpy.ndarray`
            Central surface brightness in units of
            mag/arcsec^2
    '''
    b = bnn(n, 0.5)
    lTot = 10**(-0.4*(mag-magZp))
    i0 = (lTot*(b**(2*n))) \
         / (gamma(2*n)*2*n*np.pi*rEff**2)
    mu0 = -2.5*np.log10(i0) + magZp

    return mu0


def getSersicRadProf(mag, rEff, n, maxR, pxScale):
    '''
    Derives surface brightness profile for given parameters

        Parameters
        ----------
        mag : `float`
            Model magnitude
        rEff : `float`
            Model effective radius in arcsec
        n : `float`
            Model Sersic index
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
    muEff, __ = getMuEff(mag, rEff, n)
    bn = bnn(n, 0.5)

    rad = np.arange(0, maxR+1, 1)*pxScale
    radPart = (rad/rEff)**(1/n) - 1
    muR = muEff + ((2.5*bn)/np.log(10))*radPart

    return rad, muR


def sbLimWidth(mag, n, rEff, axRat, sbLim, pxScale=0.168):
    '''
    Derives an isophotal radius for a projected Sersic profile, which can be
    used to determine an appropriate stamp width

        Parameters
        ----------
        mag : `float'
            Object magnitude, to be converted into counts
        n : `float`
            Sersic index (useful range is 0.5 -- 6)
        rEff : `float'
            Half-light radius, in pixels
        axRat : `float`
            Projected axis ratio (values between 0 and 1)
        sbLim : `float`
            Desired surface brightness at which to truncate the model
        pxScale : `float`
            Arcsec per pixel, for unit conversions
        Returns
        -------
        rIsoPx : `int`
            Width of the stamp that limits the model to sbLim in surface
            brightness, in pixels

    NOTE: current implementation is a bit crude, as it assumes the model is
    face-on.  So this will overestimate the stamp size for inclined models.
    '''
    bn = bnn(n, 0.5)
    rEffArcsec = rEff * pxScale
    muEff, __ = getMuEff(mag, rEffArcsec, n)
    _a = np.log(10)/(2.5*bn)
    _b = muEff + 2.5*np.log10(axRat)
    rIso = rEff*((sbLim - _b)*_a + 1)**n

    if len(mag) == 1:
        rIsoPx = int(np.round(rIso/pxScale, 0))
    else:
        rIsoPx = np.array([int(np.round(i, 0)) for i in rIso])

    return rIsoPx
