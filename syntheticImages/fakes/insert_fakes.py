#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Software to generate fake images with model stars and galaxies as well as
noisy model skies.
'''
import math
import galsim
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from astroquery.gama import GAMA
from astroquery.sdss import SDSS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.modeling.models import Legendre2D
import numpy as np


class ImageBuilder():
    '''
    Class for creating image bases
    '''

    def __init__(self, dimX, dimY, raCen, decCen, pxScale, polyDict,
                 noise=None):
        '''
        Initializes baseline image
            Parameters
            ----------
            dimX : `int'
                Image size along x-axis
            dimY : `int'
                Image size along y-axis
            raCen : `float'
                Reference RA, in decimal degrees
            decCen : `float'
                Reference DEC, in decimal degrees
            pxScale : `float'
                Pixel scale in arcsec/pixel
            polyDict : `dict`
                Dictionary containing sky polynomial coefficients
                Keys are designated 'cN_M' where N and M are integers
            noise : `galsim.noise'
                If noise is needed, can add it to the image with this
        '''
        assert (dimX > 0) & (type(dimX) == int), \
            "dimX must be a positive integer."
        assert (dimY > 0) & (type(dimY) == int), \
            "dimY must be a positive integer."
        assert (raCen >= 0) & (raCen <= 360), \
            "Invalid RA: use decimal degrees."
        assert (decCen >= -90) & (decCen <= 90), \
            "Invalid Dec: use decimal degrees."
        assert (pxScale > 0), \
            "Pixel scale must be positive."
        assert (type(polyDict) == dict), \
            "polyDict must be a dictionary."
        assert (type(noise) == galsim.noise.CCDNoise) | (noise == None), \
            "noise must be galsim.noise.CCDNoise or None."
        self.dimX = dimX
        self.dimY = dimY
        self.raCen = raCen
        self.decCen = decCen
        self.createWcs(pxScale)
        self.makeImage(noise)
        self.polyDict = polyDict

    def createWcs(self, pxScale):
        '''
        Creates a gnomonic Astropy WCS header with the desired tangent point
        and pixel scale
            Parameters
            ----------
            pxScale : `float'
                Pixel scale in arcsec/pixel

            Returns
            -------
            self.w : `astropy.wcs.WCS'
                Astropy WCS object with desired parameters
        NOTE: no distortions, no rotation from standard projection
        '''
        pxScale /= 3600.  # Convert to degrees
        wcs_dict = {'CTYPE1': 'RA---TAN',
                    'CTYPE2': 'DEC--TAN',
                    'CRVAL1': self.raCen,
                    'CRVAL2': self.decCen,
                    'CRPIX1': self.dimX//2,  # Center of image
                    'CRPIX2': self.dimY//2,  # Center of image
                    'CD1_1': -pxScale,
                    'CD1_2': 0.0,
                    'CD2_1': 0.0,
                    'CD2_2': pxScale}
        self.w = WCS(wcs_dict)

    def makePolyDict(self, fracDisplacement, random=True):
        '''
        Generates sky polynomial coefficients, up to order 2
            Parameters
            ----------
            fracDisplacement : `float`
                Fraction of initial coefficient value to use in generating
                random offsets.  E.G., if mean is 2000, a value of 0.05 will
                adjust this to 2000 + N(0, 2000*0.05)
            random : `bool`
                Chooses whether to add random displacements to coefficients

            Returns
            -------
            None; updates Class polyDict attribute to include all potential
            terms, up to c2_2

        '''
        all_keys = ['c0_0', 'c0_1', 'c0_2',
                    'c1_0', 'c1_1', 'c1_2',
                    'c2_0', 'c2_1', 'c2_2']
        if random:
            # Randomly adjust sky coefficients
            rng = np.random.default_rng()
            for key in all_keys:
                if key not in self.polyDict:
                    self.polyDict[key] = 0
                else:
                    self.polyDict[key] += rng.normal(0,
                                                     self.polyDict[key]
                                                     * fracDisplacement)
        else:
            for key in all_keys:
                if key not in self.polyDict:
                    self.polyDict[key] = 0

    def makeModelSky(self, polyDeg, fracDisplacement=0.0, random=False):
        '''
        Creates randomized polynomial sky patterns up to second order to add to
        images
            Parameters
            ----------
            polyDeg : `int`
                Degree of the sky polynomial (0, 1, or 2)
            fracDisplacement : `float`
                Fraction of initial coefficient value to use in generating
                random offsets.  E.G., if mean is 2000, a value of 0.05 will
                adjust this to 2000 + N(0, 2000*0.05)
            random : `bool`
                Chooses whether to add random displacements to coefficients

            Returns
            -------
            self.sky : `numpy.array'
                Image of polynomial sky pattern
        '''
        X, Y = np.meshgrid(np.arange(1, self.dimX+1),
                           np.arange(1, self.dimY+1))

        self.makePolyDict(fracDisplacement, random)

        if polyDeg == 0:
            m = Legendre2D(polyDeg, polyDeg,
                           c0_0=self.polyDict['c0_0'])
        elif polyDeg == 1:
            m = Legendre2D(polyDeg, polyDeg,
                           c0_0=self.polyDict['c0_0'],
                           c0_1=self.polyDict['c0_1'],
                           c1_0=self.polyDict['c1_0'],
                           c1_1=self.polyDict['c1_1'])
        elif polyDeg == 2:
            m = Legendre2D(polyDeg, polyDeg,
                           c0_0=self.polyDict['c0_0'],
                           c0_1=self.polyDict['c0_1'],
                           c0_2=self.polyDict['c0_2'],
                           c1_0=self.polyDict['c1_0'],
                           c1_1=self.polyDict['c1_1'],
                           c1_2=self.polyDict['c1_2'],
                           c2_0=self.polyDict['c2_0'],
                           c2_1=self.polyDict['c2_1'],
                           c2_2=self.polyDict['c2_2'])

        else:
            print('Polynomial order must be between 0 and 2!')

        self.sky = m(X, Y)  # Will throw an error here for negative values!
        # Adding Poisson noise
        rng = np.random.default_rng()
        fish = rng.poisson(self.sky)
        fish_amp = self.sky - fish
        self.sky += fish_amp

    def ditheredCoordinates(self, ditherStep=100, tol=0.1, n=36):
        '''
        Calculates the centers of semi-randomly dithered exposures centered at
        (self.raCen, self.decCen), using a 9-point dither pattern as a baseline
            Parameters
            ----------
            ditherStep : `int`
                Size of dither, in pixels
            tol : `float'
                Maximum amplitude in random offsets to dither points
            n : `int'
                Number of dithered exposures

            Returns
            -------
            centers : `list'
                List of image center coordinate pairs (tuples) in (RA, Dec),
                in decimal degrees
        '''
        rng = np.random.default_rng()

        # 9-point dither pattern baseline, top-left to bottom-right
        cenx = self.image.array.shape[1]//2
        ceny = self.image.array.shape[0]//2
        gridx = [cenx-ditherStep, cenx, cenx+ditherStep,
                 cenx-ditherStep, cenx, cenx+ditherStep,
                 cenx-ditherStep, cenx, cenx+ditherStep]
        gridy = [ceny+ditherStep, cenx+ditherStep, cenx+ditherStep,
                 cenx, cenx, cenx,
                 cenx-ditherStep, cenx-ditherStep, cenx-ditherStep]
        pattern = list(zip(gridx, gridy))
        ra_cens = []
        dec_cens = []
        i = 0
        while i < n:
            for coo in pattern:
                # Maximum tol% random offset in x and y from each position
                # MAKE THIS AN INTEGER!  Then no need for partial pixel
                # translations.
                offsetx = rng.uniform(-1, 1) * self.image.array.shape[1] * tol
                offsety = rng.uniform(-1, 1) * self.image.array.shape[0] * tol
                image_pos = galsim.PositionD(coo[0]+offsetx, coo[1]+offsety)
                world_pos = self.image.wcs.toWorld(image_pos)
                ra_cens.append(world_pos.ra.deg)
                dec_cens.append(world_pos.dec.deg)
                i += 1
                if i > n-1:
                    break

        centers = list(zip(ra_cens, dec_cens))
        return centers

    def makeImage(self, noise=None):
        '''
        Creates blank image of the given dimensions with the provided WCS
            Parameters
            ----------
            w : `astropy.wcs.WCS'
                WCS object containing astrometric solution desired for output
                image (becomes attribute of image)
            noise : `galsim.noise'
                If noise is needed, can add it to the image with this

            Returns
            -------
            self.image : `galsim.Image'
                Blank Galsim image with WCS
        '''
        w = galsim.AstropyWCS(wcs=self.w)
        self.image = galsim.Image(self.dimX, self.dimY, wcs=w)

        if noise is not None:
            self.image.addNoise(noise)


class DrawModels(ImageBuilder):
    '''
    Class for populating images with models
    Inherits ImageBuilder() class
    '''

    def __init__(self, ra, dec, image):
        '''
        Initializes model coordinates
            Parameters
            ----------
            self.ra : `float'
                Right ascension coordinate, in decimal degrees
            self.dec : `float'
                Declination coordinate, in decimal degrees
            self.image : `galsim.Image'
                Galsim image object on which to draw the models
        '''
        assert (ra >= 0) & (ra <= 360), \
            "Invalid RA: use decimal degrees."
        assert (dec >= -90) & (dec <= 90), \
            "Invalid Dec: use decimal degrees."
        assert (type(image) == galsim.image.Image), \
            "image must be galsim.image.Image class."
        self.ra = ra
        self.dec = dec
        self.image = image

    def convertCoords(self):
        '''
        Converts right ascension and declination coordinates into position
        shifts in galsim.Image object with WCS
            Returns
            -------
            image_pos : `galsim.PositionD'
                Pixel coordinates of (ra,dec) in input image array
            image_posi : `galsim.PositionI'
                Integer floor coordinates of stamp center pixel
            offset : `galsim.PositionD'
                Fractional pixel offset for alignment to pixel center
        '''
        ra = self.ra * galsim.degrees
        dec = self.dec * galsim.degrees
        world_pos = galsim.CelestialCoord(ra, dec)
        image_pos = self.image.wcs.toImage(world_pos)
        image_posi = galsim.PositionI(int(math.floor(image_pos.x)),
                                      int(math.floor(image_pos.y)))

        x_cen = image_posi.x + 0.5
        y_cen = image_posi.y + 0.5

        dx = image_pos.x - x_cen
        dy = image_pos.y - y_cen
        offset = galsim.PositionD(dx, dy)

        return image_pos, image_posi, offset

    def drawPsf(self, beta, fwhm, mag, magZp=27.0):
        '''
        Draws a model star at the given coordinates with the given parameters
        onto the given image.
            Parameters
            ----------
            beta : `float'
                Moffat beta value
            fwhm : `float'
                PSF full-width at half-maximum
            mag : `float'
                Star magnitude, to be converted into counts
            magZp : `float'
                Magnitude zeropoint, to convert from mag to counts

            Returns
            -------
            Nothing--draws PSF model onto image at specified coordinate
            If coordinate is not on the image, it passes
        '''
        flux = 10**(-0.4*(mag - magZp))
        psf = galsim.Moffat(beta=beta, fwhm=fwhm, flux=flux)

        self.image_pos, self.image_posi, self.offset = self.convertCoords()

        try:
            stamp = psf.drawImage(wcs=self.image.wcs.local(self.image_pos),
                                  offset=self.offset)
            # Must be an integer, so the initial offset is required
            stamp.setCenter(self.image_posi.x, self.image_posi.y)
            bounds = stamp.bounds & self.image.bounds
            self.image[bounds] += stamp[bounds]

        except galsim.errors.GalSimBoundsError:
            pass

    def drawFullPsf(self, mag, psfPath, magZp=27.0):
        '''
        Uses a larger PSF model FITS image to create star models with wings
        PSF model must be normalized, such that sum(flux) == 1
            Parameters
            ----------
            mag : `float`
                Model star magnitude
            psfPath : `string`
                Full path to normalized full PSF model, including filename
            magZp : TYPE, optional
                Magnitude to flux zeropoint. The default is 27.0.

            Returns
            -------
            Nothing--adds the PSF image to the input image at the appropriate
            coordinates
        NOTE: the PSF FWHM is fixed to that of the model image you feed this.
        While this can be broadened via convolution, it cannot easily be
        narrowed.  E.G., the Montes et al. (2021) HSC PSF model has
        FWHM = 1.07".

        ALSO: at the moment, partial pixel shift is not in place, so image
        coaddition will not work quite right.
        '''
        psf = fits.getdata(psfPath)
        flux = 10**(-0.4*(mag - magZp))
        psf *= flux  # Scaling to proper magnitude
        self.image_pos, self.image_posi, self.offset = self.convertCoords()
        try:
            psf = galsim.Image(psf, wcs=self.image.wcs.local(self.image_pos))
            stamp = psf
            # Must be an integer, so the initial offset is required
            stamp.setCenter(self.image_posi.x, self.image_posi.y)
            bounds = stamp.bounds & self.image.bounds
            self.image[bounds] += stamp[bounds]

        except galsim.errors.GalSimBoundsError:
            pass

    def drawSersic(self, n, rEff, axRat, pa, mag, beta, fwhm, stampWidth,
                   magZp=27.0):
        '''
        Draws a model Sersic profile at the given coordinates with the given
        parameters onto the given image.
            Parameters
            ----------
            n : `float`
                Sersic index (useful range is 0.5 -- 6)
            rEff : `float'
                Half-light radius, in pixels
            axRat : `float`
                Projected axis ratio (values between 0 and 1)
            pa : `float`
                Position angle in degrees
            mag : `float'
                Object magnitude, to be converted into counts
            beta : `float`
                PSF beta value for convolution
            fwhm : `float`
                PSF FWHM for convolution; if 0, skips convolution
            stampWidth : `int`
                Width of the model bounding box, in pixels
            magZp : `float`
                Magnitude zeropoint, to convert from mag to counts

            Returns
            -------
            Nothing--draws Sersic model onto image at specified coordinate
            If coordinate is not on the image, it passes
        NOTE: useful for drawing small objects, but for large ones use
        makeSersicStamp() function.  Takes a long while to draw the stamp.
        '''
        flux = 10**(-0.4*(mag - magZp))
        sersic = galsim.Sersic(n, rEff)
        sersic = sersic.shear(q=axRat, beta=((90 - pa)*galsim.degrees))
        sersic = sersic.withFlux(flux)
        bounds = galsim.BoundsI(1, stampWidth, 1, stampWidth)
        # Convolving w/PSF
        if fwhm != 0:
            psf = galsim.Moffat(beta=beta, fwhm=fwhm)
            sersic = galsim.Convolve([sersic, psf])

        image_pos, image_posi, offset = self.convertCoords()
        try:
            gal_im = sersic.drawImage(wcs=self.image.wcs.local(image_pos),
                                      offset=offset, bounds=bounds)
        except (galsim.errors.GalSimFFTSizeError, MemoryError):
            print('------------------')
            print('Memory error!')
            print('n: %.1f, mag: %.2f, Reff: %.2f' % (n, mag, rEff))
            return None

        try:
            gal_im.setCenter(image_posi.x, image_posi.y)
            bounds = gal_im.bounds & self.image.bounds
            self.image[bounds] += gal_im[bounds]
        except galsim.errors.GalSimBoundsError:
            pass


def get2MASSCatalogue(raCen, decCen, width):
    '''
    Downloads a star catalogue from 2MASS, to be used to generate
    fake stars for a particular field
        Parameters
        ----------
        raCen : `float'
            Right ascension coordinate of search box center, in decimal degrees
        decCen : `float'
            Declination coordinate of search box center, in decimal degrees
        width : `float'
            Width of search box, in decimal degrees

        Returns
        -------
        cat : `astropy.table.Table'
            Table with astroquery results for the search
    '''
    Vizier.ROW_LIMIT = -1
    coord = SkyCoord(raCen, decCen, unit=(u.degree, u.degree), frame='icrs')
    width = str(width)+'d'

    tab_query = Vizier.query_region(coord, width=width, catalog='2MASS')
    cat = tab_query[0]

    return cat


def getSDSSCatalogue(raCen, decCen, radius):
    '''
    As get2MASSCatalogue, but using SDSS
        Parameters
        ----------
        raCen : `float'
            Right ascension coordinate of search box center, in decimal degrees
        decCen : `float'
            Declination coordinate of search box center, in decimal degrees
        radius : `float'
            Search radius, in decimal degrees

        Returns
        -------
        cat : `astropy.table.Table'
            Table with astroquery results for the search
    Currently defaults to only returning coordinates and g, r, i magnitudes.
    Returns psfMags, which are in nanomaggies.  Need to offset by 22.5 to get
    back to ~AB magnitudes.
    '''
    coord = SkyCoord(raCen, decCen, unit=(u.degree, u.degree), frame='icrs')
    radius *= u.degree
    fields = ['ra', 'dec', 'psfMag_g', 'psfMag_r', 'psfMag_i']
    cat = SDSS.query_region(coord, radius, fields=fields, data_release=15)

    return cat


def getGamaCatalogue(raCen, decCen, width):
    '''
    Downloads an object catalogue from GAMA, centered at a particular
    coordinate
        Parameters
        ----------
        raCen : `float`
            Right ascension coordinate of search box center, in decimal degrees
        decCen : `float`
            Declination coordinate of search box center, in decimal degrees
        width : `float`
            Width of search box, in decimal degrees

        Returns
        -------
        cat : `astropy.table.Table`
            Table with astroquery results for the search
    NOTE: limited to just returning RA, Dec, and i-band photometry parameters
    (magnitude, effective radius, Sersic index, ellipticity, PA, and central
     surface brightness)
    '''
    ra1 = str(raCen + width)
    ra2 = str(raCen - width)
    dec1 = str(decCen + width)
    dec2 = str(decCen - width)
    columns = 'RA, DEC, GALMAG_i, GALRE_i, GALINDEX_i, ' \
              + 'GALELLIP_i, GALPA_i, GALMU0_i'
    lim_ra = 'RA < ' + ra1 + ' AND RA > ' + ra2
    lim_dec = ' AND DEC < ' + dec1 + ' AND DEC > ' + dec2
    lims = lim_ra + lim_dec
    sql_query = 'SELECT ' + columns + ' FROM SersicCatSDSS WHERE ' + lims
    cat = GAMA.query_sql(sql_query)

    return cat


def findCentralStar(ra, dec, starCatalogue, ditherStep, bWidth, pxScale=0.168):
    angleStep = (ditherStep * pxScale) / 3600  # Degrees
    angleWidth = (bWidth * pxScale) / 3600  # Degrees
    want_ra = (starCatalogue['RAJ2000'] <= ra + angleStep - angleWidth/2) \
        & (starCatalogue['RAJ2000'] >= ra - (angleStep - angleWidth/2))
    want_dec = (starCatalogue['DEJ2000'] <= dec + angleStep - angleWidth/2) \
        & (starCatalogue['DEJ2000'] >= dec - (angleStep - angleWidth/2))

    id_ra = np.where(want_ra)[0]
    id_dec = np.where(want_dec)[0]

    return id_ra, id_dec


def makeSersicStamp(ra, dec, n, rEff, axRat, pa, mag, beta, fwhm, stampName,
                    magZp=27.0):
    '''
    Same as DrawModels.drawSersic(), but write the stamp to the disk
        Parameters
        ----------
        ra : `float`

        n : `float`
            Sersic index (useful range is 0.5 -- 6)
        rEff : `float'
            Half-light radius, in pixels
        axRat : `float`
            Projected axis ratio (values between 0 and 1)
        pa : `float`
            Position angle in degrees
        mag : `float'
            Object magnitude, to be converted into counts
        beta : `float`
            PSF beta value for convolution
        fwhm : `float`
            PSF FWHM for convolution
        stampName : `string`
            Filename of output image
        magZp : `float`
            Magnitude zeropoint, to convert from mag to counts

        Returns
        -------
        Nothing--writes image to hard disk
    NOTE: use this for large Sersic models!
    '''
    flux = 10**(-0.4*(mag - magZp))
    sersic = galsim.Sersic(n, rEff)
    sersic = sersic.shear(q=axRat, beta=((90 - pa)*galsim.degrees))
    sersic = sersic.withFlux(flux)
    # Convolving w/PSF
    psf = galsim.Moffat(beta=beta, fwhm=fwhm)
    sersic = galsim.Convolve([sersic, psf])

    try:
        gal_im = sersic.drawImage(scale=0.168, method="real_space").array
    except (galsim.errors.GalSimFFTSizeError, MemoryError):
        print('Memory error!')
        return None

    hdu = fits.PrimaryHDU(gal_im)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(stampName)
