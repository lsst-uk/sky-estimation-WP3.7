'''
A variety of useful functions that don't belong anywhere else.
'''
import numpy as np
import astropy
import astropy.coordinates as coord
import pandas as pd
from scipy.special import gamma, gammaincinv


def matchTables(table1, table2,
                table1Ra = 'ra', table1Dec = 'dec', table1Unit = 'deg',
                table2Ra = 'ra', table2Dec = 'dec', table2Unit = 'deg',
                maxSep = None):
    '''
    Matches and joins two tables.  Modification of a code originally
    written by Lee Kelvin.
        Parameters
        ----------
        table1 : `astropy.table.table.Table` OR `pandas.core.frame.DataFrame`
            First input table
        table2 : `astropy.table.table.Table` OR `pandas.core.frame.DataFrame`
        table1Ra : `string`
            Right ascension key name in table1
        table1Dec : `string`
            Declination key name in table1
        table1Unit : `string`
            Unit of RA and Dec in table1 (for astropy.coordinates.coord.SkyCoord)
        table2Ra : `string`
            Right ascension key name in table2
        table2Dec : `string`
            Declination key name in table2
        table2Unit : `string`
            Unit of RA and Dec in table2 (for astropy.coordinates.coord.SkyCoord)   
        maxSep : `float`
            Maximum allowable separation for coordinate matches, in arcsec
            To not cull selection based on this, set to None
            
        Yields
        -------
        res : `pandas.core.frame.DataFrame`
            Concatenated, matched table combining table1 and table2
    '''
    if type(table1) is astropy.table.table.Table:
        table1 = table1.to_pandas()
    if type(table2) is astropy.table.table.Table:
        table2 = table2.to_pandas()
    table1 = pd.DataFrame(table1)
    table2 = pd.DataFrame(table2)
    
    # Matching coordinates between tables
    c1 = coord.SkyCoord(ra=table1[table1Ra], dec=table1[table1Dec], unit=table1Unit)
    c2 = coord.SkyCoord(ra=table2[table2Ra], dec=table2[table2Dec], unit=table2Unit)
    idx1, sep2d1, dist3d1 = coord.match_coordinates_sky(c1, c2)
    table21 = table2.iloc[idx1,:]
    
    # Combining the matched tables
    res = pd.concat([table1, table21.reset_index(drop=True)], axis=1, ignore_index=True, sort=False)
    res.columns = table1.columns.to_list() + table21.columns.to_list()
    res['sep2d'] = sep2d1.arcsec
    res['idx'] = idx1
    if maxSep is not None:
        good = sep2d1.arcsec <= maxSep
        res = res[good]
        
    return res


def findNearest(arr, val, tol):
    '''
    Finds the index in an array where the array value
    matches the input value to some tolerance
        Parameters
        ----------
        arr : `numpy.array`
            Array in which to search for the value
        val : `float`
            Value to search for in the array
        tol : `float`
            Tolerance on success at finding the value
            
        Yields
        -------
        i : `int`
            Index in arr where arr[i] is closest to val,
            within linear tolerance tol
    '''
    i = -1
    arr2 = -99
    while (i < len(arr)-1) & (arr2 < (val-tol)):
        i += 1
        if arr[i] == -99:
            arr2 = -99
        else:
            arr2 = arr[i]
            
    return i


def mkLogIm(image):
    '''
    Creates a logarithmically scaled version of an image
    in the manner of DS9, showing also negative flux.
    DO NOT USE FOR ANALYSIS!
        Parameters
        ----------
        image : `numpy.array`
            Image array (2D)
            
        Yields
        -------
        lgim : `numpy.array`
            Log-scaled version of image, with negative values
            replaced by positive ones
    '''
    lgim = np.log10(image)
    lg2 = np.log10(-image)
    lgim[np.isnan(lgim)] = lg2[np.isnan(lgim)]
    
    return lgim


# =============================================================================
# Sersic Profile Functions
# =============================================================================
def bnn(n, frac):
    '''
    Derives Sersic bn parameter for given light fraction
        Parameters
        ----------
        n : `float`
            Sersic index
        frac : `float`
            Light fraction, e.g. 0.5
        
        Yields
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
            
        Yields
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
            
        Yields
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
