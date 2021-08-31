'''
Functions for doing surface photometry on 2D arrays using elliptical isophotes.
'''
import numpy as np


def mkEll(imData, xCen, yCen, pa, ell):    
    '''
    For reprojecting an array into elliptical coordinates
        Parameters
        ----------
        imData : `numpy.array`
            Input image array
        xCen : `float` or `int`
            Ellipse center x coordinate on the image, in px
        yCen : `float` or `int`
            Ellipse center y coordinate on the image, in px
        pa : `float`, `int`, or `numpy.array'
            Ellipse position angle(s)
            Due north in standard image orientation is 0 degrees,
            increasing CCW (DS9 definition of PA)
        ell : `float` or `numpy.array'
            Ellipse ellipticity (ellipticities): 1 - b/a
        
        Yields
        -------
        ellipse : `dict'
            Python dictionary containing the following keys:
            `xEll' : `numpy.array'
                x-axis coordinates in the ellipse plane
            `yEll' : `numpy.array'
                y-axis coordinates in the ellipse plane
            `ellRad' : `numpy.array'
                Ellipse-plane radius values at each pixel
            `theta' : `numpy.array'
                Ellipse-plane angle values at each pixel, in radians
                Starts at 0 at pa and increases CCW
                
        NOTE: can view output by doing either of the following in Matplotlib:
        plt.pcolormesh(xEll, yEll, imData)
        plt.imshow(ellRad, origin='lower')
    '''
    # Orientation depends on the image shape for unequal axis lengths
    dimx = imData.shape[1]
    dimy = imData.shape[0]
    if (dimy > dimx):
        x = np.arange(1, dimy+1)
        y = x.reshape(-1, 1)
    else:
        x = np.arange(1, dimx+1)
        y = x.reshape(-1, 1)
    # Currently written to start at PA and increase CCW (due north is 0 degrees)
    xEll = (y - yCen)*np.cos(np.radians(pa)) - (x - xCen)*np.sin(np.radians(pa))
    yEll = (y - yCen)*np.sin(np.radians(pa)) - (x - xCen)*np.cos(np.radians(pa))

    ellRad = np.sqrt(xEll**2 + (yEll/(1-ell))**2)
    theta = np.arctan2(yEll, xEll)
    theta[theta < 0.] = theta[theta < 0.] + 2.0*np.pi # Removing negative values
    
    ellipse = {'xEll': xEll,
               'yEll': yEll,
               'ellRad': ellRad,
               'theta': theta}
    
    return ellipse


def surfBriProfs(imData, xCen, yCen, pa, ell, bWidth, maxBin, maskVal=-999., magZp=27.0):
    '''
    Makes isophotal radial profiles from image data.  Sped up slightly by 
    using only relevant image sections per isophote.
        Parameters
        ----------
        imData : `numpy.array`
            Input image array
        xCen : `float` or `int`
            Ellipse center x coordinate on the image, in px
        yCen : `float` or `int`
            Ellipse center y coordinate on the image, in px
        pa : `float`, `int`, or `numpy.array'
            Ellipse position angle
            Due north in standard image orientation is 0 degrees,
            increasing CCW (DS9 definition of PA)
        ell : `float` or `numpy.array'
            Ellipse ellipticity: 1 - b/a
        bWidth : `float' or `int'
            Radial bin width in ellipse coordinates, in px
        maxBin : `float' or `int'
            Maximum radius for photometry, in px
        maskVal : `float'
            Value to ignore in imData when doing photometry (default -999.)
        magZp : `float'
            Magnitude zeropoint
            Soley used to ensure bad values are returned as 99 (default 27.0)
            
        Yields
        -------
        photDict : `dict'
            Python dictionary containing the following keys:
            `rad' : `numpy.array'
                Array of bins used for photometry (designated by bin midpoints)
            `meanI' : `numpy.array'
                Mean values of isophotes with radii `rad', in native image units
            `medI' : `numpy.array'
                Median values of isophotes with radii `rad', in native image units
            `cog' : `numpy.array'
                Total integrated counts within radii `rad', in native image units
            `nPix' : `numpy.array'
                Number of pixels enclosed within radii `rad'
    NOTE: `cog' is only a crude measure for real sources.  It ignores masks and includes
    ALL flux within the aperture, including overlapping sources.  Proper curves of growth
    require something like interpolation across masked out overlapping sources.
    So you can use the output surface brightnesses generally for masked images, but not the 
    curves of growth.
    '''
    xcen_i = int(np.round(xCen, 0)) # Converting to integer value
    ycen_i = int(np.round(yCen, 0)) # Converting to integer value
    
    # Check if multiple values of pa and ell are provided
    if len(np.array([pa]))==1:
        ellipse = mkEll(imData, xcen_i, ycen_i, pa, ell)
        ellRad = ellipse['ellRad']
        ellRad = ellRad[ : imData.shape[0], : imData.shape[1]]
    
    # Creating bins array using bin midpoints, for easier access
    bins = np.arange(0.5*bWidth, maxBin+0.5*bWidth, bWidth)
    # Setting up dictionary and storage arrays for photometry
    photDict = {}
    mean_i = np.zeros(len(bins))-999
    med_i = np.zeros(len(bins))-999
    cog = np.zeros(len(bins))-999
    npix = np.zeros(len(bins))
    for i in range(len(bins)):
        # First shrinking image array to only needed size
        minbin = bins[i]-0.5*bWidth
        maxbin = bins[i]+0.5*bWidth
        offset = int(np.round(maxbin+5)) # How far to grow the image box
        imsec = imData[ycen_i-offset : ycen_i+offset,
                       xcen_i-offset : xcen_i+offset]
        
        # Trimming ellRad array to imsec dimensions
        if len(np.array([pa]))==1:
            ellRadsec = ellRad[ycen_i-offset : ycen_i+offset,
                               xcen_i-offset : xcen_i+offset]
            want_sb = (ellRadsec >= minbin) & (ellRadsec < maxbin) & (imsec != maskVal)
            want_cog = (ellRadsec < maxbin) # Warning: no masking here!
        else:
            # If multiple pa, ell are provided, have to remake the ellRad array for each bin
            ellipse = mkEll(imsec, offset, offset, pa[i], ell[i])
            ellRad = ellipse['ellRad']
            want_sb = (ellRad >= minbin) & (ellRad < maxbin) & (imsec != maskVal)
            want_cog = (ellRad < maxbin) # Warning: no masking here!
            
        # Ignore empty bins
        if len(want_sb[want_sb]) == 0:
            # Flaggin bad values as 99 when converted to surface brightness
            mean_i[i] = 10**(-0.4*(99-magZp))
            med_i[i] = 10**(-0.4*(99-magZp))
        else:
            mean_i[i] = np.nanmean(imsec[want_sb])
            med_i[i] = np.nanmedian(imsec[want_sb])
        cog[i] = np.sum(imsec[want_cog])
        npix[i] = len(imsec[want_cog])
    # Flagging bad values as 99 when converted to surface brightness
    cog[cog<=0] = 10**(-0.4*(99-magZp))
    
    photDict['rad'] = bins
    photDict['meanI'] = mean_i
    photDict['medI'] = med_i
    photDict['cog'] = cog
    photDict['nPix'] = npix
    
    return photDict
