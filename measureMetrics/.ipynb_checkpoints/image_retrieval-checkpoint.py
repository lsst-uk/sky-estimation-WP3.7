'''
Module containing functions for retrieving stamps using the Butler.
'''
import numpy as np
from lsst.daf.persistence import Butler # Gen 2 Butler
#from lsst.daf.butler import Butler # Gen 3 Butler
import lsst.afw.image as afwImage
import lsst.geom as geom


def getCoaddCutout(ra, dec, stampWidth, butler, band, datasetType, tract=9615):
    '''
    Make cutout of image from the coadd given coordinates    
        Parameters
        ----------
        ra : `float`
            Right ascension in radians
        dec : `float`
            Declination in radians
        stampWidth : `int`
            The desired cutout width, in px
        butler : `daf.persistence.Butler`
            Data fetching butler (gen. 2, for now)
        band : `string`
            Photometric band, e.g., 'i' or 'I' for i-band
        datasetType : `string`
            Desired product, e.g. 'calexp', 'fakes_calexp'
        tract : `int`
            Tract ID (default 9615)
        
        Yields
        -------
        cutout : `numpy.array`
            The postage stamp cutout, centered at the coordinate
        magZp : `float`
            Magnitude zeropoint for the stamp (mag = -2.5log10(flux) + magZp)
    '''    
    # Convert ra and dec to spherepoint object
    coordinate = geom.SpherePoint(ra, dec, geom.radians)
    
    # Derive the appropriate patch ID for the coordinate
    skymap = butler.get('deepCoadd_skyMap', immediate=True)
    tractInfo = skymap[tract]
    patchInfo = tractInfo.findPatch(coordinate)
    patchId = patchInfo.getIndex()
    pch = "%d,%d" % patchId
    
    # Retrieve the exposure and derive central pixel coordinates (x,y)
    coaddId = dict(tract=tract, patch=pch, filter="HSC-"+band.upper())
    exp = butler.get(datasetType, coaddId, immediate=True)
    xy = exp.getWcs().skyToPixel(coordinate) - exp.getBBox().getMin()
    x = xy[0]
    y = xy[1]
    
    # Check image boundaries; return None if stampWidth exceeds patch bounds
    sz = exp.image.array.shape[0]
    boundX = (int(np.round(x-stampWidth//2)) <= 0) | (int(np.round(x+stampWidth//2)) >= sz)
    boundY = (int(np.round(y-stampWidth//2)) <= 0) | (int(np.round(y+stampWidth//2)) >= sz)
    bounds = boundX | boundY
    if bounds:
        print('stampWidth exceeds patch boundaries: use getRobustCutout instead!')
        return None, None
    
    # Trim the exposure to the desired stamp size, centered at (x,y)
    stamp = exp.image.array[int(np.round(y-stampWidth//2)) : int(np.round(y+stampWidth//2)),
                            int(np.round(x-stampWidth//2)) : int(np.round(x+stampWidth//2))]
    
    return stamp, exp.getPhotoCalib().instFluxToMagnitude(1)


def getCcdVisit(exp):
    '''
    Finds all CCD and Visit IDs from a particular exposure object
        Parameters
        ----------
        exp : `lsst.afwImage.ExposureF`
            An exposure object retrieved via the Butler
        
        Yields
        -------
            ccds : `list'
                A list of CCD IDs that correspond to the exposure object
            visits : `list'
                A list of visit IDs that correspond to the exposure object
    '''
    ccdInputs = exp.getInfo().getCoaddInputs().ccds
    visitKey = ccdInputs.schema.find("visit").key
    ccdKey = ccdInputs.schema.find("ccd").key
    
    # Find all the visit and ccd IDs on the given patch
    ccds = []
    visits = []
    for ccdRecord in ccdInputs:
        v = ccdRecord.get(visitKey)
        c = ccdRecord.get(ccdKey)
        ccds.append(c)
        visits.append(v)
        
    return ccds, visits


def getXyCoord(wcs, coord):
    '''
    Converts celestial coordinate to CCD coordinates on calexp image
        Parameters
        ----------
        wcs: `lsst.afw.geom.SkyWcs'
            Exposure WCS
        coord : `lsst.geom.SpherePoint`
            RA, Dec coordinates
        
        Yields
        -------
            x : `float'
                Image x coordinate in pixels, corresponding to coord
            y : `float'
                Image y coordinate in pixels, corresponding to coord
    '''
    xy = wcs.skyToPixel(coord)
    x = xy[0]
    y = xy[1]
    
    return x, y
    

def getVisitCutout(ra, dec, stampWidth, butler, band, datasetType, tract=9615, exp=None):
    '''
    Make cutout of an image at visit level based on object coordinates    
        Parameters
        ----------
        ra : `float`
            Right ascension in radians
        dec : `float`
            Declination in radians
        stampWidth : `int`
            The desired cutout width, in px 
        butler : `daf.persistence.Butler`
            Data fetching butler (gen. 2, for now)
        band : `string`
            Photometric band, e.g., 'i' or 'I' for i-band
        datasetType : `string`
            Desired product, e.g. 'calexp', 'fakes_calexp'
        tract : `int`
            Tract ID (default 9615)
        exp : `lsst.afwImage.ExposureF`
            If not None, the exposure object loaded in previously by the Butler
        
        Yields
        -------
        cutout : `numpy.array`
            The postage stamp cutout, centered at the coordinate
        magZp : `float`
            Magnitude zeropoint for the stamp (mag = -2.5log10(flux) + magZp)
            
    NOTE: this function will retrieve only the first instance of a visit-level
    calexp on which the full model appears, not a specific visit,CCD ID.  This
    is useful for deriving model parameters at the point of injection, prior to 
    sky-subtraction, not for tracking visit-to-visit changes or the like.
    '''
    # Convert ra and dec to spherepoint object
    coord = geom.SpherePoint(ra, dec, geom.radians)
    
    # Derive the appropriate patch ID for the coordinate
    skymap = butler.get('deepCoadd_skyMap', immediate=True)
    tractInfo = skymap[tract]
    patchInfo = tractInfo.findPatch(coord)
    patchId = patchInfo.getIndex()
    pch = "%d,%d" % patchId
    
    # First retrieve the full coadd exposure
    # Visit and CCD keys are embedded within this
    coaddId = dict(tract=tract, patch=pch, filter="HSC-"+band.upper())
    if exp == None:
        # NOTE: below altered from fakes_deepCoadd for more general use
        exp = butler.get('deepCoadd', coaddId, immediate=True)
    ccds, visits = getCcdVisit(exp)

    for j in range(len(ccds)):
        # Check all visits until one is found containing the whole stamp
        wcs = butler.get('calexp_wcs', tract=tract, visit=visits[j], ccd=ccds[j])
        x, y = getXyCoord(wcs, coord)
        boundX = (x - stampWidth//2 >= 0) & (x + stampWidth//2 <= 2048)
        boundY = (y - stampWidth//2 >= 0) & (y + stampWidth//2 <= 4176)
        want = boundX & boundY
        if want:
            brk = 0 # Once found, break loop and grab exposure
            break
        else:
            brk = 1 # Otherwise, jump to next j loop
    # If the coordinate wasn't found on any CCD or Visit, return error
    if brk:
        print('No viable visit was found for the given model!')
        return (None, None)
    else:
        # Otherwise, grab the exposure, then select proper stamp from it
        calexp = butler.get(datasetType, tract=tract, visit=visits[j], ccd=ccds[j])
        stamp = calexp.image.array[int(y-stampWidth//2) : int(y+stampWidth//2),
                                   int(x-stampWidth//2) : int(x+stampWidth//2)]
        return stamp, calexp.getPhotoCalib().instFluxToMagnitude(1)
    
    
def getRobustCutout(ra, dec, stampWidth, butler, band, datasetType, tract=9615, tractInfo=None):
    '''
    Courtesy of Markus Dirnberger, with minor modifications by A. Watkins
    Make cutout at coordinate with given size (regardless of underlying patch layout).    
        Parameters
        ----------
        ra : `float`
            Right ascension in radians
        dec : `float`
            Declination in radians
        stampWidth : `int`
            The desired cutout width, in px
        butler : `daf.persistence.Butler`
            Data fetching butler (gen. 2 for now)
        band : `string`
            Photometric band, e.g., 'i' or 'I' for i-band
        datasetType : `string`
            Desired product, e.g. 'fakes_deepCoadd', 'deepCoadd'
        tract : `int`
            Tract ID
        tractInfo : `afw.image.exposure`
            If None, is derived from butler at runtime
        
        Yields
        -------
        cutout : `lsst.afwImage.ExposureF`
            The postage stamp cutout, centered at the coordinate.
            
    NOTE: only this function returns the full afwImage exposure object.  The others
    return only the lsst.afwImage.ExposureF.image.array object.  Use caution!
    '''
    if tractInfo is None:
        skymap    = butler.get('deepCoadd_skyMap', immediate=True)
        tractInfo = skymap[tract]    # Get world coordinate system
    wcs = tractInfo.getWcs()
    coordinate = geom.SpherePoint(ra, dec, geom.radians)
    
    # Calculate desired bounding box for the final cutout
    size = geom.ExtentI(stampWidth, stampWidth)
    cutoutBBox = geom.Box2I().makeCenteredBox(wcs.skyToPixel(coordinate), size)
    
    # Calculate the bounding box of the patch containing the object
    patchInfo = tractInfo.findPatch(coordinate)
    patchBBox = patchInfo.getOuterBBox()
    
    # Get the patch index
    patch = patchInfo.getIndex()
    
    # Get an exposure that contains the cutout
    if patchBBox.contains(cutoutBBox):
        # The exposure is simply the patch if the BBox is fully contained
        dataId = {'tract': tract, 'filter': "HSC-"+band.upper(), 'patch': "%d,%d" % patch}
        exposure = butler.get(datasetType, **dataId)
        cutout = exposure.getCutout(coordinate, size)
    else:
        # Make a larger exposure that contains the entire cutout BBox otherwise
        x, y = patch
        nearbyPatches = [
                         (x-1,y-1), (x-1,y+0), (x-1,y+1), (x+0,y-1),
                         (x,y),
                         (x+0,y+1), (x+1,y-1), (x+1,y+0), (x+1,y+1)
                        ]
        bad = []
        for i,n in enumerate(nearbyPatches):
            if -1 in n:
                bad.append(i)
        for b in sorted(bad, reverse=True):
            del nearbyPatches[b]
            
        bad = []
        for i,n in enumerate(nearbyPatches):
            if 9 in n:
                bad.append(i)
                
        for b in sorted(bad, reverse=True):
            del nearbyPatches[b]
                
        overlappingPatches = [
                              n
                              for n in nearbyPatches
                              if cutoutBBox.overlaps(tractInfo.getPatchInfo(n).getOuterBBox())
                             ]
        
        # Make a bounding box for the big exposure
        bigBBox = patchBBox
        for o in overlappingPatches:
            bigBBox.include(tractInfo.getPatchInfo(o).getOuterBBox())
            
        # Make the big exposure by assigning subimages
        exposure = afwImage.ExposureF(bigBBox, tractInfo.getWcs())
        for o in overlappingPatches:
            dataId = {'tract': tract, 'filter': "HSC-"+band.upper(), 'patch': "%d,%d" % o}
            partialExposure = butler.get(datasetType, **dataId)
            # Get image data for a section of the big exposure
            subImage = afwImage.ImageF(exposure.getImage(), tractInfo.getPatchInfo(o).getOuterBBox())
            # Assign the patch image data
            subImage.assign(partialExposure.getImage())
            
        cutout = exposure.getCutout(coordinate, size)
            
    return cutout
