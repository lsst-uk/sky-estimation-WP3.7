'''
Software for deriving and displaying photometry metrics, as well as functions to
read in and parse pickled dictionaries where photometry is stored
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib import patches as ptchs
from scipy.signal import medfilt

import utility as ut


# =============================================================================
# Utility Functions
# =============================================================================
def getMediansIcl(xVar, yVar, nBins, buffer):
    '''
    Used for making scatterplots.  Derives medians among points for
    NH Dwarf + ICL models, which are distributed more randomly in
    parameter space than Sersic models
        Parameters
        ----------
        xVar : `numpy.array'
            x-axis variable
        yVar : `numpy.array'
            y-axis variable
        nBins : `int'
            Number of bins desired when calculating medians
        buffer : `float'
            Used to include lower limits more effectively
            
        Yields
        -------
        bins : `numpy.array'
            x-axis bins, for plotting
        meds : `numpy.array'
            Running median of yVar in spacing bins
        stds : `numpy.array'
            Standard deviation of points in bins
    '''
    bins = np.linspace(xVar.min()-buffer, xVar.max(), nBins)
    delta = bins[1]-bins[0]
    idx  = np.digitize(xVar, bins)
    meds = []
    stds = []
    # Removing constant error messages for empty bins
    for k in range(nBins):
        if len(yVar[idx==k]) == 0:
            meds.append(np.nan)
            stds.append(np.nan)
        else:
            meds.append(np.nanmedian(yVar[idx==k]))
            stds.append(np.nanstd(yVar[idx==k]))
    meds = np.array(meds)
    stds = np.array(stds)
    
    return bins, meds, stds


def getMediansSersic(initVar, xVar, yVar, buffer):
    '''
    As getMediansIcl(), but takes advantage of the grid spacing among
    the parameters for the Sersic models
        Parameters
        ----------
        initTab : `numpy.array`
            Input grid version of xVar, even spacing
        xVar : `numpy.array`
            x-axis variable
        yVar : `numpy.array`
            y-axis variable
        buffer : `float`
            Desired width to account for photometry scatter among measured values
            
        Yields
        -------
        bins : `numpy.array'
            x-axis bins, for plotting
        meds : `numpy.array'
            Running median of yVar in spacing bins
        stds : `numpy.array'
            Standard deviation of points in bins
    '''
    bins = np.unique(initVar)
    meds = []
    stds = []
    # Removing constant error messages for empty bins
    for i in range(len(bins)):
        want = (xVar <= bins[i] + buffer) & (xVar >= bins[i] - buffer)
        meds.append(np.nanmedian(yVar[want]))
        stds.append(np.nanstd(yVar[want]))
    meds = np.array(meds)
    stds = np.array(stds)
    
    return bins, meds, stds


# =============================================================================
# Table Manipulation Functions
# =============================================================================
def readPhotometryTables(inputFitsFilePath, outputPhotometryPicklePath, band, iclStr='_icl'):
    '''
    Creates accessible Astropy tables from pickle dictionaries for 
    photometry comparison
        Parameters
        ----------
        inputFitsFilePath : `string'
            Full path on the Stack to the input FITS catalogue table
        outputPhotometryPicklePath : `string'
            Directory where the output coadd photometry pickle file is stored
        iclStr : `string'
            If using Sersic profiles, set to '', else set to '_icl'
        band : `string'
            Photometric band desired, ex. 'g', 'r', etc.
            
        Yields
        -------
        initTab : `astropy.table.Table'
            Input catalogue table
        inputTab : `astropy.table.Table'
            Table of photometry from visit-level, pre-sky-subtraction
        outputTab : `astropy.table.Table'
            Table of photometry measured from coadds, post-sky-subtraction
    '''
    initTab = Table.read(inputFitsFilePath)
    outputPickle = pickle.load(open(outputPhotometryPicklePath+'/coadd_mags'+iclStr+'.p', 'rb'))
    inputPickle = pickle.load(open(outputPhotometryPicklePath+'/mags'+iclStr+'.p', 'rb'))
    
    mags = []
    profs = []
    inds = []
    maxR = []
    for i in range(len(initTab)):
        try:
            mags.append(inputPickle[str(i)][band+'magVarMeas'])
            profs.append(inputPickle[str(i)][band+'Prof'])
            maxR.append(inputPickle[str(i)][band+'maxR'])
            inds.append(i)
        except:
            continue
    inputTab = {band+'magVar' : np.array(mags),
                band+'sbProf' : profs,
                'maxRadius' : maxR,
                'idx' : inds}
    
    # Only grabbing coadd mags for models that didn't fail
    coaddMags = []
    coaddProfs = []
    maxR = []
    patchIds = []
    for i in inds:
        try:
            coaddMags.append(outputPickle[str(i)][band+'magVarMeas'])
            coaddProfs.append(outputPickle[str(i)][band+'Prof'])
            maxR.append(outputPickle[str(i)][band+'maxR'])
            patchIds.append(outputPickle[str(i)][band+'patch'])
        except:
            # If the photometry failed for some reason
            coaddMags.append(np.nan)
            coaddProfs.append([np.nan])
            maxR.append(np.nan)
            patchIds.append(np.nan)            
    outputTab = {band+'magVar' : np.array(coaddMags),
                 band+'sbProf' : coaddProfs,
                 'maxRadius' : maxR,
                 'patchId' : patchIds}
    
    return initTab, Table(inputTab), Table(outputTab)


def cullPhotometryTables(initTab, inputTab, outputTab, magLim, band):
    '''
    Selects valid photometry results for plotting.  If photometry failed,
    unreasonable values result (99 by default, otherwise significantly lower than
    input catalogue value).
        Parameters
        ----------
        initTab : `astropy.table.Table'
            Input catalogue table
        inputTab : `astropy.table.Table'
            Table of photometry from visit-level, pre-sky-subtraction
        outputTab : `astropy.table.Table'
            Table of photometry measured from coadds, post-sky-subtraction
        magLim : `float'
            Defines acceptable magnitude underestimate on pre-sky-subtraction photometry
        band : `string'
            Photometric band desired, ex. 'g', 'r', etc.
            
        Yields
        -------
        reject : `numpy.array'
            Boolean array used to select models that are rejected from analysis.
            Acceptable values are thus selected by np.array[~reject]
    '''
    inds = inputTab['idx']
    rej1 = (inputTab[band+'magVar'] >= initTab[band+'magVar'][inds] + magLim)
    rej2 = (inputTab[band+'magVar'] > 40) | (inputTab[band+'magVar'] < 0)
    rej3 = (outputTab[band+'magVar'] > 40) | (outputTab[band+'magVar'] < 0)
    reject = rej1 | rej2 | rej3
    
    return reject


# =============================================================================
# Calculating difficult metrics
# =============================================================================
def sbOverSub(inputTab, outputTab, band, sbLim):
    '''
    Finds the point in every surface brightness profile at which the
    surface brightness is oversubtracted by an amount sbLim
        Parameters
        ----------
        inputTab : `astropy.table.Table'
            Table of photometry from visit-level, pre-sky-subtraction
        outputTab : `astropy.table.Table'
            Table of photometry measured from coadds, post-sky-subtraction
        band : `string'
            Photometric band desired, ex. 'g', 'r', etc.
        sbLim : `float'
            Desired amount of over-subtraction in mags/arcsec^2
            
        Yields
        -------
        sbSub : `numpy.array'
            Array containing surface brightnesses in every radial profile
            at which the post-sky-subtraction surface brightness is over-
            subtracted compared to the pre-sky-subtraction surface brightness
            by sbLim
    '''
    sbSub = []
    for i in range(len(inputTab)):
        sbin = inputTab[i][band+'sbProf']
        sbout = outputTab[i][band+'sbProf']
        delsb = sbout - sbin
        idx = ut.findNearest(delsb, sbLim, 0.01)
        sbSub.append(sbin[idx])
        
    return np.array(sbSub)


# =============================================================================
# Plotting Software for Jupyter Notebooks
# =============================================================================
def scatterPlot(fig, ax, xVarInit, xVar, yVar, zVar, boolFlag, yLims, xVarLab, yVarLab, zVarLab, buffer):
    '''
    Makes scatterplot of two variables color-coded by a third
    Tailored for Sersic profiles, the default, which uses a grid spacing
    among the parameters
        Parameters
        ----------
        fig : `matplotlib.pyplot.figure'
            Figure object made through plt.subplots()
        ax : `matplotlib.axes._subplots'
            Subplot on which to plot the points (e.g., fig, ax = plt.subplots())
        xVarInit : `numpy.array'
            Initial grid version of xVar, even spacing
        xVar : `numpy.array'
            x-axis variable
        yVar : `numpy.array'
            y-axis variable
        zVar : `numpy.array'
            Colorbar variable
        boolFlag : `numpy.array'
            Boolean array determining which points to reject
        yLims : `list'
            Limits on y-axis
        xVarLab : `string'
            x-axis label
        yVarLab : `string'
            y-axis label
        zVarLab : `string'
            Colorbar label
        buffer : `float'
            Used to take into account natural scatter in photometric measurements
            
        Yields
        -------
        Nothing.  Plots on existing figure.  
    '''
    bins, meds, __ = getMediansSersic(xVarInit, xVar[boolFlag], yVar[boolFlag], buffer)
    
    im = ax.scatter(xVar[boolFlag], 
                    yVar[boolFlag], 
                    marker = '.', 
                    c = zVar[boolFlag])
    ax.plot(bins, meds, 'r--')
    ax.axhline(0, color='k', linestyle=':')
    ax.set_ylim(yLims)
    ax.set_xlabel(xVarLab)
    ax.set_ylabel(yVarLab)
    ax.minorticks_on()
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(zVarLab)
    
    
def scatterPlotIcl(fig, ax, xVar, yVar, zVar, boolFlag, yLims, xVarLab, yVarLab, zVarLab, nBins, buffer):
    '''
    As scatterPlot(), but adjusts for the lack of grid spacing among parameters
    found in the ICL + NH Dwarfs catalogue
        Parameters
        ----------
        fig : `matplotlib.pyplot.figure'
            Figure object made through plt.subplots()
        ax : `matplotlib.axes._subplots'
            Subplot on which to plot the points (e.g., fig, ax = plt.subplots())
        xVar : `numpy.array'
            x-axis variable
        yVar : `numpy.array'
            y-axis variable
        zVar : `numpy.array'
            Colorbar variable
        boolFlag : `numpy.array'
            Boolean array determining which points to reject
        yLims : `list'
            Limits on y-axis
        xVarLab : `string'
            x-axis label
        yVarLab : `string'
            y-axis label
        zVarLab : `string'
            Colorbar label
        nBins : `int'
            Number of bins desired when calculating medians
        buffer : `float'
            Used to include lower limits more effectively on medians
            
        Yields
        -------
        Nothing.  Plots on existing figure.
    '''
    bins, meds, __ = getMediansIcl(xVar[boolFlag], yVar[boolFlag], nBins, buffer)
    
    im = ax.scatter(xVar[boolFlag], 
                    yVar[boolFlag], 
                    marker = '.', 
                    c = zVar[boolFlag])
    ax.plot(bins, meds, 'r--')
    ax.axhline(0, color='k', linestyle=':')
    ax.set_ylim(yLims)
    ax.set_xlabel(xVarLab)
    ax.set_ylabel(yVarLab)
    ax.minorticks_on()
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(zVarLab)
    
    
def plotMedsStds(initTab, inputTab, dVal, initMuEff, muEff, rejectBool, band, yLims, yLab):
    '''
    Overplots median values with standard deviation error bars on scatterplots
        Parameters
        ----------
        initTab : `astropy.table.Table'
            Input catalogue table
        inputTab : `astropy.table.Table'
            Table of photometry outputs
        dVal : `numpy.array'
            Array of delta values for which the median values are desired
        initMuEff : `numpy.array'
            Array of effective surface brightness from initTab parameters
        muEff : `numpy.array'
            Array of effective surface brightness from inputTab parameters
        rejectBool : `numpy.array'
            Numpy boolean array, to determine which points to exclude from medians
        band : `string'
            Photometric band desired, ex. 'g', 'r', etc.
        yLims : `list'
            Upper and lower limits for y-axis, e.g., [-1, 1]
        yLab : `string'
            Label for y-axis
            
        Yields
        -------
        Nothing.  Plots on existing figure.  
    '''
    magBins, magMeds, magStd = getMediansSersic(initTab[band+'magVar'], 
                                                inputTab[band+'magVar'][rejectBool], 
                                                dVal[rejectBool], 
                                                buffer=1)
    mueffBins, mueffMeds, mueffStd = getMediansSersic(initMuEff,
                                                      muEff[rejectBool], 
                                                      dVal[rejectBool], 
                                                      buffer=1)
    reffBins, reffMeds, reffStd = getMediansSersic(np.log10(initTab['BulgeHalfLightRadius']),
                                                   np.log10(initTab['BulgeHalfLightRadius'][inputTab['idx']][rejectBool]), 
                                                   dVal[rejectBool], 
                                                   buffer=0.3)
    nBins, nMeds, nStd = getMediansSersic(initTab['bulge_n'],
                                          initTab['bulge_n'][inputTab['idx']][rejectBool], 
                                          dVal[rejectBool], 
                                          buffer=0.5)

    fig, ax = plt.subplots(2, 2, figsize=(8,8))
    ax[0,0].errorbar(magBins, magMeds, yerr=magStd, fmt='.', capsize=5, color='k')
    ax[0,0].set_xlabel('Magnitude')

    ax[0,1].errorbar(mueffBins, mueffMeds, yerr=mueffStd, fmt='.', capsize=5, color='k')
    ax[0,1].set_xlabel(r'<$\mu_{\rm eff}$>')

    ax[1,0].errorbar(reffBins, reffMeds, yerr=reffStd, fmt='.', capsize=5, color='k')
    ax[1,0].set_xlabel(r'$\log($R$_{\rm eff}/{\rm arcsec})$')

    ax[1,1].errorbar(nBins, nMeds, yerr=nStd, fmt='.', capsize=5, color='k')
    ax[1,1].set_xlabel(r'n')
    
    for i in range(len(ax)):
        for j in range(len(ax)):
            ax[i,j].axhline(0, color='k', linestyle=':')
            ax[i,j].set_ylim(yLims)
            ax[i,j].set_ylabel(yLab)
            ax[i,j].minorticks_on()
    plt.tight_layout()
    
    
def plotMedsStdsIcl(initTab, inputTab, dVal, muEff, rejectBool, band, yLims, yLab):
    '''
    Same as plotMedsStds, but tailored for ICL models
        Parameters
        ----------
        initTab : `astropy.table.Table'
            Input catalogue table
        inputTab : `astropy.table.Table'
            Table of photometry outputs
        dVal : `numpy.array'
            Array of delta values for which the median values are desired
        initMuEff : `numpy.array'
            Array of effective surface brightness from initTab parameters
        muEff : `numpy.array'
            Array of effective surface brightness from inputTab parameters
        rejectBool : `numpy.array'
            Numpy boolean array, to determine which points to exclude from medians
        band : `string'
            Photometric band desired, ex. 'g', 'r', etc.
        yLims : `list'
            Upper and lower limits for y-axis, e.g., [-1, 1]
        yLab : `string'
            Label for y-axis
            
        Yields
        -------
        Nothing.  Plots on existing figure.  
    '''
    magBins, magMeds, magStd = getMediansIcl(inputTab[band+'magVar'][rejectBool], 
                                          dVal[rejectBool],
                                          nBins=30,
                                          buffer=1)
    mueffBins, mueffMeds, mueffStd = getMediansIcl(muEff[rejectBool], 
                                                dVal[rejectBool], 
                                                nBins=30,
                                                buffer=1)
    reffBins, reffMeds, reffStd = getMediansIcl(np.log10(initTab['BulgeHalfLightRadius'][inputTab['idx']][rejectBool]), 
                                             dVal[rejectBool], 
                                             nBins=8,
                                             buffer=0.3)
    nBins, nMeds, nStd = getMediansSersic(initTab['bulge_n'],
                                          initTab['bulge_n'][inputTab['idx']][rejectBool], 
                                          dVal[rejectBool], 
                                          buffer=0.5)

    fig, ax = plt.subplots(2, 2, figsize=(8,8))
    ax[0,0].errorbar(magBins, magMeds, yerr=magStd, fmt='.', capsize=5, color='k')
    ax[0,0].set_xlabel('Magnitude')

    ax[0,1].errorbar(mueffBins, mueffMeds, yerr=mueffStd, fmt='.', capsize=5, color='k')
    ax[0,1].set_xlabel(r'<$\mu_{\rm eff}$>')

    ax[1,0].errorbar(reffBins, reffMeds, yerr=reffStd, fmt='.', capsize=5, color='k')
    ax[1,0].set_xlabel(r'$\log($R$_{\rm eff}/{\rm arcsec})$')

    ax[1,1].errorbar(nBins, nMeds, yerr=nStd, fmt='.', capsize=5, color='k')
    ax[1,1].set_xlabel(r'n')
    
    for i in range(len(ax)):
        for j in range(len(ax)):
            ax[i,j].axhline(0, color='k', linestyle=':')
            ax[i,j].set_ylim(yLims)
            ax[i,j].set_ylabel(yLab)
            ax[i,j].minorticks_on()
    plt.tight_layout()
    
    
def plotParams(tab, supTitle):
    '''
    Makes histograms of all model parameters.    
        Parameters
        ----------
        tab : `astropy.table.Table'
            The table containing the parameters
        supTitle : `string'
            Overall title of the figure, showing models used
        
        Yields
        -------
        Nothing, just plots a figure.
    '''
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))

    ax[0,0].hist(tab['imagVar'])
    ax[0,0].set_xlabel(r'Magnitudes') # Note that all models have flat SEDs

    ax[0,1].hist(tab['BulgeHalfLightRadius'])
    ax[0,1].set_xlabel(r'R$_{\rm eff}$ (arcsec)')

    ax[0,2].hist(tab['bulge_n'], 15)
    ax[0,2].set_xlabel(r'n')

    ax[1,0].hist(tab['b_b'])
    ax[1,0].set_xlabel(r'Axial ratio')

    ax[1,1].hist(tab['pa_bulge'])
    ax[1,1].set_xlabel(r'PA (degrees)')

    mueff, __ = ut.getMuEff(tab['imagVar'], 
                            tab['BulgeHalfLightRadius'], 
                            tab['bulge_n'])
    ax[1,2].hist(mueff)
    ax[1,2].set_xlabel(r'$\mu_{\rm eff}$')
    
    fig.suptitle(supTitle)
    plt.tight_layout()
    
    
def plotCoords(tab, butler, supTitle, tract=9615):
    '''
    Displays stamp boxes on the tract supplied, as well as tract vertices.
        Parameters
        ----------
        tab : `astropy.table.Table'
            The table containing the parameters
        butler : `daf.persistence.Butler'
            Data fetching butler
        supTitle : `string
            Overall title of the figure, showing models used
        tract : `int'
            Tract ID
        
        Yields
        -------
        Nothing, just plots a figure.
    '''
    skymap = butler.get('deepCoadd_skyMap', immediate=True)
    tractInfo = skymap[tract]
    wcs = tractInfo.getWcs()
    pixelScale = wcs.getPixelScale().asArcseconds()
    cenx, ceny = tractInfo.getCtrCoord()
    verts = tractInfo.getVertexList()

    to_deg = lambda x: (x*pixelScale)/3600

    fig, ax = plt.subplots(1, figsize=(8,8))
    for i in range(len(tab['raJ2000'])):
        xcoo = np.degrees(tab['raJ2000'][i]) - 0.5*to_deg(tab['stampWidth'][i])
        ycoo = np.degrees(tab['decJ2000'][i]) - 0.5*to_deg(tab['stampWidth'][i])
        rect = ptchs.Rectangle((xcoo, ycoo),
                                to_deg(tab['stampWidth'][i]), to_deg(tab['stampWidth'][i]),
                                color=(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)),
                                alpha=0.6)
        ax.add_patch(rect)
    ax.plot(np.degrees(tab['raJ2000']), np.degrees(tab['decJ2000']), 'k.', markersize=2.0)
    ax.plot(cenx.asDegrees(), ceny.asDegrees(), 'rx')
    for vert in verts:
        ax.plot(vert[0].asDegrees(), vert[1].asDegrees(), 'rx')
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('Dec (degrees)')
    ax.set_title(supTitle)
    
    
def plotSBHists(ax, inputTab, outputTab, rejectBool, band, sbLim, densFlag=True):
    '''
    Plots histogram of values output by sbOverSub()
        Parameters
        ----------
        ax : `matplotlib.axes._subplots'
            Subplot on which to plot the histogram
        inputTab : `astropy.table.Table'
            Table of photometry from visit-level, pre-sky-subtraction
        outputTab : `astropy.table.Table'
            Table of photometry measured from coadds, post-sky-subtraction
        rejectBool : `numpy.array'
            Boolean array of models to not be included in the analysis
        band : `string'
            Photometric band desired, ex. 'g', 'r', etc.
        sbLim : `float'
            Desired amount of over-subtraction in mags/arcsec^2
        densFlag : `bool'
            If True, plots a histogram.  If False, plots a CDF.
            
        Yields
        -------
        sbSub : `numpy.array'
            Array containing surface brightnesses in every radial profile
            at which the post-sky-subtraction surface brightness is over-
            subtracted compared to the pre-sky-subtraction surface brightness
            by sbLim
    '''

    sbSub = sbOverSub(inputTab[~rejectBool], outputTab[~rejectBool], band, sbLim)

    good = np.isfinite(sbSub) # rejecting bad values
    md = np.nanmedian(sbSub[good])

    if densFlag:
        ax.hist(sbSub[good], 20, density=True)
        ax.set_ylim([0, 0.3])
    else:
        ax.hist(sbSub[good], 20, density=True, cumulative=True, histtype='step', color='k')
        ax.set_ylim([0, 1.01])
    ax.axvline(md, color='k', linestyle=':', label=r'$\mu_{\rm med}=$%.2f'%md)
    ax.set_xlim([12, 35])
    ax.legend()
    