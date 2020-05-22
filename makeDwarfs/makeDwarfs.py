#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:50:57 2020

@author: Aaron Watkins

Citation: Graham, A. (2019), PASA 36, 035    
"""

import numpy as np
from scipy.special import gammaincinv
from scipy.special import gamma
from astropy.io import ascii as asc
from astropy.table import Table


class Dwarfs():
    '''
    When an instance is created, attaches Sersic parameters to each stellar
    mass with some scatter.
    
    Args:
        mstar: array of log(Mstar/Msun), or single log(Mstar/Msun)
        sigma1: dispersion around log(Mstar)-log(n) relationship
        sigma2: dispersion around log(Mstar)-log(mu0) relationship
        fit1: tuple, slope and intercept of linear fits between log(Mstar)-log(n)
        fit2: tuple, slope and intercept of linear fits between log(Mstar)-mu0
        
    Attributes:
        self.mstar: log(Mstar/Msun)
        self.sigma1: dispersion in log(Mstar) vs. log(n) relationship
        self.sigma2: dispersion in log(Mstar) vs. mu0 relationship
        self.fit1: linear fit params to log(Mstar) vs. log(n) relationship
        self.fit2: linear fit params to log(Mstar) vs. mu0 relationship
        self.re: effective radius in pc
        self.n: Sersic index
        self.I0: central mass surface density in Msun/pc^2
        self.mabs: dictionary of absolute magnitudes in ugrizy (LSST, AB)
        self.q: axial ratio (randomly sampled from U(0.05,1))
        self.pa: position angle in degrees (randomly sampled from U(-90, 90))
    '''
    def __init__(self, mstar, sigma1=0.03, sigma2=1.2, fit1=(0.1, -0.73), fit2=(-1.48, 34.77)):
        self.mstar = mstar
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.fit1 = fit1
        self.fit2 = fit2
        self.re, self.n, self.I0, self.mabs = self.derive_params(mstar, 
                                                      sigma1, 
                                                      sigma2, 
                                                      fit1, 
                                                      fit2)
        self.q = np.random.uniform(0.05, 1, size=len(mstar)) # Minimum is GalSim min
        self.pa = np.random.uniform(-90, 90, size=len(mstar))
        self.n[self.n < 0.3] = 0.3 # GalSim doesn't take n < 0.3
    
    
    def sersic(self, r, I0, re, n):
        '''
        For testing/verification purposes
        
        Args:
            r: radius array
            I0: central surface brightness
            re: effective radius
            n: Sersic index
            
        Returns:
            I(r) in Mstar/pc^2 for chosen Sersic parameters
        '''
        b = bnn(n, 0.5)
        
        return I0 * np.exp(-b*(r/re)**(1/n))
    
    
    def mstar_v_n(self, mstar, sigma, fit):
        '''
        Predicts Sersic n given log-linear relation with log(Mstar)
        
        Args:
            m: slope
            b: intercept
        
        Returns:
            n: Sersic index given value of log(Mstar)
        '''
        
        noise = np.random.normal(0, sigma, size=len(mstar))
        logn = fit[0]*mstar + fit[1]
        logn += noise
        
        return 10**logn
    
    
    def mstar_v_mu0(self, mstar, sigma, fit):
        '''
        Predicts central surface brightness given log-linear relation with 
        log(Mstar)
        
        Args:
            m: slope
            b: intercept
        
        Returns:
            Central surface brightness in mstar/pc^2
        '''
        
        noise = np.random.normal(0, sigma, size=len(mstar))
        mu0 = fit[0]*mstar + fit[1]
        mu0 += noise
        # Derived from SPITZER 3.6um magnitudes
        I0 = 10**(-0.4*(mu0 - 27.59)) * 0.5
        
        return I0
    
    
    def get_mags(self, mstar):
        '''
        Derive absolute magnitudes for galaxies given stellar mass, assuming
        the Kirby + (2013) mass-metallicity relationship for dwarfs holds
        
        Model parameters, using online tool:
        E-MILES basti kb 1.30 baseFe parametric
        3000 - 12000 A, linear, AB mags
        
        Args:
            mstar: log(solar mass) value
            
        Returns:
            Dictionary of absolute AB magnitudes in ugrizy LSST bands
        '''
        magtab = asc.read('lsst.MAG', comment='#')
        masstab = asc.read('lsst.MASS', comment='#')
        m_on_h = np.arange(-2.4, 0.7, 0.1)
        
        # Applying errors on the params from Kirby + 2013
        ran1 = np.random.normal(0.0, 0.04, len(mstar))
        ran2 = np.random.normal(0.0, 0.02, len(mstar))
        z_data = -1.69+ran1 + (0.3+ran2)*(mstar - 6)
        
        filters = [key for key in magtab.keys() if key!='model']
        mabs = {}
        if len(mstar)==0:
            for filt in filters:
                idx = find_nearest(m_on_h, z_data)
                mag = magtab[filt][idx]
                mfrac = masstab['Mstar_ali'][idx]
                
                mag1msun = mag + 2.5*np.log10(mfrac)
                mabs[filt] = mag1msun - 2.5*mstar \
                             + np.random.uniform(0, 0.05)
        else:
            for filt in filters:
                mabs[filt] = np.zeros(len(mstar))
                for i in range(len(mstar)):
                    idx = find_nearest(m_on_h, z_data[i])
                    mag = magtab[filt][idx]
                    mfrac = masstab['Mstar_ali'][idx]
                    
                    mag1msun = mag + 2.5*np.log10(mfrac)
                    mabs[filt][i] = mag1msun - 2.5*mstar[i] \
                                    + np.random.uniform(0, 0.05)
        
        return mabs
    
    
    def derive_params(self, mstar, sigma1, sigma2, fit1, fit2):
        '''
        Derive effective radius using total stellar mass
        Returns the two parameters needed by GalSim (Re and n) to build model
        galaxies.
        
        Args:
            mstar: log10(mstar/msun)
            fit1: tuple of slope and intercept for Mstar vs. n relationship
            fit2: tuple of slope and intercept for Mstar vs. mu0 relationship
            
        Returns:
            Effective radius (in pc)
            Sersic index
            Central surface brightness (Mstar/pc^2)
        '''
        
        n = self.mstar_v_n(mstar, sigma1, fit1)
        I0 = self.mstar_v_mu0(mstar, sigma2, fit2)
        bn = bnn(n, 0.5)
        
        re = np.sqrt((10**(mstar)*bn**(2*n)) / (I0 * 2*n * np.pi * gamma(2*n)))
        
        mabs = self.get_mags(mstar)
        
        return re, n, I0, mabs
    

# =============================================================================
# Utility Functions
# =============================================================================

def find_nearest(array, value):
    '''
    Utility function to find array element closest to input value
    
    Args:
        array: array in which you want to find the value
        value: value you want to find in the array
        
    Returns:
        Index of the array containing the element closest to input value
    '''
    array = array[np.isfinite(array)]
    idx = (np.abs(array-value)).argmin()
    return idx


def bnn(n, frac):
    '''
    Calculates value of b for Sersic index/light fraction nn
    
    Args:
        n: Sersic index
        frac: fraction of total light (e.g. 0.5 = 50%)
        
    Returns:
        b: Sersic constant b_n
    '''
    
    return gammaincinv(2*n, frac)


def schechter(logmstar, phi_star=1, logm_star=10.2, alpha=-1.35):
    '''
    Output a Schechter function for stellar mass
    Default values approximately derived from Venhola et al. (2019)
    
    Args:
        phi_star: normalization at m_star
        m_star: the knee of the function
        alpha: the slope at the low mass end
        
    Returns:
        A Schechter function with the given parameters
    '''
    mstar = 10**logmstar
    m_star = 10**logm_star
    return phi_star * np.exp(-mstar/m_star) * (mstar/m_star)**(alpha+1)
    

def mass_from_schechter(logmstar, N, phi_star=1, logm_star=10.2, alpha=-1.35):
    '''
    Creates a random distribution of stellar masses that follow the low-mass
    end of the Schechter function.
    
    Args:
        logmstar: array of stellar masses in log(M/Msun)
        N: output sample size
        
    Returns:
        Array of logmstar sampled randomly from a Schechter function
    '''
    phi = np.log10(schechter(logmstar, phi_star, logm_star, alpha))
    if np.min(phi) < 0:
        phi -= np.min(phi)
    phi /= np.sum(phi) # Using this as the weights for random sampling
    
    return np.random.choice(logmstar, size=N, p=phi)


def distribute_coords_as_gaussian(cenra, cendec, stdra, stddec, size):
    '''
    Outputs random RA, DEC positions centered somewhere on the sky as a 2D
    Gaussian
    
    Args:
        cenra: right ascension of central coordinate
        cendec: declination of central coordinate
        stdra: standard deviation along the RA direction
        stddec: standard deviation along the DEC direction
        size: number of random (RA, DEC) pairs to output
        
    Returns:
        Tuple containing right ascensions and declinations distributed as 
        N(mu, Sigma) with mu = (cenra, cendec) and Sigma=[[stdra, 0], [0, stddec]]
    '''
    mean = [cenra, cendec]
    cov = [[stdra, 0], [0, stddec]]
    rand_ra, rand_dec = np.random.default_rng().multivariate_normal(mean, 
                                                                    cov, 
                                                                    size).T
    coords = (rand_ra, rand_dec)
    return coords
    

# =============================================================================
# Outputting values as FITS table
# =============================================================================

def write_to_table(dwarf, coords, fname):
    '''
    Outputs dwarf galaxy catalog as a FITS table in HSC injection format
    
    Args:
        dwarf: instance of Dwarfs() class
        coords: tuple of RA and DEC coordinates (see distribute_as_gaussian)
        fname: output table filename (throws error if not a FITS table)
        
    Returns:
        Nothing, but write table to working directory
    '''
    if '.fits' not in fname:
        print('Table filename must be ***.fits')
    else:
        model = {
                'raJ2000': coords[0],
                'decJ2000': coords[1],
                'a_b': np.zeros(len(dwarf.n))+1, # Normalized semi-major axis to 1
                'b_b': dwarf.q, # q = b/a = b given above
                'pa_bulge': dwarf.pa,
                'umagVar': dwarf.mabs['umagVar'],
                'gmagVar': dwarf.mabs['gmagVar'],
                'rmagVar': dwarf.mabs['rmagVar'],
                'imagVar': dwarf.mabs['imagVar'],
                'zmagVar': dwarf.mabs['zmagVar'],
                'ymagVar': dwarf.mabs['ymagVar'],
                'BulgeHalfLightRadius': dwarf.re,
                'bulge_n': dwarf.n,
                'DiskHalfLightRadius': dwarf.re, # same params as bulge
                'disk_n': dwarf.n #same params as bulge
                }
        
        t = Table(model)
        t.write(fname, format='fits', overwrite=True)
        print('File written as ./'+fname)






























