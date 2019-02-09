####
# last edited by Claire Valva on Feb 9, 2019
# this file performs the extended ifft for the full repeats 
# EDITED FOR ENSO FILES ONLY
# (as a hope to get convergence)
####

## FILES THAT ARE NEEDED:
# test_phases_nowind
# names_matched

#import packages
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
import scipy.integrate as sciint
import pandas as pd
import matplotlib.cm as cm
import matplotlib.ticker as tck
from math import pi
from sympy import solve, Poly, Eq, Function, exp, re, im
from netCDF4 import Dataset, num2date # This is to read .nc files and time array
from scipy.optimize import fsolve
from IPython.display import display, Markdown, Latex
import matplotlib.colors as colors
from seaborn import cubehelix_palette #for contour plot colors
import seaborn as sns
from cartopy.util import add_cyclic_point
from decimal import Decimal
import pickle
import time
import random

#get phases of transforms, that were solved in phase_check notebook
file_Name = "test_phases_nowind"
file_pickle = open(file_Name,'rb') 
file = pickle.load(file_pickle)

##get names to later match season + year to arrays and seasonal averaging
file_Name = "names_seasons"
file_pickle = open(file_Name,'rb') 
names_matched, indices_matched_time = pickle.load(file_pickle)

## convert loaded objects in usable formats, average over seasons
#get zonal spacing array for plotting later
zonal_spacing = fftfreq(240,1.5)
zonal_spacing = 1/zonal_spacing
zonal_spacing= 360 / zonal_spacing

#puts arrays into list formats (not sure exactly why this is necessary)
#however it is, otherwise isn't compatible with modulo operation
tested = [[[list(leaf)[1] for leaf in stem] for stem in trunk] for trunk in file]
phases = [[[np.remainder(leaf, 2*pi) 
            for leaf in stem] 
           for stem in trunk] 
          for trunk in tested]

amps = [[[list(leaf)[0] for leaf in stem] for stem in trunk] for trunk in file]

#sorts phases and amplitudes into seasons, whose index matches list: seasons
seasons = ['winter', 'spring', 'summer', 'fall']

#sort them into each season
season_phases = [[phases[i] for i in range(len(phases)) 
               if names_matched[i][1] == part] for part in seasons]

#sort them into each season
season_amps = [[amps[i] for i in range(len(amps) - 1) 
               if names_matched[i][1] == part] for part in seasons]

#make lists of el nino/regular/la nina years
nino = [1980,1983,1987,1988,1992,
        1995,1998,2003,2007,2010]
neutral = [1979,1981,1982,1984,1985,1986,1990,
           1991,1993,1994,1996,1997,2001,2002,
           2004,2005,2006,2009,2013,2014,2015,2016]
nina = [1989,1999,2000,2008,2011,2012]

### simulate these separate ENSO years

#sort into those years and seasons and put into lists (surprise surprise)
nino_amps = [[amps[i] for i in range(len(phases) - 1) 
               if names_matched[i][1] == part and names_matched[i][0] in nino] 
               for part in seasons]
neutral_amps = [[amps[i] for i in range(len(phases) - 1) 
               if names_matched[i][1] == part and names_matched[i][0] in neutral] 
               for part in seasons]
nina_amps = [[amps[i] for i in range(len(phases) - 1) 
               if names_matched[i][1] == part and names_matched[i][0] in nina] 
               for part in seasons]

#adjust for winter averaging
#TO DO: come up with better procedure rather 
#current: chopping off edges to make the same length for averaging
norml = 359
longl = 363

def padded(to_pad, index):
    length = len(to_pad)
    if index == 0:
        zeros = longl - length
        to_pad = list(to_pad)
        for i in range(zeros):
            to_pad.append(0)
        return to_pad
    else:
        return to_pad

#pad rows with zeros to account for leap year
nino_amps_adj = [[[padded(row, index = i)  
                     for row in entry] 
                  for entry in nino_amps[i]] 
                 for i in range(4)]
nina_amps_adj = [[[padded(row, index = i)  
                     for row in entry] 
                  for entry in nina_amps[i]] 
                 for i in range(4)]
neutral_amps_adj = [[[padded(row, index = i)  
                     for row in entry] 
                     for entry in neutral_amps[i]] 
                    for i in range(4)]

#get averages for each year type
nino_avgs = [[np.average(season, axis = 0)] for season in nino_amps_adj]
nina_avgs = [[np.average(season, axis = 0)] for season in nina_amps_adj]
neutral_avgs = [[np.average(season, axis = 0)] for season in neutral_amps_adj]

avg_lists = [nino_avgs, nina_avgs, neutral_avgs]
name_list = ["nino", "nina", "neutral"]


## generate coeffs and perform ifft

#get function to generate random coeffs
def entry_fft(amp, phase = random.uniform(0, 2*pi)):
    #takes amplitude and phase to give corresponding fourier coeff
    entry = amp*np.exp(1j*phase)
    return entry

#write functions to make a longer ifft
def ext_row(row, n):
    ext_f = np.zeros(((len(row) - 1) * n + 1,), dtype="complex128")
    ext_f[::n] = row * n
    
    return ext_f

def ext_ifft_new(n, input_array):
    #add the zeros onto each end
    ext_f = [ext_row(entry,n) for entry in input_array]
    
    #make up for the formulat multiplying for array length
    olddim = len(input_array[5])
    newdim = len(ext_f[0])
    mult = newdim/olddim
    
    #ext_f = np.multiply(mult, ext_f)
    adjusted_tested = np.fft.ifft2(ext_f)
    
    return adjusted_tested

#write flatten function
flatten = lambda l: [item for sublist in l for item in sublist]

def combined(amps, length):
    #combines generation of random phase with inverse transform
    newarray = [[entry_fft(amp = timed, phase = random.uniform(0, 2*pi)) for timed in wave]
                for wave in amps]
    newarray = [np.array(leaf) for leaf in newarray]
    iffted = ext_ifft_new(length, newarray)
    return iffted

def repeater(season, length, times):
    #repeats the phase creation and inverse transform
    newarray = [combined(season,length) for leaf in range(times)] 
    return(newarray)

def compressed(given):
    #takes a given array and compresses the info for better storage
    compressed_list = []
    for j in range(len(given)):
        compressed_list_j = []
        last_element = None
        
        for i, element in enumerate(given[j]):
            newe = int(np.real(element))
            if newe == last_element: continue
            else:
                compressed_list_j.append((i, newe))
                last_element = newe
            
        compressed_list.append(compressed_list_j)
    
    return compressed_list

def repeater_comp(season, length, times):
    #repeats the phase creation and inverse transform
    newarray = [compressed(combined(season,length)) for leaf in range(times)] 
    return(newarray)

## run ifft
# change values for repeats 

runlen = 100
runtimes = 10
repeattimes = 10

for j in range(3):

    nino_arr = avg_lists[j]
    
    for k in range(repeattimes):
    

        repeated_comp = [repeater_comp(nino_arr[i], runlen, runtimes)
                     for i in range(4)]

    #save the results
    file_name = "randomphase_comp_" + str(k) + "_" + str(name_list[j])
    file_pickle = open(file_name,"wb")
    pickle.dump(repeated_comp, file_pickle)
    file_pickle.close()