# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:51:13 2017

@author: Elias Ayrey

Script for creating 'voxels' out of a LiDAR point cloud plot. This essentially just creates 3D bins and 
counts the number of points inside of those bins.

This script requires normalized and tiled LiDAR data. The elevation of the ground must be equal to zero. 
You must read in a LAS file that is a single tile, as in an area that is 10x10 m, 20x20 m, ect..
"""
import numpy as np
import laspy
import os
from glob import glob
from PIL import Image
import pandas as pd

#Directory or directories storing the tiled LiDAR files (LAS files)
directories=["G:\\1_19_2017_Combination\\Null_Plots\\"]

#Populate a list of LAS files in that directory
pattern="*.las"
files=[]
for d in range(len(directories)):
    dir=directories[d]
    files.extend(glob(os.path.join(dir,pattern)))
    
#Simple function to weed out duplicate LiDAR points.
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

#Populate a list of matricies which represent voxels.    
matrices=[]
Plots=[]
for file in files:
    #Read in LAS file and extract coordinates of every point into an array
    infile=laspy.file.File(file, mode='r')
    coords=np.vstack((infile.x,infile.y,infile.z)).transpose()
    coords=unique_rows(coords)
    
    #extract min and max values of points
    mins=[min(coords[:,0]), min(coords[:,1]), min(coords[:,2])]
    maxs=[max(coords[:,0]), max(coords[:,1]), max(coords[:,2])]
    
    #creates 40 bins on the XY axis, and 105 bins on the Z axis for each unbuffered plot up to 35m in the air
    #with a plot size of 10, those are 25x25x30 cm voxels
    num_bins=40
    num_vert_bins=105
    plt_size=10
    H, edges=np.histogramdd(coords, bins=(np.linspace(mins[0]-.001,mins[0]+plt_size,num_bins+1),np.linspace(mins[1]-.001,mins[1]+plt_size,num_bins+1), np.linspace(0,35,num_vert_bins+1)))
    
    ###OPTIONAL###########################
    #Some neural networks perform better with normalized input data these are two ways to normalize the voxels
    #NOTE, neither way produced better results for me than the unnormalized voxels
    ######################################
    #Convert any voxel with a point in it to a value of 1 (creates binary voxels)
    #H=np.where(H!=0,H/np.nanmax(np.where(H!=0,H,np.nan)),0)
    #H=np.where(H!=0,1,0)
    
    #Z-score the whole thing. Which doesn't make much sense given the zero-inflation
    #H=H-np.mean(H, axis=(0,1,2))
    #H=H/np.std(H, axis=(0,1,2))
    
    matrices.append(H)
    dir_info=file.split('\\')
    Plots.append(dir_info[-1][:-4])
    
matrices=np.array(matrices)
   
#save these as a numpy array to be used with a tensorflow neural network   
np.save('H:\\Temp\\Chapter_2_bits\\LiDAR\\Leaf_off.npy', matrices)
#save the plot names
files=pd.DataFrame(files)
files.to_csv('H:\\Temp\\Chapter_2_bits\\LiDAR\\Leaf_offL_files.csv')
    