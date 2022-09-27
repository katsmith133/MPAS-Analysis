#!/usr/bin/env python
# This software is open source software available under the BSD-3 license.
#
# Copyright (c) 2022 Triad National Security, LLC. All rights reserved.
# Copyright (c) 2022 Lawrence Livermore National Security, LLC. All rights
# reserved.
# Copyright (c) 2022 UT-Battelle, LLC. All rights reserved.
#
# Additional copyright and license information can be found in the LICENSE file
# distributed with this code, or at
# https://raw.githubusercontent.com/MPAS-Dev/MPAS-Analysis/master/LICENSE

"""
A script for downloading and preprocessing data sets from ASTE for use in
MPAS-Analysis
"""
# Authors
# -------
# Kat Smith

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import xarray as xr
import os
import argparse

from mpas_analysis.shared.io.download import download_files

def concat_data3D(urlBase, varIn, inRegions, inDir):
    """
    Load in all monthly climatology and process for MPAS-Analysis.
    """
    fileList = []
    for region in inRegions:
        fileList.append('{}/{}.{}.nc'.format(varIn,varIn,region))
        
    download_files(fileList, urlBase, inDir)

    varOut  = np.zeros((12,50,90*29,90*29))
    
    i = 0
    for region in inRegions:
        ds  = xr.open_mfdataset('{}/{}/{}.{}.nc'.format(inDir,varIn,varIn,region))
        varOut[:,:,i*90:(i+1)*90,i*90:(i+1)*90] = ds.variables[varIn][:,:,:,:]

    return varOut

def concat_data2D(urlBase, varIn, inRegions, inDir):
    """
    Load in all monthly climatology and process for MPAS-Analysis.
    """
    fileList = []
    for region in inRegions:
        fileList.append('{}/{}.{}.nc'.format(varIn,varIn,region))
        
    download_files(fileList, urlBase, inDir)

    varOut  = np.zeros((12,90*29,90*29))
    
    i = 0
    for region in inRegions:
        ds  = xr.open_mfdataset('{}/{}/{}.{}.nc'.format(inDir,varIn,varIn,region))
        varOut[:,i*90:(i+1)*90,i*90:(i+1)*90] = ds.variables[varIn][:,:,:]

    return varOut

def grab_grid(urlBase, inRegions, varIn, inDir):
    """
    Load in all monthly climatology and process for MPAS-Analysis.
    """
    fileList = []
    for region in inRegions:
        fileList.append('{}/{}.{}.nc'.format(varIn,varIn,region))
        
    download_files(fileList, urlBase, inDir)

    lat     = np.zeros((90*29))
    lon     = np.zeros((90*29))
    
    i = 0
    for region in inRegions:
        ds  = xr.open_mfdataset('{}/{}/{}.{}.nc'.format(inDir,varIn,varIn,region))
        lat[i*90:(i+1)*90] = ds.variables['lat'][:,0]
        lon[i*90:(i+1)*90] = ds.variables['lon'][0,:]

    z = ds.variables['dep'][:]

    return lat, lon, z
    

def grab_grid_angles(urlBase, inRegions, inDir):
    """
    Load in all monthly climatology and process for MPAS-Analysis.
    """
    gridList = []
    for region in inRegions:
        gridList.append('/{}.{}.nc'.format('GRID',region))
        
    download_files(gridList, urlBase, inDir)

    AngleCS = np.zeros((90*29,90*29))
    AngleSN = np.zeros((90*29,90*29))
    
    i = 0
    for region in inRegions:
        ds  = xr.open_mfdataset('{}/{}.{}.nc'.format(inDir,'GRID',region))
        AngleCS[i*90:(i+1)*90,i*90:(i+1)*90] = ds.variables['AngleCS'][:,:]
        AngleSN[i*90:(i+1)*90,i*90:(i+1)*90] = ds.variables['AngleSN'][:,:]

    return AngleCS, AngleSN


def process_velocities(uIn, vIn, AngleCS, AngleSN):
    """
    Load in all monthly climatology and process for MPAS-Analysis.
    """

    uCenter = np.zeros_like((uIn))
    vCenter = np.zeros_like((uIn))
    uOut    = np.zeros_like((uIn))
    vOut    = np.zeros_like((uIn))

    for ix in range(0,90*29-1):
        uCenter[:,:,ix,:] = (uIn[:,:,ix,:]+uIn[:,:,ix+1,:])/2
        vCenter[:,:,:,ix] = (vIn[:,:,:,ix]+vIn[:,:,:,ix+1])/2
        
    for it in range(0,12):
        for iz in range(0,50):
            uOut[it,iz,:,:] = uCenter[it,iz,:,:]*AngleCS[:,:] - vCenter[it,iz,:,:]*AngleSN[:,:]
            vOut[it,iz,:,:] = uCenter[it,iz,:,:]*AngleSN[:,:] + vCenter[it,iz,:,:]*AngleCS[:,:]

    return uOut, vOut
        

def output_data3D(outDir, varOut, lat, lon, z, varOutName, varOutUnits):
    """
    Load in all monthly climatology and process for MPAS-Analysis.
    """

    outFileName = '{}/{}.nc'.format(outDir,varOutName)
    
    if os.path.exists(outDir):
        ds = xr.open_dataset(outFileName)
    else:
        print("Saving ASTE {} data...".format(varOutName))
        field = varOut
        
        description = 'Monthly {} climatologies from ' \
                      '2002-2017 average of the Arctic Subpolar ' \
                      'gyre sTate Estimate (ASTE)'.format(varOutName)
        
        dictonary = {'dims': ['Time', 'depth', 'lat', 'lon'],
                     'coords': {'month': {'dims': ('Time'),
                                          'data': range(1, 13),
                                          'attrs': {'units': 'months'}},
                                'year': {'dims': ('Time'),
                                         'data': numpy.ones(12),
                                         'attrs': {'units': 'years'}},
                                'depth': {'dims': ('depth'),
                                          'data': z,
                                          'attrs': {'units': 'meters'}},
                                'lon': {'dims': ('lon'),
                                        'data': lon,
                                        'attrs': {'units': 'degrees'}},
                                'lat': {'dims': ('lat'),
                                        'data': lat,
                                        'attrs': {'units': 'degrees'}}},
                     'data_vars': {varOutName:
                                   {'dims': ('Time', 'depth', 'lat', 'lon'),
                                    'data': field,
                                    'attrs': {'units': varOutUnits,
                                              'description': description}}}}
        
        ds = xr.Dataset.from_dict(dictonary)
        write_netcdf(ds, outFileName)

    return ds

def output_data2D(outDir, varOut, lat, lon, varOutName, varOutUnits):
    """
    Load in all monthly climatology and process for MPAS-Analysis.
    """

    outFileName = '{}/{}.nc'.format(outDir,varOutName)
    
    if os.path.exists(outDir):
        ds = xr.open_dataset(outFileName)
    else:
        print("Saving ASTE {} data...".format(varOutName))
        field = varOut
        
        description = 'Monthly {} climatologies from ' \
                      '2002-2017 average of the Arctic Subpolar ' \
                      'gyre sTate Estimate (ASTE)'.format(varOutName)
        
        dictonary = {'dims': ['Time', 'lat', 'lon'],
                     'coords': {'month': {'dims': ('Time'),
                                          'data': range(1, 13),
                                          'attrs': {'units': 'months'}},
                                'year': {'dims': ('Time'),
                                         'data': numpy.ones(12),
                                         'attrs': {'units': 'years'}},
                                'lon': {'dims': ('lon'),
                                        'data': lon,
                                        'attrs': {'units': 'degrees'}},
                                'lat': {'dims': ('lat'),
                                        'data': lat,
                                        'attrs': {'units': 'degrees'}}},
                     'data_vars': {varOutName:
                                   {'dims': ('Time', 'depth', 'lat', 'lon'),
                                    'data': field,
                                    'attrs': {'units': varOutUnits,
                                              'description': description}}}}
    
        ds = xr.Dataset.from_dict(dictonary)
        write_netcdf(ds, outFileName)

    return ds
        

def main():
    # Grab input and output directories from command line
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--inDir", dest="inDir", required=True,
                        help="Directory where intermediate files used in "
                             "processing should be downloaded")
    parser.add_argument("-o", "--outDir", dest="outDir", required=True,
                        help="Directory where final preprocessed observation "
                             "are stored")
    args = parser.parse_args()
                     
    # Decide what ASTE fields we want to grab and process
    varibles    = ['THETA','SALT','UVELMASS','VVELMASS','ETAN']

    # ASTE data is saved in 29 region "tiles" 
    inRegions = ['0001','0002','0003','0004','0005',\
                     '0006','0007','0008','0009','0010',\
                     '0011','0012','0013','0014','0015',\
                     '0016','0017','0018','0019','0020',\
                     '0021','0022','0023','0024','0025',\
                     '0026','0027','0028','0029']

    # Make input and output directories
    try:
        os.makedirs(args.inDir)
    except OSError:
        pass

    try:
        os.makedirs(args.outDir)
    except OSError:
        pass
        
    # Download and concatenate the 29 "tiles" for each field
    urlBase = 'https://arcticdata.io/data/10.18739/A2CV4BS5K/nctiles_climatology/'
    for prefix in varibles:
        if prefix == 'THETA':
            pt = concat_data3D(urlBase, prefix, inRegions, args.inDir)
        elif prefix == 'SALT':
            sa = concat_data3D(urlBase, prefix, inRegions, args.inDir)
        elif prefix == 'UVELMASS':
            uIn = concat_data3D(urlBase, prefix, inRegions, args.inDir)
        elif prefix == 'VVELMASS':
            vIn = concat_data3D(urlBase, prefix, inRegions, args.inDir)
        elif prefix == 'ETAN':
            SSH = concat_data2D(urlBase, prefix, inRegions, args.inDir)

    # Download and concatenate the 29 "tiles" of lat, lon, and z grid dimentions 
    lat, lon, z = grab_grid(urlBase, inRegions, 'THETA', args.inDir)

    # Download and concatenate the 29 "tiles" of angles to calculate NS-EW velocities
    urlBase = 'https://arcticdata.io/data/10.18739/A2CV4BS5K/nctiles_grid/'
    AngleCS, AngleSN = grab_grid_angles(urlBase, inRegions, args.inDir)

    # Calculate N-S, E-W oriented velocities
    u, v = process_velocities(uIn, vIn, AngleCS, AngleSN)

    # Write processed fields to netcdf files 
    for prefix in varibles:
        if prefix == 'THETA':
            print("Writing PT...")
            output_data3D(args.outDir, pt, lat, lon, z, 'Temperature','C')
        if prefix == 'SALT':
            print("Writing SA...")
            output_data3D(args.outDir, sa, lat, lon, z, 'Salinity','psu')
        elif prefix == 'UVELMASS':
            print("Writing U...")
            output_data3D(args.outDir, u, lat, lon, z, 'uVel','m/s')
        elif prefix == 'VVELMASS':
            print("Writing V...")
            output_data3D(args.outDir, v, lat, lon, z, 'vVel','m/s')
        elif prefix == 'ETAN':
            print("Writing SSH...")
            output_data2D(args.outDir, SSH, lat, lon, z, 'SSH','m')


if __name__ == "__main__":
    main()
