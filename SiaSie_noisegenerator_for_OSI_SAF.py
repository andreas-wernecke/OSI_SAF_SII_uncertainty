#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:29:48 2024

@author: Andreas Wernecke


"""


import netCDF4 as nc
import numpy as np
import scipy.signal
#from scipy import stats
#import skgstat as skg
import pandas as pd
pd.options.mode.use_inf_as_na = True
#import xarray as xr
#import pywt
#import warnings
import os, calendar#, glob, datetime, sys
#import random

#home=os.getenv('HOME')
#genpath=home+'/Dropbox/scripts/'
#sys.path.append(genpath)

import cartopy.crs as ccrs
#import cartopy.feature as cf
import matplotlib.pyplot as plt

from datetime import date, datetime, timedelta#, time
from dateutil import rrule#, relativedelta
#import uuid

from copy import copy
import json

#from matplotlib import pylab as plt
#from matplotlib import cm


def find_sic_files(first_day, last_day, area, sources, fn_patt, fn_patt_src):
    files = []
    srcs = []
    # iterate over all days in the month
    #first_day, last_day = get_first_last_date(dt)
    for d in rrule.rrule(rrule.DAILY, dtstart=first_day,
                                        until=last_day):
        # find the path/url to the file. There are precedence rules for what type of files
        #   to select.
        found_one_file = False
        for cdr in ('cdr', 'icdr', 'icdrft'):
            fn = fn_patt.format(a=area, d=d, c=fn_patt_src[cdr])
            fn = os.path.join(sources[cdr],'{:%Y/%m/}'.format(d),fn)
            #print(fn)
            try:
                # this url exists, append it and move to next date
                #ds = xr.open_dataset(fn, decode_times=False)#, decode_times=False
                with nc.Dataset(fn) as ds:
                    found_one_file = True
                    files.append(fn)
                    srcs.append(cdr)
                continue
            except OSError:
                # no valid file at this url, check the next rule
                pass

        # no file found. Add a warning (but we can continue)
        if not found_one_file:
            print("WARNING: could not find OSI SAF SIC v3 file for {} {}; at {}".format(area, d.date(), fn))

    return files, srcs

def read_SIC(interval, years, months, read_dir, data_version, dx, hem, day='all', indirs = None):
    #if indirs is provided, read_dir is not used (not added to sources), but still a required variable

    #interval, years, months, day='all'
    if interval=='day':
        dt = date(years[0], months[0], day[0])
        first_day=dt
        last_day=dt
    if interval=='month':
        dt = date(years[0], months[0], 15)
        first_day=date(years[0], months[0], 1)
        last_day=date(years[-1], months[-1], calendar.monthrange(years[-1], months[-1])[1])

    if hem=='north':
        area='nh'
    elif hem=='south':
        area='sh'

    # handle datestring format and conversion
    if not isinstance(dt, date):
        if isinstance(dt,str):
            if len(dt) == 6:
                dt += '16'
            if len(dt) != 8:
                raise ValueError("Datestring should br YYYYMM or YYYYMMDD")
            try:
                yyyy,mm,dd = int(dt[:4]),int(dt[4:6]),int(dt[6:8])
                dt = date(yyyy,mm,dd)
            except Exception:
                raise ValueError('Invalid datestring {}'.format(dt))

    # check area parameter
    if area not in ('nh', 'sh'):
        raise ValueError('Invalid hemisphere (area={})'.format(area))

    # check if the JSON file for input paths exists (if provided)
    if indirs is not None and not os.path.exists(indirs):
        raise ValueError('Indirs (JSON file with path to input directories) does not exist ({})'.format(indirs))

    # input daily SIC files
    fn_patt = 'ice_conc_{a:}_ease2-250_{c:}_{d:%Y%m%d}1200.nc'
    fn_patt_src = {'cdr': 'cdr-{}'.format(data_version), 'icdr': 'icdr-{}'.format(data_version), 'icdrft': 'icdrft-{}'.format(data_version)}

    if indirs is None:
        # access through THREDDS/OpenDAP
        sources = {#'cdr':'https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_450a_files/',
                  #'cdr':'https://thredds.met.no/thredds/catalog/osisaf/met.no/reprocessed/ice/conc_450a_files/',
                  'cdr' : read_dir[0],
                  'icdr' : read_dir[1],
                  'icdrft' : read_dir[2],
                  #'icdr':'https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_cra_files/',
                  #'icdr':'https://thredds.met.no/thredds/catalog/osisaf/met.no/reprocessed/ice/conc_cra_files/',
                  #'icdrft':'https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_cra_files/'
                  #'icdrft':'https://thredds.met.no/thredds/catalog/osisaf/met.no/reprocessed/ice/conc_cra_files/'
                  }
        #jsond = json.dumps(sources, sort_keys=True, indent=4)
    else:
        # load json file with path to input directories (if the daily SIC files are downloaded to a local disk)
        # an example json file (prepare_monthly_osisaf_sic_opendap.json) is provided to demonstrate the format
        #   expected for the json file (but the effect will be the same as setting indirs to None: read from
        #   THREDDS/opendap)
        with open(indirs, 'r') as f:
            sources = json.load(f)

    files, srcs = find_sic_files(first_day, last_day, area, sources=sources, fn_patt=fn_patt, fn_patt_src=fn_patt_src)
    #ds = xr.open_mfdataset(files,)

    with nc.MFDataset(files,) as ds:
        SIC_trunc=ds.variables['ice_conc'][:,:]
        SIC_raw=ds.variables['raw_ice_conc_values'][:,:]
        try: total_err=ds.variables['total_standard_error'][:,:]
        except: total_err=ds.variables['total_standard_uncertainty'][:,:]
        try: alg_err=ds.variables['algorithm_standard_error'][:,:]
        except: alg_err=ds.variables['algorithm_standard_uncertainty'][:,:]
        try: smear_err=ds.variables['smearing_standard_error'][:,:]
        except:smear_err=ds.variables['smearing_standard_uncertainty'][:,:]
        status=ds.variables['status_flag'][:,:]
        lon=ds.variables['lon'][:,:]
        lat=ds.variables['lat'][:,:]
        SICs_date_seconds = ds.variables['time'][:]
        if ds.variables['time'].units == 'seconds since 1978-01-01 00:00:00':
            SICs_date_t0=datetime(1978, 1, 1, 0, 0, 0)

    SICs=copy(SIC_trunc)
    raw_100_mask = np.array((SIC_trunc==100) * (~SIC_raw.mask))
    SICs[raw_100_mask] = SIC_raw[raw_100_mask]
    #SICs[SIC_raw.mask==False]=SIC_raw[SIC_raw.mask==False]
    SICs[(status & 4) == 4] = SIC_raw[(status & 4) == 4]


    SICs_date=np.zeros_like(SICs_date_seconds, dtype=date)
    for i in range(len(SICs_date_seconds)):
        SICs_date[i]=SICs_date_t0+timedelta(seconds=SICs_date_seconds[i])

    #total_err[total_err.mask]=0.#??

    return lon, lat, SICs_date, status, SICs, total_err, alg_err, smear_err

#ds = xr.open_mfdataset(files,)

# A routine to reconstruct the unfiltered, unthresholded SICs from a OSI SAF SIC CDR file
#def reconstruct_sic(ds):
#    ice_conc = ds['ice_conc'].to_masked_array()
#    raw_ice_conc_values = ds['raw_ice_conc_values'].to_masked_array()
#    status_flag = ds['status_flag'].to_masked_array().astype('short')
#
#   # combine ice_conc with raw_ice_conc_values using the status_flag
#    new_ice_conc = copy(ice_conc)
#    raw_100_mask = np.array((ice_conc==100) * (~raw_ice_conc_values.mask))
#    new_ice_conc[raw_100_mask] = raw_ice_conc_values[raw_100_mask]
#
#    new_ice_conc[(status_flag & 4) == 4] = raw_ice_conc_values[(status_flag & 4) == 4]
#
#    # re-enter "full" ice_conc into the xarray dataset
#    ds['ice_conc'][:] = new_ice_conc
#    return ds

#ds = reconstruct_sic(ds)



def rednoise(size, efold, time='none', useasfilter=False, data='none'):
    '''
    Correlated noise creation using a Gaussisan low-pass filter on white noise

    Parameters
    ----------
    size : ArrayLike
        Size of noise field of length n where n is the number of dimesnsions of the array field (e.g. n=3 for two space and one time dimension)
    efold : np-array
        Array of shape [n] where for each dimension the e-folding length of the correlation is provided
    time : ArrayLike, optional
        1d integer array corresponding to days. The absolute value is irrelavant but gaps in the Timeseries (the time axis is assumed the first axis/entry of size) can be indicated by jumps in 'time'.
        For continous datasets 'none' (the default) can be used.
    v : Bool, optional
        verbos indicator. The default is False.
    useasfilter : Bool, optional
        Switch to use function as a filter on a provided dataset (see data variable) instead of white noise. The default is False.
    data : np-array, optional
        Dataset to be filtered if useasfilter=True. The default is 'none'.
    '''

    if not np.shape(size)[0]==np.shape(efold)[0]:   raise ValueError
    if np.min(efold)<0.:                            raise ValueError
    if useasfilter:
        if np.shape(data)!=size:                    raise ValueError
    if isinstance(time, str):
        if time!='none':                            raise ValueError
        else:  time=np.arange(size[0])
    elif isinstance(time, np.ndarray):
        if np.shape(time)[0]!=size[0]:              raise ValueError
    else:                                           raise ValueError

    if useasfilter:
        if np.max(time)-np.min(time)+1 != size[0]:  raise ValueError
        #timeseries with gaps in record not implemented for filtering

    gaussfac=1./np.sqrt(2.*np.pi)#factor to get efolding to gauss width

    if len(efold) ==3:
        gausswidth=(efold[0]*gaussfac,efold[1]*gaussfac,efold[2]*gaussfac)
    if len(efold) ==2:
        gausswidth=(efold[0]*gaussfac,efold[1]*gaussfac)
    time=time-np.min(time)
    intgausswidth=np.zeros(len(gausswidth), dtype=int)
    for i in range(len(gausswidth)):
        intgausswidth[i]=int(np.ceil(gausswidth)[i])

    if np.shape(size)[0]==3:
        t = np.arange(-3*intgausswidth[0], 3*intgausswidth[0]+1)
        x = np.arange(-3*intgausswidth[1], 3*intgausswidth[1]+1)
        y = np.arange(-3*intgausswidth[2], 3*intgausswidth[2]+1)
        if len(x)==0:x=np.arange(1)
        if len(y)==0:y=np.arange(1)
        if len(t)==0:t=np.arange(1)
        T, X, Y = np.meshgrid(t, x, y, indexing='ij')
        filter_kernel = np.exp(-np.sqrt(X*X/np.max([gausswidth[1]**2,0.0001]) + Y*Y/np.max([gausswidth[2]**2,0.0001]) + T*T/np.max([gausswidth[0]**2,0.0001]) ))
        filter_kernel=filter_kernel/np.sum(filter_kernel)
        #the max() above stabilizes (avoid deviding by zero) but is small enough to have no noteworthy correlation


    noise_ex_tlen=np.max(time)-np.min(time)+1+6*intgausswidth[0]

    # Generate n-by-n-by-nt grid of spatially correlated noise
    if useasfilter==False:

        noise_ex = np.random.randn(noise_ex_tlen,size[1]+6*intgausswidth[1], size[2]+6*intgausswidth[2])
        noise_ex = scipy.signal.fftconvolve(noise_ex, filter_kernel, mode='same')
    else:
        if np.ma.isMaskedArray(data):
            noise_withland=scipy.signal.fftconvolve(data.filled(0), filter_kernel, mode='full')
            padding_impact=1.-scipy.signal.fftconvolve(np.ones_like(data.mask, dtype=float), filter_kernel, mode='full')
            land_impact=scipy.signal.fftconvolve(np.asarray(data.mask, dtype=float), filter_kernel, mode='full')
            tot_impact=land_impact+padding_impact
            noise_withland=np.ma.array(noise_withland, mask=tot_impact>0.999)
            land_impact=np.ma.array(land_impact, mask=tot_impact>0.999)
            noise_ex=noise_withland/(1-tot_impact)
        else:
            noise_ex = scipy.signal.fftconvolve(data, filter_kernel, mode='full')
    if intgausswidth[0]>0: noise_ex=noise_ex[time+3*intgausswidth[0],:,:]
    if intgausswidth[1]>0: noise_ex=noise_ex[:,3*intgausswidth[1]:-3*intgausswidth[1],:]
    if intgausswidth[2]>0: noise_ex=noise_ex[:,:,3*intgausswidth[2]:-3*intgausswidth[2]]
    if useasfilter:
        noise_ex.mask=np.logical_or(data.mask, noise_ex.mask)
        return noise_ex
    else:
        return noise_ex/np.std(noise_ex)


def create_sample(noise_shape, SICs_date, l_scales, data_source='OSI_SAF'):

    timeps=pd.Series({}, index=SICs_date, name='time')
    years=np.unique(timeps.index.year)
    months=np.unique(timeps.index.month)
    days=np.unique(timeps.index.day)

    dateordinal=np.zeros(len(SICs_date), dtype=int)
    for in_sic_date in range(len(SICs_date)):
        dateordinal[in_sic_date]=SICs_date[in_sic_date].toordinal()

    #noise=rednoise(np.shape(SICs), (lcor_temp,lcor_sp,lcor_sp))

    noise=rednoise(noise_shape, l_scales, time=dateordinal)

    #SIC_with_err=SICs_low+SIC_sigts*noise#[dateyear==year]
    ##axnoise.plot(np.sum(np.sum(noise, axis=1), axis=1)*50.*50./1e6)
    #if truncate:
    #    SIC_with_err[SIC_with_err<0.]=0.
    #    SIC_with_err[SIC_with_err>100]=100.

    return noise


def save_daily(daily_savedir, data, batch_number, i, l_scales, dx, hem, data_version, sampledirs=True):

    variables=data.keys()

    for req_var in ['dates', 'sic_with_err']:#'lon', 'lat',
        if req_var not in variables:
            raise ValueError('Please provide the required variable {}'.format(req_var))
    for variable in variables:
        if variable not in ['dates', 'lon', 'lat', 'sic_with_err', 'x', 'y', 'stat', 'noise']:
            raise Warning('{} provided but not used, maybe misspelled?'.format(variable))

    if 'dates' in variables and type(data['dates'][0])==datetime:
        SICs_date=data['dates']
    else: raise ValueError('Please provide the dates in datetime.datetime format')

    if hem=='north':
        xh = 'nh'
    if hem=='south':
        xh='sh'

    timeps=pd.Series({}, index=SICs_date, name='time')
    years=np.unique(timeps.index.year)
    months=np.unique(timeps.index.month)
    days=np.unique(timeps.index.day)

    #print(data['dates'])

    dateordinal=np.zeros(len(SICs_date), dtype=int)
    for in_sic_date in range(len(SICs_date)):
        dateordinal[in_sic_date]=SICs_date[in_sic_date].toordinal()


    #save current sample, but looking as much like CCI data as possible
    #do this only one for a compleate time series since (e.g.) september for trends has jumps im overriding yearly ts
    id_noise = 'batch{:03d}i{:03d}'.format(batch_number,i)

    if sampledirs:
        daily_savedir=os.path.join(daily_savedir, id_noise,'')

        if not os.path.exists(daily_savedir):
            os.mkdir(daily_savedir)

    for year in years:
        savedir_yeartmp=daily_savedir+'{:04d}/'.format(year)
        if not os.path.exists(savedir_yeartmp):
            os.mkdir(savedir_yeartmp)
        for month in months:
            savedir_tmp=daily_savedir+'{:04d}/{:02d}/'.format(year, month)
            if not os.path.exists(savedir_tmp):
                os.mkdir(savedir_tmp)
            for day in days:
                i_time=np.nonzero((timeps.index.year==year) & (timeps.index.month==month) & (timeps.index.day==day))[0]
                if len(i_time)!=1:
                    try:
                        dt_opjekt=datetime(year=year,month=month,day=day)
                    except:
                        print()
                    else:
                        print('WARNING: No data for {:04d}-{:02d}-{:02d}'.format(dt_opjekt.year, dt_opjekt.month, dt_opjekt.day))#warnings.warn
                else:
                    fn='NOISE_{}_lx{:.0f}km_lt{:.0f}d_ice_conc_{}_ease2-{:02.0f}0_cdr-{}_{:04d}{:02d}{:02d}1200.nc'.format(id_noise, l_scales[1]*dx, l_scales[0], xh, dx, data_version, year, month, day)
                    #print(fn)
                    ncds = nc.Dataset(savedir_tmp+fn, 'w', format='NETCDF4_CLASSIC')
                    ncds.description = 'Noisy realisation of OSI SAF SIC data version {} with {:02.1f}km nominal resolution, {} hemisphere -{:04d}{:02d}{:02d}-fv2.1.nc file, accessable from https://climate.esa.int/en/odp/#/project/sea-ice'.format(data_version, dx, hem, year, month, day)

                    dimt, dimy, dimx = np.shape(data['sic_with_err'])
                    timeunits='days since 1900-01-01 00:00:00'
                    # dimensions
                    ncds.createDimension('time', None)
                    ncds.createDimension('x', dimx)
                    ncds.createDimension('y', dimy)

                    # variables
                    #time = ncds.createVariable('time', 'f8', ('time',))
                    if 'x' in variables:
                        x_var = ncds.createVariable('xc', 'f4', ('x','y',))#, fill_value=-9999)
                        x_var.setncatts({'units': u"km"})
                        x_var[:,:] = data['x'][:,:]/1000.

                    if 'y' in variables:
                        y_var = ncds.createVariable('yc', 'f4', ('x','y',))#, fill_value=-9999)
                        y_var.setncatts({'units': u"km"})
                        y_var[:,:] = data['y'][:,:]/1000.

                    if 'lon' in variables:
                        lon_var = ncds.createVariable('lon', 'f4', ('x','y',))#, fill_value=-9999)
                        lon_var.setncatts({'units': u"deg"})
                        lon_var[:,:] = data['lon'][:,:]

                    if 'lat' in variables:
                        lat_var = ncds.createVariable('lat', 'f4', ('x','y',))#, fill_value=-9999)
                        lat_var.setncatts({'units': u"deg"})
                        lat_var[:,:] = data['lat'][:,:]

                    if 'sic_with_err' in variables:
                        sic_var = ncds.createVariable('SIC_sample', 'f4', ('time','x','y',), fill_value=-32767)
                        sic_var.setncatts({'long_name':u"Sea Ice Concentration sample, combining CCI signal and noise realisation. Not truncated to [0,100]"})
                        sic_var.setncatts({'units': u"% SIC"})
                        #sic_var.setncatts({'scale_factor': u"1.0"})
                        sic_var[:,:,:] = data['sic_with_err'].filled(fill_value=-32767.)[i_time[0],:,:].reshape([1,dimx,dimy])[:,:,:]

                    if 'stat' in variables:
                        status_var = ncds.createVariable('status_flag', 'i4', ('time','x','y',))
                        status_var.setncatts({'long_name':u"status flag bit array for sea ice concentration retrievals"})
                        status_var.setncatts({'standard_name':u"sea_ice_area_fraction status_flag"})
                        status_var.setncatts({'flag_masks': u"[  1   2   4   8  16  32  64 128]"})
                        status_var.setncatts({'flag_meanings': u"""land lake open_water_filtered land_spill_over high_t2m spatial_interp temporal_interp max_ice_climo
                                              flag_descriptions:
                                                  all bits to 0 (flag 0): Nominal retrieval by the SIC algorithm
                                                  bit 1 (flag 1): Position is over land
                                                  bit 2 (flag 2): Position is lake
                                                  bit 3 (flag 4): SIC is set to zero by the open water filter
                                                  bit 4 (flag 8): SIC value is changed for correcting land spill-over effects
                                                  bit 5 (flag 16): Handle with caution, the 2m air temperature is high at this position, and this might be false ice
                                                  bit 6 (flag 32): Value is the result of spatial interpolation
                                                  bit 7 (flag 64): Value is the result of temporal interpolation
                                                  bit 8 (flag 128): SIC is set to zero since position is outside maximum sea ice climatology."""})
                        status_var[:,:,:] = data['stat'][i_time[0],:,:].reshape([1,dimx,dimy])[:,:,:]

                    if 'noise' in variables:
                        noise_var = ncds.createVariable('noise', 'f4', ('time','x','y',), fill_value=-32767)
                        noise_var.setncatts({'long_name':u"Sea Ice Concentration noise field, scaled by CCI total uncertainty but excluding the average SIC"})
                        noise_var.setncatts({'units': u"% SIC"})
                        noise_var[:,:,:] = data['noise'].filled(fill_value=-32767.)[i_time[0],:,:].reshape([1,dimy,dimx])[:,:,:]

                    if 'dates' in variables:
                        sic_date_var = ncds.createVariable('time', 'i4', ('time'))
                        sic_date_var.setncatts({'units': timeunits})
                        sic_date_var[:] = nc.date2num(SICs_date[i_time[0]], timeunits)

                    ncds.close()


def fill_unc_interpolation(SICs, SIC_sigts, SIC_algunc, SIC_smearunc, status):

    #I think copying tecnically not necessary, just going save here
    SIC_sigts_new=copy(SIC_sigts)
    SIC_algunc_new=copy(SIC_algunc)
    SIC_smearunc_new=copy(SIC_smearunc)

    #locations with either temporal or spatial interpolation according to flag
    inpol_mask=((status & 32)==32) + ((status & 64)==64)

    #within 10% SIC increments, derive the median for all three uncertainties and fill in where it is missing
    #this might result in total!=alg+smear maybe? Could get total from gap_filled alg and gap_filled smear
    for SIC_range_min, SIC_range_max in zip([-200., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.], [0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 300.]):
        range_mask=(SICs.data >= SIC_range_min) * (SICs.data < SIC_range_max)
        #plt.figure()
        #plt.hist(SICs[range_mask])
        if np.sum(range_mask):#avoid warings
            SIC_sigts_new[np.logical_and(range_mask, inpol_mask)]=np.median(SIC_sigts[range_mask].data[SIC_sigts[range_mask].mask==0])#the strange A.data[A.mask==0] sintax is a workaround to median() ignoring the mask warning
            SIC_algunc_new[np.logical_and(range_mask, inpol_mask)]=np.median(SIC_algunc[range_mask].data[SIC_algunc[range_mask].mask==0])#the strange A.data[A.mask==0] sintax is a workaround to median() ignoring the mask warning
            SIC_smearunc_new[np.logical_and(range_mask, inpol_mask)]=np.median(SIC_smearunc[range_mask].data[SIC_smearunc[range_mask].mask==0])#the strange A.data[A.mask==0] sintax is a workaround to median() ignoring the mask warning
        #print(np.median(SIC_smearunc[range_mask].data[SIC_smearunc[range_mask].mask==0]))

    if 0:#what do we fill up with?
        plt.figure()
        plt.hist(SIC_sigts_new[inpol_mask])

    if abs(np.sum(SIC_sigts_new.mask) - np.sum(SIC_sigts.mask)) != np.sum(inpol_mask):
        print(np.sum(SIC_sigts_new.mask))
        print(np.sum(SIC_sigts.mask))
        print(np.sum(inpol_mask))
        print('We either did not fill all interpolated uncertainties, overdid it somehow')#raise Warning(
    if np.min(SIC_sigts_new)<0. or np.min(SIC_algunc_new)<0. or np.min(SIC_smearunc_new)<0.:
        raise Warning('Nope')
    if np.isnan(SIC_sigts_new).any() or np.isnan(SIC_algunc_new).any() or np.isnan(SIC_smearunc_new).any():
        raise Warning('We filled with NaNs :(')

    return SIC_sigts_new, SIC_algunc_new, SIC_smearunc_new

def create_ensemble(year, month, n_noise, read_dir, save_dir, hem, interval, lcor_temp, lcor_sp_km, data_version, day='all', dx=25, batch_number=1, fill_unc=True, sampledirs=True, apply_weather_filter=True):
    data={}
    if hem=='north':
        ref_crs=ccrs.NorthPolarStereo()
    elif hem=='south':
        ref_crs=ccrs.SouthPolarStereo()

    print('Reading {}-{}'.format(year, month))
    lon, lat, SICs_date, status, SICs, SIC_sigts, SIC_algunc, SIC_smearunc = read_SIC(interval, year, month, read_dir, data_version, dx, hem, day=day)

    if 0:
        plt.figure()
        plt.hist(SICs[(status & 4)==4], bins=40)
        plt.figure()
        plt.hist(SIC_sigts[(status & 4)==4], bins=40)
        print('Mean: {}'.format(np.mean(SIC_sigts[(status & 4)==4])))
        print('Max: {}'.format(np.max(SIC_sigts[(status & 4)==4])))
        print('n>10: {}'.format(np.sum(SIC_sigts[(status & 4)==4]>10.)))
        print('shape: {}'.format(np.shape(SIC_sigts)))


    if fill_unc:
        SIC_sigts, SIC_algunc, SIC_smearunc = fill_unc_interpolation(SICs, SIC_sigts, SIC_algunc, SIC_smearunc, status)

    lcor_sp = lcor_sp_km/dx#in grid cells-50km #mean 50km kerncorr=288km
    l_scales=(lcor_temp,lcor_sp,lcor_sp)

    xyz_cci = ref_crs.transform_points(ccrs.PlateCarree(), lon, lat)
    x=xyz_cci[:,:,0]
    y=xyz_cci[:,:,1]

    #data['x']=x
    #data['y']=y
    #data['lon']=lon
    #data['lat']=lat
    data['dates']=SICs_date
    data['stat']=status

    SICs_low=rednoise(np.shape(SICs), l_scales, useasfilter=True, data=SICs)
    if apply_weather_filter:#Read out the raw SIC for weather filter and set mean to 0 now. Otherwise the smoothing increases SIC again
        SICs_low[(status & 4)==4]=0.

    #fignoise, axnoise=plt.subplots()
    for i in range(n_noise):
        #['dates', 'lon', 'lat', 'sic_with_err', 'x', 'y', 'stat', 'noise']
        print('Creating sample number {} of {}'.format(i+1, n_noise))
        data['noise']=SIC_sigts * create_sample(np.shape(SIC_sigts), SICs_date, l_scales)
        data['sic_with_err']=SICs_low+data['noise']

        save_daily(save_dir, data, batch_number, i, l_scales, dx, hem, data_version, sampledirs=sampledirs)


def create_ensemble_from_file(input_file)  :
    print('Using file {} for input parameters'.format(input_file))
    with open(input_file, 'r') as f:
        par = json.load(f)
    if 'day' not in par.keys(): par['day']="all"
    if 'dx' not in par.keys(): par['dx']=25
    if 'batch_number' not in par.keys(): par['batch_number']=1
    if 'fill_unc' not in par.keys(): par['fill_unc']=True
    if 'sampledirs' not in par.keys(): par['sampledirs']=True
    if 'apply_weather_filter' not in par.keys(): par['apply_weather_filter']=True
    create_ensemble(
            year=par['year'],
            month=par['month'],
            n_noise=par['n_noise'],
            read_dir=par['read_dir'],
            save_dir=par['save_dir'],
            hem=par['hem'],
            interval=par['interval'],
            lcor_temp=par['lcor_temp'],
            lcor_sp_km=par['lcor_sp_km'],
            data_version=par['data_version'],
            day=par['day'],
            dx=par['dx'],
            batch_number=par['batch_number'],
            fill_unc=par['fill_unc'],
            sampledirs=par['sampledirs'],
            apply_weather_filter=par['apply_weather_filter'])


if __name__ == "__main__":

    if 1:

        #read_dir= ['https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_450a_files/',
        #           'https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_450a_files/',
        #           "https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_cra_files/"]
        read_dir= ["/media/dusch/T7 Shield/SIC/SIC/OSI_CDR/SICv30/",
                     "/media/dusch/T7 Shield/SIC/SIC/OSI_CDR/SICv30/",
                     "/media/dusch/T7 Shield/SIC/SIC/OSI_CDR/SICv30/"]
        #Wher do we find the data? The first is searched for 'cdr', second for 'icdr', third for 'icdrft' files

        data_version="v3p0"
        #version of the input data files

        hem="north"
        #hemisphere

        dx=25
        #nominal data resolution in km

        interval="month"
        #is a day or a month (or more) being processed? if =='day': a dataday is required
        years= [1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        #years= [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        #years=[2015]
        #year(s) to process, typically you would give only one at a time (see below)
        #months= np.arange(1,13)
        months=[9]
        #month to process, typically you would give only one at a time (see below)
        dataday="all"
        #month to process, only needed for intervall=='day'
        batch_number=1
        #just a number which is added to the output file names (and directory if sampledirs==True) to avoid overriding earlier ones
        n_noise=20
        #number of samples created by this call. Naming will from 0 to n_noise-1
        save_dir="/media/dusch/T7 Shield/SIC/noise/daily/osi_v3/"
        #location where to save samples to
        lcor_temp = 5.0
        lcor_sp_km= 288.0
        #spatial and themporal correlation parameters. Typically not to be changed.
        sampledirs=True
        #Decide whether each sample gets its only directory in save_dir (called /batchxxxixxx/) or all samples in the same directory, being differenciated only by their names.
        #apply_weather_filter=True
        #this is default now

        #create_ensemble() (and create_ensemble_from_file() can handle periods>one month, but will try to generate the full sample from the first day to the last day at once (where input data is available).
        #This is not practical for periods > one year and unnecessary if there are gaps >> lcor_temp anyways.
        #This is why below we call each year seperately. However, do not do this if you want december and january errors to be correlated (e.g. a seasonal winter uncertainty).
        #If the aim is to get an uncertanty for monthly mean SIA/SIE it is best to call create_ensemble()/create_ensemble_from_file() for each year-month combination seperately
        #If input data is not available for a day (e.g. before 1987) within the processing period, this will be noted ('No file for: yyyymmdd') but sample generation with gaps is implemented.

        #Note that, in particular SIE, is non-linear, therefore the mean of the SIE ensemble will not be the same of the SIE of the mean SIC (=SIC product). Therefore do NOT use mean(SIE(ensemble)) +/- std(SIE(ensemble)) but instead DO use SIE(SIC_450a) +/- std(SIE(ensemble))
        for year in years:
            for month in months:
                create_ensemble([year], [month], n_noise, read_dir, save_dir, hem, interval, lcor_temp, lcor_sp_km, data_version, day=dataday, dx=25, batch_number=batch_number, sampledirs=sampledirs)

    else:
        #same as above but using a file instead of parameter definition in script.

        input_fn='/media/dusch/T7 Shield/SIC/SIC/noise/test2/input.json'
        #dt0=datetime.now()
        create_ensemble_from_file(input_fn)

        #dt1=datetime.now()

        #print(dt1-dt0)





