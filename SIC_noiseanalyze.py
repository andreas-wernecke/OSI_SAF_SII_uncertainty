#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:07:28 2024

@author: dusch
"""

#import gdal
import netCDF4 as nc
import numpy as np
#import pandas as pd
from scipy import stats, odr
#import skgstat as skg
#from scipy.interpolate import griddata
#from scipy.signal import detrend
import pandas as pd
pd.options.mode.use_inf_as_na = True
#import xarray as xr
from dateutil import rrule#, relativedelta

import os, calendar, csv#, glob#, datetime
#from os import path

import geopandas, shapely, shapely.vectorized, warnings
import matplotlib.pyplot as plt

from datetime import date, timedelta, datetime
from copy import copy
from sklearn.linear_model import LinearRegression


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

def read_SIC_osi(first_day, last_day, read_dir, fn_patt, data_version, dx, area):
    #if indirs is provided, read_dir is not used (not added to sources), but still a required variable

    # check area parameter
    if area not in ('nh', 'sh'):
        raise ValueError('Invalid hemisphere (area={})'.format(area))


    fn_patt_src = {'cdr': 'cdr-{}'.format(data_version), 'icdr': 'icdr-{}'.format(data_version), 'icdrft': 'icdrft-{}'.format(data_version)}


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
        lon=ds.variables['lon'][:,:]
        lat=ds.variables['lat'][:,:]
        status=ds.variables['status_flag'][:,:]
        SICs_date_seconds = ds.variables['time'][:]
        if ds.variables['time'].units == 'seconds since 1978-01-01 00:00:00':
            SICs_date_t0=datetime(1978, 1, 1, 0, 0, 0)
            SICs_date=np.zeros_like(SICs_date_seconds, dtype=date)
            for i in range(len(SICs_date_seconds)):
                SICs_date[i]=SICs_date_t0+timedelta(seconds=SICs_date_seconds[i])
        else:
            print(ds.variables['time'].units)
    SICs=copy(SIC_trunc)
    raw_100_mask = np.array((SIC_trunc==100) * (~SIC_raw.mask))
    SICs[raw_100_mask] = SIC_raw[raw_100_mask]
    #SICs[SIC_raw.mask==False]=SIC_raw[SIC_raw.mask==False]
    SICs[(status & 4) == 4] = SIC_raw[(status & 4) == 4]

    return SICs_date, status, SICs

def read_SIC_noise(first_day, last_day, read_dir, fn_patt, data_version, dx, area):
    #if indirs is provided, read_dir is not used (not added to sources), but still a required variable

    # check area parameter
    if area not in ('nh', 'sh'):
        raise ValueError('Invalid hemisphere (area={})'.format(area))
    fn_patt_src = {'cdr': 'cdr-{}'.format(data_version), 'icdr': 'icdr-{}'.format(data_version), 'icdrft': 'icdrft-{}'.format(data_version)}

    sources = {'cdr' : read_dir[0],
              'icdr' : read_dir[1],
              'icdrft' : read_dir[2],
              }

    files, srcs = find_sic_files(first_day, last_day, area, sources=sources, fn_patt=fn_patt, fn_patt_src=fn_patt_src)
    #ds = xr.open_mfdataset(files,)

    with nc.MFDataset(files,) as ds:
        SIC_trunc=ds.variables['SIC_sample'][:,:]
        status=ds.variables['status_flag'][:,:]
        SICs_date_seconds = ds.variables['time'][:]
        if 0:#only to test what the filtered values look like
            noise=ds.variables['noise'][:,:]
            SIC_trunc=SIC_trunc-noise
        if ds.variables['time'].units == 'seconds since 1978-01-01 00:00:00':
            SICs_date_t0=datetime(1978, 1, 1, 0, 0, 0)
            SICs_date=np.zeros_like(SICs_date_seconds, dtype=date)
            for i in range(len(SICs_date_seconds)):
                SICs_date[i]=SICs_date_t0+timedelta(seconds=SICs_date_seconds[i])

        elif ds.variables['time'].units=='days since 1900-01-01 00:00:00':
            SICs_date_t0=datetime(1900, 1, 1, 0, 0, 0)
            SICs_date=np.zeros_like(SICs_date_seconds, dtype=date)
            for i in range(len(SICs_date_seconds)):
                SICs_date[i]=SICs_date_t0+timedelta(days=int(SICs_date_seconds[i]))
        else:
            print(ds.variables['time'].units)
        SICs=copy(SIC_trunc)

    return SICs_date, status, SICs






#---------------------------------------------------------------------------------------------------
###reading in CCI SIC



XH='NH'
testmonth=9

comp_cci_ens=False
#data='CCI'
#data='CCI_filt'
data_noise=1
data_osi=1
data_SIAUHH=0
data_SIEUHH=0

n_noise=10
batches=[1]#,2,3,4,5,6,7]

sampledirs=True

regional=False

sic_owf_zero=False

center_ens=True

load = 1
save = 0


#years=[2015]
#np.arange(2010, 2024)
years=np.arange(1979, 2025)
#months=[10]
months=np.arange(1,13)




#---------------------------------------------------------------------------------------------------
lcor_temp = 5.
lcor_sp_km =288.
dx=25.#in km, 50km nominal, corrected for typical arctic ocean (instead of NP)



if XH=='NH':
    NorthSouth='north'
    xh='nh'
elif XH=='SH':
    NorthSouth='south'
    xh='sh'

if data_osi:
    read_dir= "/media/dusch/T7 Shield/SIC/SIC/OSI_CDR/SICv30/"
    data_version="v3p0"
    fn_patt = 'ice_conc_{a:}_ease2-250_{c:}_{d:%Y%m%d}1200.nc'

    if not load:

        for year in years:
            print('OSI SAF [{}]'.format(year))
            first_day=date(year, months[0], 1)
            last_day=date(year, months[-1], calendar.monthrange(years[-1], months[-1])[1])

            sources= [read_dir,read_dir,read_dir]
            try:
                SICs_date, status, SICs = read_SIC_osi(first_day, last_day, sources, fn_patt, data_version, dx, xh)
                #SICs_date, status, SICs = read_SIC_noise(first_day, last_day, sources, fn_patt, data_version, dx, xh)
            except:
                continue

            SIA_tmp=np.sum(np.sum(SICs, axis=1), axis=1)*dx*dx/100.
            SIE_tmp=np.sum(np.sum(SICs>15., axis=1), axis=1)*dx*dx

            if 0:#plot specific date
                if year==2016:
                    SICs.mask=np.logical_or(SICs.mask, (status[257,:,:] & 4) !=4)
                    plt.figure()
                    plt.imshow(SICs[257,:,:])
                    plt.colorbar()

                    plt.figure()
                    plt.imshow((status[257,:,:] & 4) ==4)


            if 'df_SIA_osi' in globals() and 'df_SIE_osi' in globals():
                df_SIA_osi=df_SIA_osi.combine_first(pd.DataFrame({'OSI_SAF':SIA_tmp.data}, index=SICs_date))
                df_SIE_osi=df_SIE_osi.combine_first(pd.DataFrame({'OSI_SAF':SIE_tmp}, index=SICs_date))
            else:
                d={'OSI_SAF':pd.Series(SIA_tmp, index=SICs_date)}
                df_SIA_osi=pd.DataFrame(data=d, index=SICs_date)
                d={'OSI_SAF':pd.Series(SIE_tmp, index=SICs_date)}
                df_SIE_osi=pd.DataFrame(data=d, index=SICs_date)
        if save:
            df_SIA_osi.to_csv(read_dir+'SIA_{}.csv'.format(XH))
            df_SIE_osi.to_csv(read_dir+'SIE_{}.csv'.format(XH))
    if load:
        df_SIA_osi=pd.read_csv(read_dir+'SIA_{}.csv'.format(XH), index_col=0)
        df_SIE_osi=pd.read_csv(read_dir+'SIE_{}.csv'.format(XH), index_col=0)

    df_SIA_osi=df_SIA_osi/1e6
    df_SIE_osi=df_SIE_osi/1e6

    df_SIA_osi['time']=pd.DatetimeIndex(df_SIA_osi.index)
    df_SIE_osi['time']=pd.DatetimeIndex(df_SIE_osi.index)
    df_SIA_osi.set_index('time', inplace=True)
    df_SIE_osi.set_index('time', inplace=True)

    df_SIA_osi_month=df_SIA_osi.groupby(pd.PeriodIndex(df_SIA_osi.index, freq="M"))['OSI_SAF'].mean()
    df_SIA_osi_month = df_SIA_osi_month.reset_index()
    df_SIA_osi_month['time'] = pd.DatetimeIndex(data=df_SIA_osi_month['time'].apply(lambda x: x.strftime('%Y-%m-15')))
    df_SIA_osi_month.set_index('time', inplace=True)

    df_SIE_osi_month=df_SIE_osi.groupby(pd.PeriodIndex(df_SIE_osi.index, freq="M"))['OSI_SAF'].mean()
    df_SIE_osi_month = df_SIE_osi_month.reset_index()
    df_SIE_osi_month['time'] = pd.DatetimeIndex(data=df_SIE_osi_month['time'].apply(lambda x: x.strftime('%Y-%m-15')))
    df_SIE_osi_month.set_index('time', inplace=True)

    #paths=['/media/dusch/T7 Shield/SIC/noise/daily/{}/osi_sep_full/'.format(XH)]


if data_noise:
    read_dir_og='/media/dusch/T7 Shield/SIC/noise/daily/osi_v3/batch{:03d}i{:03d}/'
    save_dir='/media/dusch/T7 Shield/SIC/noise/daily/osi_v3/'
    fn_patt_og='NOISE_batch{bb:03d}i{ii:03d}_lx{lx:.0f}km_lt{lt:.0f}d_ice_conc_{a:}_ease2-250_{c:}_{d:}1200.nc'
    data_version="v3p0"
    if not load:
        for i_batch in batches:
            for i_noise in np.arange(n_noise):
                batch_sample='batch{:03d}i{:03d}'.format(i_batch, i_noise)
                print(batch_sample)

                for year in years:
                    first_day=date(year, months[0], 1)
                    last_day=date(year, months[-1], calendar.monthrange(years[-1], months[-1])[1])

                    read_dir=read_dir_og.format(i_batch,i_noise)
                    sources= [read_dir,read_dir,read_dir]
                    fn_patt=fn_patt_og.format(bb=i_batch, ii=i_noise, lx=lcor_sp_km, lt=lcor_temp, a='{a:}', c='{c:}', d='{d:%Y%m%d}')

                    try:
                        SICs_date, status, SICs = read_SIC_noise(first_day, last_day, sources, fn_patt, data_version, dx, xh)
                    except:
                        continue

                    SIA_tmp=np.sum(np.sum(SICs, axis=1), axis=1)*dx*dx/100.
                    SIE_tmp=np.sum(np.sum(SICs>15., axis=1), axis=1)*dx*dx

                    if 0:#plot specific date
                        if year==2016:
                            SICs.mask=np.logical_or(SICs.mask, (status[257,:,:] & 4) !=4)
                            plt.figure()
                            plt.imshow(SICs[257,:,:])
                            plt.colorbar()

                            plt.figure()
                            plt.imshow((status[257,:,:] & 4) ==4)




                    if 'df_SIA' in globals() and 'df_SIE' in globals():
                        if (batch_sample in df_SIA.keys()) and (batch_sample in df_SIE.keys()):
                            df_SIA=df_SIA.combine_first(pd.DataFrame({batch_sample:SIA_tmp.data}, index=SICs_date))
                            df_SIE=df_SIE.combine_first(pd.DataFrame({batch_sample:SIE_tmp}, index=SICs_date))
                        else:
                            df_SIA[batch_sample]=pd.Series(SIA_tmp, index=SICs_date)
                            df_SIE[batch_sample]=pd.Series(SIE_tmp, index=SICs_date)
                    else:
                        d={batch_sample:pd.Series(SIA_tmp, index=SICs_date)}
                        df_SIA=pd.DataFrame(data=d, index=SICs_date)
                        d={batch_sample:pd.Series(SIE_tmp, index=SICs_date)}
                        df_SIE=pd.DataFrame(data=d, index=SICs_date)
        if save:
            df_SIA.to_csv(save_dir+'SIA_{}.csv'.format(XH))
            df_SIE.to_csv(save_dir+'SIE_{}.csv'.format(XH))
    if load:
        df_SIA=pd.read_csv(save_dir+'SIA_{}.csv'.format(XH), index_col=0)
        df_SIE=pd.read_csv(save_dir+'SIE_{}.csv'.format(XH), index_col=0)
    #make sure index is of type datetimeindex
    df_SIA['time']=pd.DatetimeIndex(df_SIA.index)
    df_SIE['time']=pd.DatetimeIndex(df_SIE.index)
    df_SIA.set_index('time', inplace=True)
    df_SIE.set_index('time', inplace=True)

    df_SIA=df_SIA/1e6
    df_SIE=df_SIE/1e6

    df_SIA_month=df_SIA.groupby(pd.PeriodIndex(df_SIA.index, freq="M")).mean()
    df_SIA_month = df_SIA_month.reset_index()
    df_SIA_month['time'] = pd.DatetimeIndex(data=df_SIA_month['time'].apply(lambda x: x.strftime('%Y-%m-15')))
    df_SIA_month.set_index('time', inplace=True)

    df_SIE_month=df_SIE.groupby(pd.PeriodIndex(df_SIE.index, freq="M")).mean()
    df_SIE_month = df_SIE_month.reset_index()
    df_SIE_month['time'] = pd.DatetimeIndex(data=df_SIE_month['time'].apply(lambda x: x.strftime('%Y-%m-15')))
    df_SIE_month.set_index('time', inplace=True)

if data_SIAUHH:
    areadir='/mnt/icdc/ice_and_snow/uhh_seaiceareatimeseries/DATA/'
    if XH=='NH': areafn='SeaIceArea__NorthernHemisphere__monthly__UHH__v2024_fv0.01.nc'
    else:        areafn='SeaIceArea__SouthernHemisphere__monthly__UHH__v2024_fv0.01.nc'
    ds=nc.Dataset(areadir+areafn, "r")
    SIAUHH_time=ds.variables['time'][:]
    SIA_osi=ds.variables['osisaf'][:]
    SIA_esa=ds.variables['esa'][:]
    SIA_hado=ds.variables['HadISST_orig'][:]
    SIA_had_nsidc=ds.variables['HadISST_nsidc'][:]
    SIA_nt=ds.variables['nsidc_nt'][:]
    SIA_bt=ds.variables['nsidc_bt'][:]
    ds.close()

    start_time=date(1849,12,31)#-1 day to hi the 15th of the month
    time_tmp=np.asarray([start_time+timedelta(SIAUHH_time.data[i]) for i in np.nonzero(np.isnan(SIAUHH_time.data)==0)[0]])
    SIAUHH_time=pd.DatetimeIndex(data=time_tmp)

    df_SIAUHH=pd.DataFrame(data={'HadISST':pd.Series(SIA_hado, index=SIAUHH_time)},  index=SIAUHH_time)
    df_SIAUHH['HadISST_nsidc']=pd.Series(SIA_had_nsidc, index=SIAUHH_time)
    df_SIAUHH['OSI_SAF']=pd.Series(SIA_osi, index=SIAUHH_time)
    df_SIAUHH['ESA_CCI']=pd.Series(SIA_esa, index=SIAUHH_time)
    df_SIAUHH['NSIDC_NT']=pd.Series(SIA_nt, index=SIAUHH_time)
    df_SIAUHH['NSIDC_BT']=pd.Series(SIA_bt, index=SIAUHH_time)
    #df_SIAUHH['WALSH']=pd.Series(SIA_walsh, index=SIAUHH_time)

if data_SIEUHH:
    #UHH files
    extentdir='/media/dusch/T7 Shield/SIC/SIC/SIE/'
    if XH=='NH': extentfn='SIE_observations_nh_v2023_fv0.01.nc'#HadISST SIE from quentin (like icdc SIA)
    else:        extentfn='SIE_observations_sh_v2023_fv0.01.nc'
    ds=nc.Dataset(extentdir+extentfn, "r")
    SIEUHH_time=ds.variables['time'][:]
    SIE_hado=ds.variables['HadISST_orig'][:]
    SIE_hadnsidc=ds.variables['HadISST_nsidc'][:]
    ds.close()
    start_time=date(1850,1,15)
    time_tmp=np.asarray([start_time+timedelta(SIEUHH_time.data[i]) for i in np.nonzero(np.isnan(SIEUHH_time.data)==0)[0]])
    SIEUHH_time=pd.DatetimeIndex(data=time_tmp)

    df_SIEUHH=pd.DataFrame(data={'HadISST':pd.Series(SIE_hado, index=SIEUHH_time)},  index=SIEUHH_time)
    df_SIEUHH['HadISST_nsidc']=pd.Series(SIE_hadnsidc, index=SIEUHH_time)
    #df_SIEUHH['NSIDC_NT']=pd.Series(SIA_nt, index=SIEUHH_time)
    #df_SIEUHH['WALSH']=pd.Series(SIA_walsh, index=SIEUHH_time)

    #OSI SAF SIE
    if XH=='NH': extentfn='osisaf_nh_sie_monthly.nc'#https://osisaf-hl.met.no/v2p1-sea-ice-index
    else:        extentfn='osisaf_sh_sie_monthly.nc'

    with nc.Dataset(extentdir+extentfn, "r") as ds:
        SIE_osisii=ds.variables['sie'][:]
        time_osisii=ds.variables['time'][:]
        if not ds.variables['time'].units=='days since 1970-01-01 00:00:00': raise ValueError

    start_time=date(1969,12,31)
    time_tmp=np.asarray([start_time+timedelta(int(time_osisii.data[i])) for i in np.nonzero(np.isnan(time_osisii.data)==0)[0]])
    SIEUHH_time=time_tmp

    df_SIEUHH['OSI_SAF']=pd.Series(SIE_osisii, index=SIEUHH_time)

    #NOAA files
    if XH=='NH': extentfn='N_seaice_extent_daily_v3.0.csv'#NOAA SIE from https://noaadata.apps.nsidc.org/NOAA/G02135
    else:        extentfn='S_seaice_extent_daily_v3.0.csv'
    df_SIE_NOAA=pd.read_csv(extentdir+extentfn)
    df_SIE_NOAA.drop(0, inplace=True)


    NOAA_time=np.asarray([date(year=int(df_SIE_NOAA['Year'][i]), month=int(df_SIE_NOAA[' Month'][i]), day=int(df_SIE_NOAA[' Day'][i])) for i in range(1,len(df_SIE_NOAA)+1)])

    #pd.concat([df_SIE_NOAA, pd.Series(data=SIEUHH_time, name='time_NOAA')])

    df_SIE_NOAA['time']=pd.DatetimeIndex(data=NOAA_time)
    #df_SIE_NOAA['Date']=pd.Series(data=NOAA_time, index=df_SIE_NOAA.index)
    df_SIE_NOAA.set_index('time', inplace=True)
    df_SIE_NOAA.pop('Year')
    df_SIE_NOAA.pop(' Month')
    df_SIE_NOAA.pop(' Day')
    df_SIE_NOAA.pop(' Source Data')
    df_SIE_NOAA.pop('    Missing')
    #df_SIE_NOAA.rename(columns={"     Extent":"NOAA"}, inplace=True)
    df_SIE_NOAA['NOAA']=df_SIE_NOAA["     Extent"].astype({"     Extent":"float"})
    df_SIE_NOAA.pop("     Extent")

    #pd.PeriodIndex(df_SIE_NOAA['Date'], freq='M')
    #df_SIEUHH['NOAA_m']=
    NOAA_month=df_SIE_NOAA.groupby(pd.PeriodIndex(df_SIE_NOAA.index, freq="M"))['NOAA'].mean()
    NOAA_month = NOAA_month.reset_index()
    NOAA_month['time'] = pd.DatetimeIndex(data=NOAA_month['time'].apply(lambda x: x.strftime('%Y-%m-15')))
    NOAA_month.set_index('time', inplace=True)


    df_SIEUHH['NOAA']=pd.Series(NOAA_month['NOAA'], index=NOAA_month.index)
    #time_osisii_year=np.asarray([(start_time+timedelta(int(time_osisii.data[i]))).year for i in np.nonzero(np.isnan(time_osisii.data)==0)[0]])
    #time_osisii_month=np.asarray([(start_time+timedelta(int(time_osisii.data[i]))).month for i in np.nonzero(np.isnan(time_osisii.data)==0)[0]])




#plotting
if 1:#september SIE

    #df_SIEUHH_sep=0
    if data_SIEUHH:
        df_SIEUHH_sep=df_SIEUHH.loc[df_SIEUHH.index.month==testmonth].copy()
        df_SIEUHH_sep.index=df_SIEUHH_sep.index.year
    df_SIE_sep=df_SIE_month.loc[df_SIE_month.index.month==testmonth]
    df_SIE_sep.index=df_SIE_sep.index.year
    df_SIE_osi_sep=pd.DataFrame({'Ens_m_og':df_SIE_sep.mean(axis=1)}, index=df_SIE_sep.index)
    #df_SIE_osi_sep['time']=pd.DatetimeIndex(df_SIE_osi_sep.index)
    #df_SIE_osi_sep.set_index('time', inplace=True)

    if center_ens: #center ensemble on (self calculated) OSI SAF
        tmp=df_SIE_osi_month.loc[df_SIE_osi_month.index.month==testmonth].copy()
        tmp.index=tmp.index.year
        df_SIE_osi_sep['OSI_SAF']=tmp
        #print(df_SIEUHH_sep['OSI_SAF'])
        for key in df_SIE_sep.keys():
            df_SIE_sep.loc[:,key] = df_SIE_sep.loc[:,key]-(df_SIE_osi_sep['Ens_m_og']-df_SIE_osi_sep['OSI_SAF'])#)

    SIE_probs={'color':{'HadISST':'c', 'HadISST_nsidc':'b', 'OSI_SAF':'r', 'NOAA':'y', 'Ens_m': 'k', 'Ens_m_og': 'gray'}}
    df_SIE_osi_sep['Ens_m']=pd.Series(df_SIE_sep.mean(axis=1), index=df_SIE_sep.index)

    if 1: #remove bias
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        if data_SIEUHH:
            for key in df_SIEUHH_sep.keys():
                if key=='OSI_SAF': continue
                dSIE = df_SIEUHH_sep[key]-df_SIE_osi_sep['OSI_SAF']
                ax1.hist(dSIE, histtype='step', density=True, color=SIE_probs['color'][key], label=key)

                ax2.hist(dSIE-np.nanmean(dSIE), histtype='step', density=True, color=SIE_probs['color'][key], label=key)

        dSIE = df_SIE_sep['batch001i000']-df_SIE_osi_sep['OSI_SAF']
        ax1.hist(dSIE, histtype='step', density=True, color='r', label='batch001i000')
        ax2.hist(dSIE-np.nanmean(dSIE), histtype='step', density=True, color='r', label='batch001i000')


        plt.legend()
        ax1.set_title('September SIE differences to ensemble mean')
        ax2.set_title('September SIE differences, bias corrected')


    fig, ax = plt.subplots()
    if data_SIEUHH:
        for key in df_SIEUHH_sep.keys():
            plt.scatter(df_SIEUHH_sep.index, df_SIEUHH_sep[key], c=SIE_probs['color'][key], label=key)
            #plot = df_SIEUHH_sep.reset_index().plot(ax=ax, kind='scatter',x='index', y=key, c=SIE_probs['color'][key], label=key)
            #plot = df_SIEUHH_sep.plot(ax=ax, kind='scatter',use_index=True, y=key, c=SIE_probs['color'][key], label=key)
    for time in df_SIE_sep.index:
        #df_SIE_sep.loc[df_SIE_sep.index==time]
        SIE_tmp=df_SIE_sep.loc[df_SIE_sep.index==time].to_numpy()[0]
        plt.boxplot(SIE_tmp[np.isnan(SIE_tmp)==0], positions=[time], widths=1)
    lrmodel = LinearRegression()
    for batchi in df_SIE_sep.keys():
        x=df_SIE_sep.index.to_numpy()
        y=df_SIE_sep[batchi].to_numpy()
        x=x[np.isnan(y)==0].reshape(-1,1)
        y=y[np.isnan(y)==0].reshape(-1,1)

        lrmodel.fit(x, y)#
        plt.plot(x, lrmodel.predict(x), color='gray', linewidth=1, alpha=0.4)
    #plot2=(df_SIE_sep.reset_index()).T.plot(kind='box', ax=ax)
    ax.set_xticks(np.arange(1979, 2025, 5))
    ax.set_xticklabels(np.arange(1979, 2025, 5))
    ax.set_xlim([1978, 2025])
    ax.set_xlabel('Time')
    ax.set_ylabel('Sea-Ice Extent [m km²]')
    ax.set_title('{}ern Hemisphere, Month: [{}]'.format({'NH':'North', 'SH':'South'}[XH], testmonth))
    if data_SIEUHH: plt.legend()

    #plt.scatter(2012, 6)

    #plt.legend
    if data_SIAUHH:
        df_SIAUHH_sep=df_SIAUHH.loc[df_SIAUHH.index.month==testmonth].copy()
        df_SIAUHH_sep.index=df_SIAUHH_sep.index.year
    df_SIA_sep=df_SIA_month.loc[df_SIA_month.index.month==testmonth].copy()
    df_SIA_sep.index=df_SIA_sep.index.year

    df_SIA_osi_sep=pd.DataFrame({'Ens_m_og':df_SIA_sep.mean(axis=1)}, index=df_SIA_sep.index)

    if center_ens: #center ensemble on (self calculated) OSI SAF
        tmp=df_SIA_osi_month.loc[df_SIA_osi_month.index.month==testmonth].copy()
        tmp.index=tmp.index.year
        df_SIA_osi_sep['OSI_SAF']=tmp
        for key in df_SIA_sep.keys():
            df_SIA_sep.loc[:,key] = df_SIA_sep[key]-(df_SIA_osi_sep['Ens_m_og']-df_SIA_osi_sep['OSI_SAF'])

    df_SIA_osi_sep['Ens_m']=pd.Series(df_SIA_sep.mean(axis=1), index=df_SIA_sep.index)
    SIA_probs={'color':{'HadISST':'c', 'NSIDC_NT':'b', 'NSIDC_BT':'g', 'HadISST_nsidc': 'm', 'OSI_SAF':'r', 'ESA_CCI':'y', 'Ens_m': 'k', 'Ens_m_og': 'gray'}}

    if 1: #remove bias
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        if data_SIAUHH:
            for key in df_SIAUHH_sep.keys():
                if key=='OSI_SAF': continue
                dSIA = df_SIAUHH_sep[key]-df_SIA_osi_sep['OSI_SAF']
                ax1.hist(dSIA, histtype='step', color=SIA_probs['color'][key], label=key)

                ax2.hist(dSIA-np.nanmean(dSIA), histtype='step', color=SIA_probs['color'][key], label=key)

        dSIA = df_SIA_sep['batch001i000']-df_SIA_osi_sep['OSI_SAF']
        ax1.hist(dSIA, histtype='step',  color='r', label='batch001i000')#density=True,
        ax2.hist(dSIA-np.nanmean(dSIA), histtype='step',  color='r', label='batch001i000')#density=True,


        plt.legend()
        ax1.set_title('September SIA differences to ensemble mean')
        ax1.set_ylabel('Count')
        ax2.set_title('September SIA differences, bias corrected')
        ax2.set_ylabel('Count')

    fig, ax = plt.subplots()
    if data_SIAUHH:
        for key in df_SIAUHH_sep.keys():
            plt.scatter(df_SIAUHH_sep.index, df_SIAUHH_sep[key], c=SIA_probs['color'][key], label=key)
            #plot = df_SIAUHH_sep.reset_index().plot(ax=ax, kind='scatter',x='index', y=key, c=SIA_probs['color'][key], label=key)
            #plot = df_SIAUHH_sep.plot(ax=ax, kind='scatter',use_index=True, y=key, c=SIA_probs['color'][key], label=key)
    for time in df_SIA_sep.index:
        #df_SIA_sep.loc[df_SIA_sep.index==time]
        SIA_tmp=df_SIA_sep.loc[df_SIA_sep.index==time].to_numpy()[0]
        plt.boxplot(SIA_tmp[np.isnan(SIA_tmp)==0], positions=[time], widths=1)
    lrmodel = LinearRegression()
    for batchi in df_SIA_sep.keys():
        x=df_SIA_sep.index.to_numpy()
        y=df_SIA_sep[batchi].to_numpy()
        x=x[np.isnan(y)==0].reshape(-1,1)
        y=y[np.isnan(y)==0].reshape(-1,1)

        lrmodel.fit(x, y)#
        plt.plot(x, lrmodel.predict(x), color='gray', linewidth=1, alpha=0.4)
    #plot2=(df_SIA_sep.reset_index()).T.plot(kind='box', ax=ax)
    ax.set_xticks(np.arange(1979, 2025, 5))
    ax.set_xticklabels(np.arange(1979, 2025, 5))
    ax.set_xlim([1978, 2025])
    ax.set_xlabel('Time')
    ax.set_ylabel('Sea-Ice Area [m km²]')
    ax.set_title('{}ern Hemisphere, Month: [{}]'.format({'NH':'North', 'SH':'South'}[XH], testmonth))
    if data_SIAUHH: plt.legend()

    #plt.scatter(2012, 6)

    #plt.legend

if 0: #dist SIA/SIE all month
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for tmonth in np.arange(1,13):
        #df_SIEUHH_sep=0
        df_SIEUHH_sep=df_SIEUHH.loc[df_SIEUHH.index.month==tmonth].copy()
        df_SIEUHH_sep.index=df_SIEUHH_sep.index.year
        df_SIE_sep=df_SIE_month.loc[df_SIE_month.index.month==tmonth].copy()
        df_SIE_sep.index=df_SIE_sep.index.year
        df_SIEUHH_sep['Ens_m']=pd.Series(df_SIE_sep.mean(axis=1), index=df_SIE_sep.index)

        for i, key in enumerate(df_SIEUHH_sep.keys()):
            if key=='OSI_SAF': continue
            #key='HadISST'
            dSIE = df_SIEUHH_sep[key]-df_SIEUHH_sep['OSI_SAF']
            ax1.boxplot(dSIE[np.isnan(dSIE)==0], positions=[10.*tmonth+i], widths=1, labels=[key])
            ax2.boxplot(dSIE[np.isnan(dSIE)==0]-np.nanmean(dSIE), positions=[10.*tmonth+i], widths=1, labels=[key])

            #ax1.boxplot(df_SIEUHH_sep[key][np.isnan(df_SIEUHH_sep[key])==0], positions=[10.*tmonth+i], widths=1)
        dSIE = df_SIE_sep['batch001i000']-df_SIEUHH_sep['OSI_SAF']
        ax1.boxplot(dSIE[np.isnan(dSIE)==0], positions=[10.*tmonth+i+1], widths=1, labels=['batch001i000'])
        ax2.boxplot(dSIE[np.isnan(dSIE)==0]-np.nanmean(dSIE), positions=[10.*tmonth+i+1], widths=1, labels=['batch001i000'])

        ax1.text(10.*tmonth+1, 1.5, tmonth)
        ax2.text(10.*tmonth+1, 0.673, tmonth)
    ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
    ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)

    ax1.set_title('SIE differences to OSI SAF SII, {}'.format(XH))
    ax1.set_ylabel('[Million km2]')
    ax2.set_title('SIE differences, bias corrected, {}'.format(XH))
    ax1.set_ylabel('[Million km2]')

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for tmonth in np.arange(1,13):
        df_SIAUHH_sep=df_SIAUHH.loc[df_SIAUHH.index.month==tmonth].copy()
        df_SIAUHH_sep.index=df_SIAUHH_sep.index.year
        df_SIA_sep=df_SIA_month.loc[df_SIA_month.index.month==tmonth]
        df_SIA_sep.index=df_SIA_sep.index.year
        df_SIAUHH_sep.loc[:,'Ens_m']=pd.Series(df_SIA_sep.mean(axis=1), index=df_SIA_sep.index)

        for i, key in enumerate(df_SIAUHH_sep.keys()):
            if key=='OSI_SAF': continue
            #key='HadISST'
            dSIA = df_SIAUHH_sep[key]-df_SIAUHH_sep['OSI_SAF']
            ax1.boxplot(dSIA[np.isnan(dSIA)==0], positions=[10.*tmonth+i], widths=1, labels=[key])
            ax2.boxplot(dSIA[np.isnan(dSIA)==0]-np.nanmean(dSIA), positions=[10.*tmonth+i], widths=1, labels=[key])

            #ax1.boxplot(df_SIAUHH_sep[key][np.isnan(df_SIAUHH_sep[key])==0], positions=[10.*tmonth+i], widths=1)
        dSIA = df_SIA_sep['batch001i000']-df_SIAUHH_sep['OSI_SAF']
        ax1.boxplot(dSIA[np.isnan(dSIA)==0], positions=[10.*tmonth+i+1], widths=1, labels=['batch001i000'])
        ax2.boxplot(dSIA[np.isnan(dSIA)==0]-np.nanmean(dSIA), positions=[10.*tmonth+i+1], widths=1, labels=['batch001i000'])

        spread_ens=np.mean(np.std(df_SIA_sep.T, axis=0))
        print('Yearly-mean SIA ensemble spread in [{}]: {:0f}'.format(tmonth, spread_ens))

        ax1.text(10.*tmonth+1, 1.5, tmonth)
        ax2.text(10.*tmonth+1, 0.673, tmonth)
    ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
    ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)

    ax1.set_title('SIA differences to OSI SAF SII, {}'.format(XH))
    ax1.set_ylabel('[Million km2]')
    ax2.set_title('SIA differences, bias corrected, {}'.format(XH))
    ax2.set_ylabel('[Million km2]')


if 1 and not load:#plot individual dates SIC

    #SICs.mask=np.logical_or(SICs.mask, (status[15,:,:] & 4) !=4)
    idate=14
    #for idate in range(30):
    print(SICs_date[idate])
    fig, ax = plt.subplots()
    plo=ax.imshow(SICs[idate,:,:], vmin=100, vmax=150)
    ax.set_title('{} - {} - {}'.format(SICs_date[idate].year, SICs_date[idate].month, SICs_date[idate].day))
    ax.set_xlim([110, 220])
    ax.set_ylim([280, 160])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(plo, ax=ax, label='OSI SAF SIC [%]')

    plt.figure()
    plt.imshow((status[idate,:,:] & 4) ==4)
