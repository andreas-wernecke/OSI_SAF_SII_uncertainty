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
from scipy import optimize


import os, calendar, csv, random#, glob#, datetime
#from os import path

import geopandas, shapely, shapely.vectorized, warnings
import matplotlib.pyplot as plt

from datetime import date, timedelta, datetime
from copy import copy
from sklearn.linear_model import LinearRegression
import seaborn
import cartopy.crs as ccrs
import cartopy.feature as cf

def plotmap(plotfield, ref_crs, Title='', diff=False):

    xyz_cci = ref_crs.transform_points(ccrs.PlateCarree(), lon, lat)
    x=xyz_cci[:,:,0]
    y=xyz_cci[:,:,1]

    #FigN, axN = plt.subplots()
    FigN=plt.figure()
    axN = plt.axes(projection = ref_crs)  # create a set of axes with Mercator projection
    plt.title(Title)
    if diff:
        conf=axN.contourf(x, y, plotfield, \
                          cmap='seismic', levels=[-80, -60, -40, -20, 0, 20, 40, 60, 80], zorder=1)
    else:
        conf=axN.pcolor(x, y, plotfield, zorder=1)
    axN.set_xlabel('X')
    axN.set_ylabel('Y')
    axN.add_feature(cf.LAND, color='gray', zorder=100)
    #ax.add_feature(cf.COASTLINE, color='lightgray')
    axN.gridlines(xlocs=np.linspace(-180, 180, 9), ylocs=np.linspace(-80, 80, 9))
    axN.coastlines(linewidth=2)
    plt.colorbar(conf,ax=axN, label='% SIC')
    axN.set_xlim([-4.5e6, 4.5e6])
    axN.set_ylim([-4.5e6, 4.5e6])
    plt.show()


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
        for cdr in ('cdr_patch','cdr', 'icdr', 'icdrft'):#,#looking for versions of files in this order
            fn = fn_patt.format(a=area, d=d, c=fn_patt_src[cdr])
            fn = os.path.join(sources[cdr].format(y=d.year, m=d.month),fn)
            #print(fn)
            try:
                # this url exists, append it and move to next date
                #ds = xr.open_dataset(fn, decode_times=False)#, decode_times=False
                with nc.Dataset(fn) as ds:
                    found_one_file = True
                    files.append(fn)
                    srcs.append(cdr)
                break#use only one file per date
            except OSError:
                # no valid file at this url, check the next rule
                pass

        # no file found. Add a warning (but we can continue)
        if not found_one_file:
            print("WARNING: could not find OSI SAF SIC v3 file for {} {}; at {}".format(area, d.date(), fn))
        #else:
        #    print(fn)
    return files, srcs

def read_SIC_osi(first_day, last_day, read_dirs, fn_patt, data_version, dx, area):
    #if indirs is provided, read_dir is not used (not added to sources), but still a required variable

    # check area parameter
    if area not in ('nh', 'sh'):
        raise ValueError('Invalid hemisphere (area={})'.format(area))


    fn_patt_src = {'cdr_patch':'cdr-{}-patch'.format(data_version), 'cdr': 'cdr-{}'.format(data_version), 'icdr': 'icdr-{}'.format(data_version), 'icdrft': 'icdrft-{}'.format(data_version)}


    sources = {'cdr_patch' : read_dirs[0],
              'cdr' : read_dirs[1],
              'icdr' : read_dirs[2],
              'icdrft' : read_dirs[3],
              }
    #jsond = json.dumps(sources, sort_keys=True, indent=4)


    files, srcs = find_sic_files(first_day, last_day, area, sources=sources, fn_patt=fn_patt, fn_patt_src=fn_patt_src)
    #ds = xr.open_mfdataset(files,)
    if len(files)==0:
        success=False
        print('File list between beginnig {}-{} to end of {}-{} is empty, proceeding'.format(years[0], months[0], years[-1], months[-1]))
        return [], [], [], [], [], [], [], [], success

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
    #this uses raw values where the OWF is active, i.e. it is not used and allows SIC<0

    SICs_date=np.zeros_like(SICs_date_seconds, dtype=date)
    for i in range(len(SICs_date_seconds)):
        SICs_date[i]=SICs_date_t0+timedelta(seconds=SICs_date_seconds[i])
    success=True

    #bugfix for 12+13March2024 with pixel SIC_raw approx -2e7
    SICs[SICs<-1e3]=SIC_trunc[SICs<-1e3]
    if area=='sh':#removes day with SIC in huge areas at -20% to -40%
        d_out=datetime(2022, 8, 22, 12,0)
        SICs_date, status, SICs, total_err = SICs_date[SICs_date!=d_out], status[SICs_date!=d_out,:,:], SICs[SICs_date!=d_out,:,:], total_err[SICs_date!=d_out,:,:]

    return SICs_date, status, SICs, total_err, lon, lat

def read_SIC_noise(first_day, last_day, read_dirs, fn_patt, data_version, dx, area, includenoise=False):
    #if indirs is provided, read_dir is not used (not added to sources), but still a required variable

    # check area parameter
    if area not in ('nh', 'sh'):
        raise ValueError('Invalid hemisphere (area={})'.format(area))
    fn_patt_src = {'cdr_patch':'cdr-{}-patch'.format(data_version), 'cdr': 'cdr-{}'.format(data_version), 'icdr': 'icdr-{}'.format(data_version), 'icdrft': 'icdrft-{}'.format(data_version)}

    sources = {'cdr_patch' : read_dirs[0],
                  'cdr' : read_dirs[1],
                  'icdr' : read_dirs[2],
                  'icdrft' : read_dirs[3],
                  }
    #print(sources)
    files, srcs = find_sic_files(first_day, last_day, area, sources=sources, fn_patt=fn_patt, fn_patt_src=fn_patt_src)
    #ds = xr.open_mfdataset(files,)

    with nc.MFDataset(files,) as ds:
        SICs=ds.variables['SIC_sample'][:,:]
        status=ds.variables['status_flag'][:,:]
        SICs_date_seconds = ds.variables['time'][:]
        if includenoise:#only to test what the filtered values look like
            noise=ds.variables['noise'][:,:]
            SIC_smooth=SICs-noise
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
        #SICs=copy(SIC_trunc)
    if includenoise:
        return SICs_date, status, SICs, SIC_smooth, noise
    else:
        return SICs_date, status, SICs



def regionalize(lon, lat, XH):
    #NH:
    #            Region  Sea_ID
    #0   Central_Arctic     1.0
    #1         Beaufort     2.0
    #2       Chukchi-NA     3.0
    #3     Chukchi-Asia     3.0
    #4       E_Siberian     4.0
    #5           Laptev     5.0
    #6             Kara     6.0
    #7          Barents     7.0
    #8      E_Greenland     8.0
    #9           Baffin     9.0
    #10         St_Lawr    10.0
    #11          Hudson    11.0
    #12        Can_Arch    12.0
    #13       Bering-NA    13.0
    #14     Bering-Asia    13.0
    #15         Okhotsk    14.0
    #16           Japan    15.0
    #17           Bohai    16.0
    #18          Baltic    17.0
    #19     Gulf_Alaska    18.0
    # SH:
    #         Region  Sea_ID
    #   0    Weddell     1.0
    #   1     Indian     2.0
    #   2    Pacific     3.0
    #   3   RossEast     4.0
    #   4   RossWest     4.0
    #   5  BellAmund     5.0
    if len(lon)!=len(lat):
        raise ValueError('Wrong Lat/Lon shape!')
    if XH=='NH':
        regions_file='/media/dusch/T7 Shield/SI_regions_NSIDC/NSIDC-0780_SeaIceRegions_NH_v1.0.shp'
        #EPSG='3411'
    elif XH=='SH':
        regions_file='/media/dusch/T7 Shield/SI_regions_NSIDC/NSIDC-0780_SeaIceRegions_SH-NASA_v1.0.shp'
        #EPSG='3412'
    else:
        raise ValueError('Which Hemisphere? (NH or SH)')

    reg_ds=geopandas.read_file(regions_file)
    polyid=np.zeros_like(lon,dtype=int).flatten()-1
    #an array holding the index of the ic poligon at each pm location
    for i in reg_ds.index:
        inpolyi=shapely.vectorized.contains(reg_ds.geometry[i], lon.flatten() , lat.flatten())
        if np.sum(polyid[inpolyi]!=-1)>=1:
            warnings.warn(str(np.sum(polyid[inpolyi]!=-1))+' locations in multipe regions', UserWarning)
        polyid[inpolyi]=reg_ds['Sea_ID'][i]
    polyid=polyid.reshape(np.shape(lon))
    polyid.mask=polyid==-1
    return polyid

#---------------------------------------------------------------------------------------------------
###Controls

#Which Hemisphere: (NH/SH)
XH='NH'

#which datasets to read:
data_noise=1
data_osi=1
data_SIAUHH=1#turn off to use the full unc. ts, not cut to the UHH length

#number of noise samples
n_noise=50
#are they processed in different batches? If so name them:
batches=[1]#,2,3,4,5,6,7]
#the samples can be in two different locations:
second_read_dir=1

#Use UHH incl. HadISST (SIA2020=True) or only osi/team/bootstrap? (=False)
SIA2020=0

#The OSI SAF record processed at UHH can be used, or the OSI SIA calculated here (neglecible impact)
useUHHOSI_tocenter=0

#regional SIA?
regional=True #

#include the MIZ calulations?
MIZ=1
#MIZ length equal the 50% contour (MIZ15==False) or 15% contour (==True)?
MIZ15=0

#Start at the SIC fields (load=False) or SIA/MIZ values (load=True)
load = 1
#if load=False, do you want to override/save the SIA/MIZ values (=True) or use only once (=False)?
save = 0

#directory extentions of different studies
save_dir_attach = ''
#save_dir_attach = '_testconverge'
#save_dir_attach = '_test10pc'#how much SIA unc weould we have if we use SIC unc=10% everywhere (where unc>0)

#years of SIC to read in
years=np.arange(1979, 2026)#default
#years=[2020]

#month of SIC to read in
months=np.arange(1,13)
#months=[12]

trends_rel1981_2010=False


###End of controls
#---------------------------------------------------------------------------------------------------

center_ens=True
lcor_temp = 5.
lcor_sp_km =288.
dx=25.#in km, 50km nominal, corrected for typical arctic ocean (instead of NP)


if XH=='NH':
    NorthSouth='north'
    xh='nh'
    ref_crs=ccrs.NorthPolarStereo()
elif XH=='SH':
    NorthSouth='south'
    xh='sh'
    ref_crs=ccrs.SouthPolarStereo()


if data_osi:
    read_dir="/media/dusch/T7 Shield/SIC/SIC/OSI_CDR/SICv30/"

    read_dirs= ["/media/dusch/T7 Shield/SIC/SIC/OSI_CDR/SICv30_patch/v3p0-patch/",
               "/media/dusch/T7 Shield/SIC/SIC/OSI_CDR/SICv30/{y:04d}/{m:02d}/",
               "/media/dusch/T7 Shield/SIC/SIC/OSI_CDR/SICv30/{y:04d}/{m:02d}/",
               "/media/dusch/T7 Shield/SIC/SIC/OSI_CDR/SICv30/{y:04d}/{m:02d}/"]
    data_version="v3p0"
    fn_patt = 'ice_conc_{a:}_ease2-250_{c:}_{d:%Y%m%d}1200.nc'

    if not load:
        first_year=True
        for year in years:
            print('OSI SAF [{}]'.format(year))
            first_day=date(year, months[0], 1)
            last_day=date(year, months[-1], calendar.monthrange(years[-1], months[-1])[1])

            try:
                SICs_date, status, SICs, SIC_uncs, lon, lat = read_SIC_osi(first_day, last_day, read_dirs, fn_patt, data_version, dx, xh)
                #SICs_date, status, SICs = read_SIC_noise(first_day, last_day, sources, fn_patt, data_version, dx, xh)
            except:
                continue

            SIA_tmp=np.sum(np.sum(SICs, axis=1), axis=1)*dx*dx/100.


            if 0:#plot specific date
                date_plot=datetime(2022, 8, 21, 12,0)
                if year==date_plot.year:
                    datemask=SICs_date==date_plot
                    plotmap(SICs[datemask,:,:][0,:,:], ref_crs, Title='OSI 25km {}-{}-{}'.format(date_plot.year, date_plot.month, date_plot.day), diff=False)
                    plotmap(SIC_uncs[datemask,:,:][0,:,:], ref_crs, Title='OSI 25km Total Uncertainties {}-{}-{}'.format(date_plot.year, date_plot.month, date_plot.day), diff=False)


            if first_year:
                d={'OSI_SAF':pd.Series(SIA_tmp, index=SICs_date)}
                df_SIA_osi=pd.DataFrame(data=d, index=SICs_date)

            else:
                df_SIA_osi=df_SIA_osi.combine_first(pd.DataFrame({'OSI_SAF':SIA_tmp.data}, index=SICs_date))

            if regional:
                ids = regionalize(lon, lat, XH)
                uniqie_ids=np.unique(ids)
                for region in uniqie_ids[uniqie_ids.mask==0]:
                    SIA_tmp=np.sum(SICs[:,ids==region], axis=1)*dx*dx/100.

                    if first_year:

                        if 'df_SIA_osi_reg' not in globals():
                            d={str(region):pd.Series(SIA_tmp, index=SICs_date)}
                            df_SIA_osi_reg=pd.DataFrame(data=d, index=SICs_date)
                        else:
                            df_SIA_osi_reg[str(region)]=pd.Series(SIA_tmp, index=SICs_date)
                    else:
                        df_SIA_osi_reg=df_SIA_osi_reg.combine_first(pd.DataFrame({str(region):SIA_tmp.data}, index=SICs_date))

            if MIZ:
                if 'y_forMIZ' not in globals():
                    xyz_cci = ref_crs.transform_points(ccrs.PlateCarree(), lon, lat)
                    x_forMIZ=xyz_cci[:,:,0]
                    y_forMIZ=xyz_cci[:,:,1]
                l_MIZs=np.zeros(np.shape(SICs)[0])
                for i_SICs in range(np.shape(SICs)[0]):
                    FigN, axN =plt.subplots()
                    axN = plt.axes(projection = ref_crs)  # create a set of axes with Mercator projection

                    #conf=axN.pcolor(x, y, SICs[10,:,:], zorder=1)
                    if MIZ15:#to understand SIE better
                        cont=plt.contour(x_forMIZ, y_forMIZ, SICs[i_SICs,:,:], [15])
                    else:#best estimate of MIZ
                        cont=plt.contour(x_forMIZ, y_forMIZ, SICs[i_SICs,:,:], [50])
                    polys=cont.allsegs[0]
                    n_polies=len(polys)
                    for i_poly in range(n_polies):
                        x_steps, y_steps = np.diff(polys[i_poly], axis=0)[:,0], np.diff(polys[i_poly], axis=0)[:,1]
                        l_MIZs[i_SICs]+=np.sqrt(x_steps**2+y_steps**2).sum()/1e3#add length of polygon to l_miz in km

                    #if SICs_date[i_SICs]==datetime(2006, 7, 18, 12,0):
                    #    axN.set_xlabel('X')
                    #    axN.set_ylabel('Y')
                    #    axN.add_feature(cf.LAND, color='gray', zorder=100)
                    #    #ax.add_feature(cf.COASTLINE, color='lightgray')
                    #    axN.gridlines(xlocs=np.linspace(-180, 180, 9), ylocs=np.linspace(-80, 80, 9))
                    #    axN.coastlines(linewidth=2)
                    #    plt.colorbar(cont,ax=axN, label='% SIC')
                    #    axN.set_xlim([-4.5e6, 4.5e6])
                    #    axN.set_ylim([-4.5e6, 4.5e6])
                    #    plt.show()
                    #    plt.savefig('/home/dusch/Downloads/map_contour.png', dpi=200)
                    plt.close(FigN)

                    if 1:
                        #if l_MIZs[i_SICs]>45000 and SICs_date[i_SICs].month==12:
                        #if SICs_date[i_SICs].day==15:
                        if SICs_date[i_SICs]==datetime(2020, 5, 29, 12,0):
                            Fig =plt.figure()
                            ax = plt.axes(projection = ref_crs)
                            conf=ax.pcolor(x_forMIZ, y_forMIZ, SICs[i_SICs,:,:], vmax=120)
                            cont=plt.contour(x_forMIZ, y_forMIZ, SICs[i_SICs,:,:], [15], colors='r')
                            #polys=cont.allsegs[0]
                            #n_polies=len(polys)
                            #for i_poly in range(n_polies):
                            #    x_steps, y_steps = np.diff(polys[i_poly], axis=0)[:,0], np.diff(polys[i_poly], axis=0)[:,1]
                            #    l_MIZs[i_SICs]+=np.sqrt(x_steps**2+y_steps**2).sum()/1e3#add length of polygon to l_miz in km
                            ax.set_title('Mon: {}; MIZ L [km]: {}'.format(SICs_date[i_SICs].month, int(l_MIZs[i_SICs])))
                            ax.set_title('Date: {}; MIZ L [km]: {}'.format(SICs_date[i_SICs], int(l_MIZs[i_SICs])))
                            #ax.set_xlabel('X')
                            #ax.set_ylabel('Y')
                            #ax.add_feature(cf.LAND, color='gray', zorder=100)
                            ax.add_feature(cf.COASTLINE, color='lightgray')
                            ax.gridlines(xlocs=np.linspace(-180, 180, 9), ylocs=np.linspace(-80, 80, 9))
                            ax.coastlines(linewidth=2)
                            #plt.colorbar(conf,ax=ax, label='% SIC')
                            ax.set_xlim([-4.5e6, 4.5e6])
                            ax.set_ylim([-4.5e6, 4.5e6])

                #plt.close('all')

                if first_year:
                    df_MIZ=pd.DataFrame({'l_MIZ':l_MIZs}, index=SICs_date)

                else:
                    df_MIZ=df_MIZ.combine_first(pd.DataFrame({'l_MIZ':l_MIZs}, index=SICs_date))



            first_year=False

        if save:
            df_SIA_osi.to_csv(read_dir+'SIA_{}.csv'.format(XH))

            if regional:
                df_SIA_osi_reg.to_csv(read_dir+'SIA_{}_reg.csv'.format(XH))

            if MIZ:
                if MIZ15:
                    df_MIZ.to_csv(read_dir+'l_MIZ15_{}.csv'.format(XH))
                else:
                    df_MIZ.to_csv(read_dir+'l_MIZ_{}.csv'.format(XH))

    if load:
        df_SIA_osi=pd.read_csv(read_dir+'SIA_{}.csv'.format(XH), index_col=0)
        if regional:
            df_SIA_osi_reg=pd.read_csv(read_dir+'SIA_{}_reg.csv'.format(XH), index_col=0)
        if MIZ:
            if MIZ15:
                df_MIZ=pd.read_csv(read_dir+'l_MIZ15_{}.csv'.format(XH), index_col=0)
            else:
                df_MIZ=pd.read_csv(read_dir+'l_MIZ_{}.csv'.format(XH), index_col=0)
            df_MIZ['time']=pd.DatetimeIndex(df_MIZ.index)
            df_MIZ.set_index('time', inplace=True)

    df_SIA_osi=df_SIA_osi/1e6

    df_SIA_osi['time']=pd.DatetimeIndex(df_SIA_osi.index)
    df_SIA_osi.set_index('time', inplace=True)

    df_SIA_osi_month=df_SIA_osi.groupby(pd.PeriodIndex(df_SIA_osi.index, freq="M"))['OSI_SAF'].mean()
    df_SIA_osi_month = df_SIA_osi_month.reset_index()
    df_SIA_osi_month['time'] = pd.DatetimeIndex(data=df_SIA_osi_month['time'].apply(lambda x: x.strftime('%Y-%m-15')))
    df_SIA_osi_month.set_index('time', inplace=True)

    df_SIA_osi_month_count=df_SIA_osi.groupby(pd.PeriodIndex(df_SIA_osi.index, freq="M")).count()
    df_SIA_osi_month_count = df_SIA_osi_month_count.reset_index()
    df_SIA_osi_month_count['time'] = pd.DatetimeIndex(data=df_SIA_osi_month_count['time'].apply(lambda x: x.strftime('%Y-%m-15')))
    df_SIA_osi_month_count.set_index('time', inplace=True)

    #paths=['/media/dusch/T7 Shield/SIC/noise/daily/{}/osi_sep_full/'.format(XH)]
    if regional:
        regions=df_SIA_osi_reg.keys()
        df_SIA_osi_reg_m={}
        for region in regions:
            df_SIA_tmp=pd.DataFrame(df_SIA_osi_reg[region])
            #make sure index is of type datetimeindex
            df_SIA_tmp['time']=pd.DatetimeIndex(df_SIA_tmp.index)
            df_SIA_tmp.set_index('time', inplace=True)

            df_SIA_tmp=df_SIA_tmp/1e6

            #print(df_SIA_tmp.loc[df_SIA_tmp.index==datetime(2004, 10, 13)])

            df_SIA_month_tmp=df_SIA_tmp.groupby(pd.PeriodIndex(df_SIA_tmp.index, freq="M")).mean()
            df_SIA_month_tmp = df_SIA_month_tmp.reset_index()
            df_SIA_month_tmp['time'] = pd.DatetimeIndex(data=df_SIA_month_tmp['time'].apply(lambda x: x.strftime('%Y-%m-15')))
            df_SIA_month_tmp.set_index('time', inplace=True)
            df_SIA_osi_reg_m[region]=df_SIA_month_tmp


if data_noise:
    save_dir='/media/dusch/T7 Shield/SIC/noise/daily/osi_v3{}/'.format(save_dir_attach)
    #save_dir='/media/dusch/AndreasExt/SIC/noise/daily/osi_v3{}/'.format(save_dir_attach)
    read_dir_og=save_dir+'batch{:03d}i{:03d}/'
    fn_patt_og='NOISE_batch{bb:03d}i{ii:03d}_lx{lx:.0f}km_lt{lt:.0f}d_ice_conc_{a:}_ease2-250_{c:}_{d:}1200.nc'
    data_version="v3p0"
    if not load:
        for i_batch in batches:
            for i_noise in np.arange(n_noise):
                batch_sample='batch{:03d}i{:03d}'.format(i_batch, i_noise)
                print(batch_sample)

                if i_noise>29 and second_read_dir:
                    read_dir_og= '/media/dusch/AndreasExt/SIC/noise/daily/osi_v3/batch{:03d}i{:03d}/'
                read_dir=read_dir_og.format(i_batch,i_noise)
                read_dirs= [read_dir+'/{y:04d}/{m:02d}/',read_dir+'/{y:04d}/{m:02d}/',read_dir+'/{y:04d}/{m:02d}/', read_dir+'/{y:04d}/{m:02d}/']
                fn_patt=fn_patt_og.format(bb=i_batch, ii=i_noise, lx=lcor_sp_km, lt=lcor_temp, a='{a:}', c='{c:}', d='{d:%Y%m%d}')

                for year in years:
                    first_day=date(year, months[0], 1)
                    last_day=date(year, months[-1], calendar.monthrange(years[-1], months[-1])[1])

                    try:
                        #SICs_date, status, SICs = read_SIC_noise(first_day, last_day, sources, fn_patt, data_version, dx, xh)
                        SICs_date, status, SICs, SICs_smooth, SICs_noise = read_SIC_noise(first_day, last_day, read_dirs, fn_patt, data_version, dx, xh, includenoise=True)
                    except:
                        continue

                    if 0:#plot specific date
                        if year==2016:
                            plotmap(SICs[257,:,:], ref_crs, Title='Member-1 2016-09-15', diff=False)
                            plotmap(SICs_smooth[257,:,:], ref_crs, Title='Background 2016-09-15', diff=False)
                            plotmap(SICs_noise[257,:,:], ref_crs, Title='Noise 2016-09-15', diff=True)

                            # #SICs.mask=np.logical_or(SICs.mask, (status[257,:,:] & 4) !=4)
                            # plt.figure()
                            # plt.imshow(SICs[257,:,:])
                            # plt.colorbar()

                            # plt.figure()
                            # plt.imshow((status[257,:,:] & 4) ==4)

                    SIA_tmp=np.sum(np.sum(SICs, axis=1), axis=1)*dx*dx/100.

                    if 'df_SIA' in globals():
                        if (batch_sample in df_SIA.keys()):
                            df_SIA=df_SIA.combine_first(pd.DataFrame({batch_sample:SIA_tmp.data}, index=SICs_date))

                        else:
                            df_SIA[batch_sample]=pd.Series(SIA_tmp, index=SICs_date)

                    else:
                        d={batch_sample:pd.Series(SIA_tmp, index=SICs_date)}
                        df_SIA=pd.DataFrame(data=d, index=SICs_date)



                    if regional:
                        if 'SIA_noise_reg' not in globals(): SIA_noise_reg={}

                        ids = regionalize(lon, lat, XH)
                        uniqie_ids=np.unique(ids)
                        for region in uniqie_ids[uniqie_ids.mask==0]:
                            str_reg=str(region)
                            SIA_tmp=np.sum(SICs[:,ids==region], axis=1)*dx*dx/100.

                            if str_reg not in SIA_noise_reg.keys():#first time this region
                                d={batch_sample:pd.Series(SIA_tmp, index=SICs_date)}
                                SIA_noise_reg[str_reg]=pd.DataFrame(data=d, index=SICs_date)
                            else:
                                if (batch_sample in SIA_noise_reg[str_reg].keys()):#region and sample exists, new data
                                    SIA_noise_reg[str_reg]=SIA_noise_reg[str_reg].join(pd.concat([SIA_noise_reg[str_reg][batch_sample].dropna(),pd.Series(SIA_tmp.data, index=SICs_date, name=batch_sample)], axis=0), how='outer', lsuffix='_old')
                                    SIA_noise_reg[str_reg].pop(batch_sample+'_old')

                                else:#region exists, new sample
                                    SIA_noise_reg[str_reg][batch_sample]=pd.Series(SIA_tmp, index=SICs_date)



        if save:
            df_SIA.to_csv(save_dir+'SIA_{}.csv'.format(XH))

            if regional:
                for key in SIA_noise_reg.keys():
                    SIA_noise_reg[key].to_csv(save_dir+'SIA_{}_reg_{}.csv'.format(XH, key))

    if load:
        df_SIA=pd.read_csv(save_dir+'SIA_{}.csv'.format(XH), index_col=0)
        if np.shape(df_SIA)[1]!=n_noise:
            n_noise=np.shape(df_SIA)[1]
            warnings.warn('Data is loaded and has ensemble members not equal n_noise. n_noise set to {}'.format(n_noise), UserWarning)
        if regional:
            SIA_noise_reg={}
            dirlist=os.listdir(save_dir)
            for fn in dirlist:
                if ('SIA_{}_reg_'.format(XH) in fn) and ('.csv' in fn):
                    str_reg=fn.rsplit('_')[-1].rsplit('.')[0]
                    SIA_noise_reg[str_reg]=pd.read_csv(save_dir+fn, index_col=0)



    #make sure index is of type datetimeindex
    df_SIA['time']=pd.DatetimeIndex(df_SIA.index)
    df_SIA.set_index('time', inplace=True)

    #reset the interval, if not load only the relevant data will be read in,
    #however, if load it reads in all the TS, which is why it needs to be cut
    inerval_mask=np.array([x in years for x in df_SIA.index.year])
    df_SIA=df_SIA.loc[inerval_mask]


    df_SIA=df_SIA/1e6
    #print(df_SIA.loc[df_SIA.index==datetime(2004, 10, 13)])

    df_SIA_month=df_SIA.groupby(pd.PeriodIndex(df_SIA.index, freq="M")).mean()
    df_SIA_month = df_SIA_month.reset_index()
    df_SIA_month['time'] = pd.DatetimeIndex(data=df_SIA_month['time'].apply(lambda x: x.strftime('%Y-%m-15')))
    df_SIA_month.set_index('time', inplace=True)


    #deriving monthly averages of regions
    if regional:
        regions=SIA_noise_reg.keys()
        SIA_noise_reg_m={}
        for region in regions:
            df_SIA_tmp=SIA_noise_reg[region]
            #make sure index is of type datetimeindex
            df_SIA_tmp['time']=pd.DatetimeIndex(df_SIA_tmp.index)
            df_SIA_tmp.set_index('time', inplace=True)

            df_SIA_tmp=df_SIA_tmp/1e6

            #print(df_SIA_tmp.loc[df_SIA_tmp.index==datetime(2004, 10, 13)])

            df_SIA_month_tmp=df_SIA_tmp.groupby(pd.PeriodIndex(df_SIA_tmp.index, freq="M")).mean()
            df_SIA_month_tmp = df_SIA_month_tmp.reset_index()
            df_SIA_month_tmp['time'] = pd.DatetimeIndex(data=df_SIA_month_tmp['time'].apply(lambda x: x.strftime('%Y-%m-15')))
            df_SIA_month_tmp.set_index('time', inplace=True)
            SIA_noise_reg_m[region]=df_SIA_month_tmp




if data_SIAUHH:
    #areadir='/mnt/icdc/ice_and_snow/uhh_seaiceareatimeseries/DATA/'
    areadir='/media/dusch/T7 Shield/CCI_data/SIA_obs/'
    if SIA2020:
        if XH=='NH': areafn='SeaIceArea__NorthernHemisphere__monthly__UHH__v2024_fv0.01.nc'
        else:        areafn='SeaIceArea__SouthernHemisphere__monthly__UHH__v2024_fv0.01.nc'
    else:
        if XH=='NH': areafn='SIA_observations_nh_v2025_fv0.01_nsidc_osisaf.nc'
        else:        areafn='SIA_observations_sh_v2025_fv0.01_nsidc_osisaf.nc'
    ds=nc.Dataset(areadir+areafn, "r")
    SIAUHH_time=ds.variables['time'][:]
    SIA_osi=ds.variables['osisaf'][:]
    SIA_nt=ds.variables['nsidc_nt'][:]
    SIA_bt=ds.variables['nsidc_bt'][:]
    if SIA2020:
        SIA_esa=ds.variables['esa'][:]
        SIA_hado=ds.variables['HadISST_orig'][:]
        SIA_had_nsidc=ds.variables['HadISST_nsidc'][:]
    ds.close()

    start_time=date(1849,12,31)#-1 day to hi the 15th of the month
    time_tmp=np.asarray([start_time+timedelta(SIAUHH_time.data[i]) for i in np.nonzero(np.isnan(SIAUHH_time.data)==0)[0]])
    SIAUHH_time=pd.DatetimeIndex(data=time_tmp)

    df_SIAUHH=pd.DataFrame(data={'OSI_SAF':pd.Series(SIA_osi, index=SIAUHH_time)},  index=SIAUHH_time)
    #df_SIAUHH['ESA_CCI']=pd.Series(SIA_esa, index=SIAUHH_time)
    df_SIAUHH['NSIDC_NT']=pd.Series(SIA_nt, index=SIAUHH_time)
    df_SIAUHH['NSIDC_BT']=pd.Series(SIA_bt, index=SIAUHH_time)
    if SIA2020:
        df_SIAUHH['HadISST_nsidc']=pd.Series(SIA_had_nsidc, index=SIAUHH_time)
        df_SIAUHH['HadISST']=pd.Series(SIA_hado, index=SIAUHH_time)

    #df_SIAUHH['WALSH']=pd.Series(SIA_walsh, index=SIAUHH_time)
    df_SIAUHH.dropna(inplace=True)

    if 0:
        plt.figure()
        for key in df_SIAUHH.keys():
            df_SIAUHH[key].plot()

    #rint(df_SIA_month)
    inerval_mask=np.array([x in df_SIA_month.index for x in df_SIAUHH.index])
    df_SIAUHH=df_SIAUHH.loc[inerval_mask]

    inerval_mask=np.array([x in df_SIAUHH.index for x in df_SIA_month.index])
    df_SIA_month=df_SIA_month.loc[inerval_mask]

    inerval_mask=np.array([x in df_SIAUHH.index for x in df_SIA_osi_month_count.index])
    df_SIA_osi_month_count=df_SIA_osi_month_count.loc[inerval_mask]


#the centering. Only monthly valuies so far!
if center_ens:
    if not data_osi: raise Warning('Cant center the ensemble without osi saf as center')

    ens_m_og=pd.DataFrame({'Ens_SIA_m_og':df_SIA_month.mean(axis=1)}, index=df_SIA_month.index)

    if useUHHOSI_tocenter and data_SIAUHH:#use UHH values for centering, if avail for SIA. Offfside: shorter ts
        ens_m_og['OSI_SAF_SIA']=df_SIAUHH['OSI_SAF']
    else:
        ens_m_og['OSI_SAF_SIA']=df_SIA_osi_month

    ens_m_og.dropna(inplace=True)
    for key in df_SIA_month.keys():
        df_SIA_month.loc[:,key] = df_SIA_month[key]-(ens_m_og['Ens_SIA_m_og']-ens_m_og['OSI_SAF_SIA'])
    df_SIA_month.dropna(inplace=True)
    if regional:
        for region in regions:
            df_SIA_osi_reg_m['Ens_m_og-'+region]=SIA_noise_reg_m[region].mean(axis=1)

            for key in SIA_noise_reg_m[region].keys():
                SIA_noise_reg_m[region].loc[:,key] = SIA_noise_reg_m[region][key]-(df_SIA_osi_reg_m['Ens_m_og-'+region]-df_SIA_osi_reg_m[region][region])
    if 0:
        plt.figure()
        for key in df_SIA_month.keys():
            df_SIA_month[key].plot()

#plotting

SIA_probs={'color':{'HadISST':'c', 'NSIDC_NT':'b', 'NSIDC_BT':'g', 'HadISST_nsidc': 'm', 'OSI_SAF':'r', 'ESA_CCI':'y', 'Ens_m': 'k', 'Ens_m_og': 'gray', 'batch001i000':'gray'}}
monthnames=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Y-Mean"]


if 0:#SIA unc over time
    #derive seasonal cycle
    ens_vars=np.zeros(12)
    ens_var_stds=np.zeros(12)
    ens_vars2=np.zeros(12)
    ens_var_quan5=np.zeros(12)
    ens_var_quan95=np.zeros(12)
    OSI_SC = np.zeros(12)
    OSI_SC_quan5 = np.zeros(12)
    OSI_SC_quan95 = np.zeros(12)
    for i in range(1,13):
        ens_vars[i-1]=df_SIA_month.loc[df_SIA_month.index.month==i].T.std().mean()
        ens_var_stds[i-1]=df_SIA_month.loc[df_SIA_month.index.month==i].T.std().std()
        ens_vars2[i-1]=np.mean(np.var(df_SIA_month.loc[df_SIA_month.index.month==i].T, axis=0))

        ens_var_quan5[i-1]=df_SIA_month.loc[df_SIA_month.index.month==i].T.std().quantile(0.16)
        if ens_var_quan5[i-1]<=0: ens_var_quan5[i-1]=0.
        ens_var_quan95[i-1]=df_SIA_month.loc[df_SIA_month.index.month==i].T.std().quantile(0.84)

        OSI_SC[i-1] = df_SIA_osi_month.loc[df_SIA_osi_month.index.month==i].mean()

        OSI_SC_quan5[i-1]=df_SIA_osi_month.loc[df_SIA_osi_month.index.month==i].quantile(0.16)
        if OSI_SC_quan5[i-1]<=0: OSI_SC_quan5[i-1]=0.
        OSI_SC_quan95[i-1]=df_SIA_osi_month.loc[df_SIA_osi_month.index.month==i].quantile(0.84)

    ens_vars_long=np.concatenate(([ens_vars[-1]], ens_vars, [ens_vars[0]]))#double Dec. and Jan for plotting
    ens_var_stds_long=np.concatenate(([ens_var_stds[-1]], ens_var_stds, [ens_var_stds[0]]))
    ens_var_quan5_long=np.concatenate(([ens_var_quan5[-1]], ens_var_quan5, [ens_var_quan5[0]]))
    ens_var_quan95_long=np.concatenate(([ens_var_quan95[-1]], ens_var_quan95, [ens_var_quan95[0]]))
    OSI_SC_long=np.concatenate(([OSI_SC[-1]], OSI_SC, [OSI_SC[0]]))
    OSI_SC_quan5_long=np.concatenate(([OSI_SC_quan5[-1]], OSI_SC_quan5, [OSI_SC_quan5[0]]))
    OSI_SC_quan95_long=np.concatenate(([OSI_SC_quan95[-1]], OSI_SC_quan95, [OSI_SC_quan95[0]]))

    dt_12month=timedelta(days=365)
    fig, ax=plt.subplots()
    ax_twin=ax.twinx()
    SIA_unc_ts_full = df_SIA_month.std(axis=1).loc[df_SIA_osi_month_count['OSI_SAF']>10]

    SIA_unc_anomaly=1
    if SIA_unc_anomaly: #remove season cacle from TS
        df_SIA_osi_anomaly = df_SIA_osi_month.copy()
        for i in range(1,13):
            SIA_unc_ts_full.loc[SIA_unc_ts_full.index.month==i] = SIA_unc_ts_full.loc[SIA_unc_ts_full.index.month==i] - ens_vars[i-1]
            df_SIA_osi_anomaly.loc[df_SIA_osi_anomaly.index.month==i] = df_SIA_osi_anomaly.loc[df_SIA_osi_anomaly.index.month==i] - OSI_SC[i-1]

    df_SIA_osi_anomaly['OSI_SAF'].rolling(dt_12month, center=True).mean().plot(color='#743a34', label='OSI SAF', ax=ax_twin, zorder=0, ls='--')#
    if XH == 'NH':#'#8EB69B'
        SIA_unc_ts_full.plot(color='#CCAA61', label='Monthly Unc.', ax=ax, zorder=9)#CCAA61##cc8b61
    else:
        SIA_unc_ts_full.plot(color='#8EB69B', label='Monthly Unc.', ax=ax, zorder=9)

    SIA_unc_ts_full.rolling(dt_12month, center=True).mean().plot(color='k', label='Running Mean Unc.', ax=ax, zorder=10)

    ax_twin.spines['right'].set_color('#743a34')
    ax_twin.yaxis.label.set_color('#743a34')
    ax_twin.tick_params(axis='y', colors='#743a34', labelsize=12)
    ax_twin.set_ylim([-2, 2])
    ax_twin.set_ylabel(r'SIA Anomaly [10$^6$ km$^2$]', fontsize=13)

    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim([-0.04, 0.04])
    #ax.set_ylabel(r'SIA Uncertainty Anomaly [10$^6$ km$^2$]', fontsize=13)

    #df_SIA_month.std(axis=1).loc[df_SIA_osi_month_count['OSI_SAF']>10].plot(color='#8EB69B', label='Monthly')
    #df_SIA_month.std(axis=1).loc[df_SIA_osi_month_count['OSI_SAF']>10].rolling(dt_12month, center=True).mean().plot(color='k', label='12 Month running')
    ax.legend(loc='lower left')
    ax_twin.legend(loc='lower right')
    ax.set_xlabel('Time', fontsize=13)
    ax.set_xlim([df_SIA_month.index[0], df_SIA_month.index[-1]])

    print('mean monthly SIA unc: {}'.format(df_SIA_month.std(axis=1).loc[df_SIA_osi_month_count['OSI_SAF']>5].mean()))
    if SIA_unc_anomaly:
        if XH=='NH':
            ax.set_ylabel(r'Arctic SIA Uncertainty Anomaly [10$^6$ km$^2$]', fontsize=13)
        if XH=='SH':
            ax.set_ylabel(r'Antarctic SIA Uncertainty Anomaly [10$^6$ km$^2$]', fontsize=13)
        ax.set_ylim([-0.05, 0.05])
    else:
        if XH=='NH':
            ax.set_ylabel(r'Arctic SIA Uncertainty [10$^6$ km$^2$]', fontsize=13)
        if XH=='SH':
            ax.set_ylabel(r'Antarctic SIA Uncertainty [10$^6$ km$^2$]', fontsize=13)
        ax.set_ylim([0.055, 0.19])
    ax.set_ylabel(r'SIA Uncertainty Anomaly [10$^6$ km$^2$]', fontsize=13)
    fig.tight_layout()
    if 0:
        if XH=='NH':
            fig.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/unc_ts_A.png', dpi=300)
        if XH=='SH':
            fig.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/unc_ts_AA.png', dpi=300)

    #daily
    if 0:
        fig, ax = plt.subplots()
        #df_SIA.plot()
        df_SIA.std(axis=1).plot(label='Daily')
        df_SIA.std(axis=1).rolling(365).mean().shift(-182).plot(label='12 Month running')
        plt.legend()
        ax.set_ylabel('Ensemble STD [m km2]')
        print('mean daily SIA unc: {}'.format(df_SIA.std(axis=1).mean()))

        fig, ax = plt.subplots()
        #df_SIA.plot()
        SIAstd=df_SIA_month.std(axis=1).to_numpy()
        SIAmean=df_SIA_month.mean(axis=1).to_numpy()
        plt.scatter(SIAstd, SIAmean)
        ax.set_ylabel('Ensemble mean SIA [m km2]')
        ax.set_xlabel('Ensemble std [m km2]')


    if MIZ:
        lrmodel = LinearRegression()
        fig, ax = plt.subplots()
        for tmonth in np.arange(1,13):
            SIAstd=df_SIA.loc[df_SIA.index.month==tmonth].std(axis=1).to_numpy()
            MIZmean=df_MIZ.loc[df_MIZ.index.month==tmonth].mean(axis=1).to_numpy()
            #valid=(df_SIA_osi_month_count['OSI_SAF'].loc[df_SIA_month.index.month==tmonth].to_numpy()>5)
            #SIAstd=SIAstd[valid]
            #MIZmean=MIZmean[valid]
            fit=lrmodel.fit(MIZmean.reshape(-1, 1), SIAstd)

            plt.scatter(MIZmean, SIAstd, label='{}'.format(monthnames[tmonth-1]))
            plt.plot(MIZmean, lrmodel.predict(MIZmean.reshape(-1, 1)), color='gray', linewidth=1)
        ax.set_xlabel('MIZ length [km]')
        ax.set_ylabel('SIA Uncertainty [m km2]')
        plt.legend()


    #plot seasonal cycle
    # run this first with the following if statement =True and then a second time with this turned off and for the other hemisphere.
    #Might be necessary to copy this block into an ipython console
    if 1:
        fig_clim, ax_clims = plt.subplots(nrows=2, height_ratios=[2, 1], sharex=True)
        #ax_clim_twin = ax_clim.twinx()
        ax_clim=ax_clims[0]
        ax_clim_twin=ax_clims[1]

    if XH=='NH':
        ax_clim_twin.plot(np.arange(0,14), OSI_SC_long, label='Arctic SIA', ls='--', c='k')
        #ax_clim_twin.fill_between(np.arange(0,14), OSI_SC_quan5_long, OSI_SC_quan95_long, alpha=0.2, color='k')
        ax_clim.plot(np.arange(0,14), ens_vars_long, linewidth=2, label='Arctic Uncertainty', c='#cc8b61')
        ax_clim.fill_between(np.arange(0,14), ens_var_quan5_long, ens_var_quan95_long, alpha=0.4, color='#cc8b61')
        print('Arctic SIA uncertainties (per month): {}'.format(ens_vars))

    if XH=='SH':
        ax_clim_twin.plot(np.arange(0,14), OSI_SC_long, label='Antarctic SIA', ls=':', c='k',zorder=0)
        #ax_clim_twin.fill_between(np.arange(0,14), OSI_SC_quan5_long, OSI_SC_quan95_long, alpha=0.2, color='k', zorder=0,)
        ax_clim.plot(np.arange(0,14), ens_vars_long, linewidth=2, label='Antarctic Uncertainty', c='#4f6e5c',zorder=1)
        ax_clim.fill_between(np.arange(0,14), ens_var_quan5_long, ens_var_quan95_long, alpha=0.4, color='#4f6e5c', zorder=1)
        print('Antarctic SIA uncertainties (per month): {}'.format(ens_vars))
    #ax_clim.fill_between(np.arange(0,14), ens_vars_long-ens_var_stds_long, ens_vars_long+ens_var_stds_long, alpha=0.4)

    ax_clim_twin.tick_params(axis='y', labelsize=12)
    ax_clim_twin.set_ylabel('SIA OSI SAF\n'r'[10$^6$ km$^2$]', fontsize=13)
    ax_clim.tick_params(axis='both', labelsize=12)
    ax_clim_twin.tick_params(axis='both', labelsize=12)

    #ax.set_ylabel(r'SIA Uncertainty Anomaly [10$^6$ km$^2$]', fontsize=13)

    #df_SIA_month.std(axis=1).loc[df_SIA_osi_month_count['OSI_SAF']>10].plot(color='#8EB69B', label='Monthly')
    #df_SIA_month.std(axis=1).loc[df_SIA_osi_month_count['OSI_SAF']>10].rolling(dt_12month, center=True).mean().plot(color='k', label='12 Month running')


    ax_clim.set_xlim([0.5, 12.5])
    #ax_clim.set_ylim([0., 0.18])
    ax_clim_twin.set_ylim([0., 18])
    ax_clim_twin.set_yticks([0,9,18])
    ax_clim.set_xticks(np.arange(1,13))
    ax_clim.set_xticklabels(monthnames[:12], fontsize=12)
    ax_clim.set_ylabel('SIA Uncertainty [10$^6$ km$^2$]', fontsize=13)
    if 0:
        ax_clim.legend(loc='upper left')
        ax_clim_twin.legend(loc='upper left')


    #plt.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/unc_clim.png', dpi=300)



    if 0: #save std as ncindex_col

        df_SIA_month.std(axis=1).to_csv('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/monthly_SIA_std_SH.csv', index=True, header=True)
        #a=pd.read_csv('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/monthly_SIA_std.csv', index_col='time')

        df_SIA_osi_month_count['OSI_SAF'].to_csv('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/monthly_SIA_DiM_SH.csv', index=True, header=True)
        #a=pd.read_csv('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/monthly_SIA_DiM.csv', index_col='time')
        df_SIA.std(axis=1).to_csv('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/daily_SIA_std_SH.csv', index=True, header=True)
        #a=pd.read_csv('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/daily_SIA_std.csv', index_col='time')
        df_MIZ.to_csv('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/MIZ_length_SH.csv', index=True, header=True)
        #a=pd.read_csv('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/MIZ_length_NH.csv', index_col='time')


if 0 and MIZ:
    lrmodel = LinearRegression()
    fig, ax = plt.subplots()
    scat=ax.scatter(df_MIZ, df_SIA_osi, c=df_MIZ.index.month, cmap='twilight', vmin=1, vmax=13)
    cbar=plt.colorbar(scat, ax=ax, drawedges=True, boundaries=np.arange(0,13)+0.5, ticks=np.arange(1,13))
    cbar.ax.set_yticklabels(monthnames[:12])  # vertically oriented colorbar
    ax.set_xlabel(r'Marginal Ice Zone length [km]')
    ax.set_ylabel(r'SIA [10$^6$ km$^2$]')


    fig, axes = plt.subplots(1,2, figsize=[11, 5])
    ax=axes[1]
    scat=ax.scatter(np.sqrt(df_MIZ), df_SIA.std(axis=1), c=df_MIZ.index.month, cmap='twilight', vmin=1, vmax=13, s=8)
    MIZnp=np.sqrt(df_MIZ).to_numpy().reshape(-1, 1)
    SIAnp=df_SIA.std(axis=1).to_numpy().reshape(-1, 1)
    fit=lrmodel.fit(MIZnp, SIAnp)
    r_corr = np.corrcoef(MIZnp.flatten(), SIAnp.flatten())[0,1]

    ax.plot(MIZnp, lrmodel.predict(MIZnp), color='k', linewidth=1)
    cbar=plt.colorbar(scat, ax=ax, drawedges=True, boundaries=np.arange(0,13)+0.5, ticks=np.arange(1,13))
    cbar.ax.set_yticklabels(monthnames[:12], fontsize=13)  # vertically oriented colorbar
    ax.set_xlabel(r'SQRT of Marginal Ice Zone length [km$^{1/2}$]', fontsize=14)
    ax.set_ylabel(r'SIA Uncertainty [10$^6$ km$^2$]', fontsize=14)
    #ax.text(0.05, 0.95, 'Y={:.4f}X+{:.4f}'.format(fit.coef_[0][0], fit.intercept_[0]), transform=ax.transAxes)
    ax.text(0.05, 0.95, r'R$^2$ = {:.3f}'.format(r_corr**2), transform=ax.transAxes, fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    if XH=='NH': ax.set_yticks(np.arange(0.1, 0.35, 0.05))

    print('Fraction of SIA variance explained by lin trend: {:.2f}'.format(1.-np.var(SIAnp-lrmodel.predict(MIZnp))/np.var(SIAnp)))
    print(np.corrcoef([MIZnp.flatten(), SIAnp.flatten()])[0,1]**2)
    #df_inner.corr(method=lambda x, y: stats.pearsonr(x, y)[1])[index].iloc[:-1]


    ax = axes[0]
    scat=ax.scatter(np.sqrt(df_SIA_osi), df_SIA.std(axis=1), c=df_SIA_osi.index.month, cmap='twilight', vmin=1, vmax=13, s=8)
    SIA_sqrtnp=np.sqrt(df_SIA_osi).to_numpy().reshape(-1, 1)
    SIAnp=df_SIA.std(axis=1).to_numpy().reshape(-1, 1)
    fit=lrmodel.fit(SIA_sqrtnp, SIAnp)
    r_corr = np.corrcoef(SIA_sqrtnp.flatten(), SIAnp.flatten())[0,1]

    #scat=ax.scatter(df_SIA_osi, df_SIA.std(axis=1), c=df_SIA_osi.index.month, cmap='twilight', vmin=1, vmax=13)
    ax.plot(SIA_sqrtnp, lrmodel.predict(SIA_sqrtnp), color='k', linewidth=1)
    cbar=plt.colorbar(scat, ax=ax, drawedges=True, boundaries=np.arange(0,13)+0.5, ticks=np.arange(1,13))
    cbar.ax.set_yticklabels(monthnames[:12], fontsize=13)  # vertically oriented colorbar
    ax.set_ylabel(r'SIA Uncertainty [10$^6$ km$^2$]', fontsize=14)
    #ax.set_xlabel(r'SIA [10$^6$ km$^2$]')
    ax.set_xlabel(r'SQRT of SIA [10$^3$ km]', fontsize=14)
    ax.text(0.05, 0.95, r'R$^2$ = {:.3f}'.format(r_corr**2), transform=ax.transAxes, fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    if XH=='NH': ax.set_yticks(np.arange(0.1, 0.35, 0.05))
    fig.tight_layout()
    #fig.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/UNC_SIA_MIZ_AA_v2.png', dpi=300)

#minimal version of the boxplot with, without bias in comparison and with our unc bars
if 0 and data_SIAUHH:
        lrmodel = LinearRegression()
        monthwidth=5
        df_SIAUHH_diff=df_SIAUHH.copy()
        df_SIAUHH_diff.dropna(inplace=True)
        df_SIAUHH_anomaly=df_SIAUHH_diff.copy()
        products=df_SIAUHH_diff.keys()
        df_SIAUHH_diff['mpm_SIA']=df_SIAUHH_diff.mean(axis=1)
        for product in products:
            df_SIAUHH_diff[product]=df_SIAUHH_diff[product]-df_SIAUHH_diff['mpm_SIA']
            df_SIAUHH_anomaly[product]=df_SIAUHH_diff[product].copy()


        fig4, ax4 = plt.subplots()
        fig4.subplots_adjust(bottom=0.20)

        for tmonth in np.arange(1,13):
            df_SIAUHH_diff_tmp=df_SIAUHH_diff.loc[df_SIAUHH_diff.index.month==tmonth].copy()
            df_SIAUHH_diff_tmp.dropna(inplace=True)
            for product in products:
                df_SIAUHH_diff_tmp[product]=df_SIAUHH_diff_tmp[product]-df_SIAUHH_diff_tmp[product].mean()

            bplot1=ax4.boxplot(df_SIAUHH_diff_tmp[products].to_numpy().flatten(), positions=[monthwidth*tmonth+1], widths=1, labels=[monthnames[tmonth-1]] , whis=(5, 95), sym='', patch_artist=True)#labels=['I.P. nobias'],
            #ax2.fill_between([monthwidth*tmonth+i-0.5, monthwidth*tmonth+i+0.5], [-1, -1], [1,1], color=SIA_probs['color'][key], alpha=0.3)
            #bplot1['boxes'][0].set_facecolor('darkgray')
            bplot1['boxes'][0].set_facecolor('mediumpurple')

            #bplot=ax4.boxplot(df_SIAUHH_diff_tmp['NSIDC_BT'].to_numpy().flatten(), positions=[monthwidth*tmonth-1], widths=1, labels=['OSI_nobias'], whis=(5, 95), sym='', patch_artist=True)
            ##ax2.fill_between([monthwidth*tmonth+i-0.5, monthwidth*tmonth+i+0.5], [-1, -1], [1,1], color=SIA_probs['color'][key], alpha=0.3)
            #bplot['boxes'][0].set_facecolor('g')


            df_SIAUHH_anomaly_tmp=df_SIAUHH_anomaly.loc[df_SIAUHH_anomaly.index.month==tmonth].copy()
            df_SIAUHH_anomaly_tmp.dropna(inplace=True)

            bplot2=ax4.boxplot(df_SIAUHH_anomaly_tmp[products].to_numpy().flatten(), positions=[monthwidth*tmonth], widths=1, labels=[''], whis=(5, 95), sym='', patch_artist=True)#labels=['I.P.'],
            #ax2.fill_between([monthwidth*tmonth+i-0.5, monthwidth*tmonth+i+0.5], [-1, -1], [1,1], color=SIA_probs['color'][key], alpha=0.3)
            #bplot2['boxes'][0].set_facecolor('gray')
            bplot2['boxes'][0].set_facecolor('darkslateblue')


            #mean SIA std of the period of SIAUHH
            df_SIA_tmp=df_SIA_month.loc[df_SIA_month.index.month==tmonth].copy()
            df_SIA_tmp=df_SIA_tmp.loc[df_SIA_tmp.index>df_SIAUHH_diff_tmp.index[0]]
            df_SIA_tmp=df_SIA_tmp.loc[df_SIA_tmp.index<df_SIAUHH_diff_tmp.index[-1]]
            ens_keys=df_SIA_tmp.keys()
            df_SIA_tmp_mean=df_SIA_tmp.mean(axis=1).copy()
            for ens_key in ens_keys:
                df_SIA_tmp[ens_key]=df_SIA_tmp[ens_key]-df_SIA_tmp_mean

            bplot3=ax4.boxplot(df_SIA_tmp.to_numpy().flatten(), positions=[monthwidth*tmonth+2], widths=1, labels=[''], whis=(5, 95), sym='', patch_artist=True)#labels=['Ensemble'],
            #ax2.fill_between([monthwidth*tmonth+i+inc_index-0.5, monthwidth*tmonth+i+inc_index+0.5], [-1, -1], [1,1], color='k', alpha=0.3)
            bplot3['boxes'][0].set_facecolor('#8EB69B')#which is lighter than gray!

        ymin, ymax =ax4.get_ylim()
        ymax=2.
        #for tmonth in np.arange(1,13):
        #    ax4.text(monthwidth*tmonth+1, ymax+0.06 , monthnames[tmonth-1], horizontalalignment='center')

        ax4.set_xticklabels(ax4.get_xticklabels(),rotation=90)
        ax4.set_ylabel(r'$\Delta$ SIA [10$^6$ km$^2$]', fontsize=14)
        ax4.set_ylim([-2.2, 2])
        #ax4.labelrotation=45
        ax4.set_xticks(ax4.get_xticks(), ax4.get_xticklabels(), rotation=0, ha='center')
        ax4.tick_params(axis='both', labelsize=13)
        if XH=='SH':
            ax4.legend([bplot2['boxes'][0], bplot1['boxes'][0], bplot3['boxes'][0]], ['Inter Product', 'Inter Product No-Bias', 'Ensemble'], loc='upper left', fontsize='large')
        fig4.tight_layout()
        #plt.legend(['Inter Prod', 'Inter Prod - No Bias', 'Ensemble'], ax=ax4)
        #plt.legend([bplot1['boxes'], bplot2['boxes'], bplot3['boxes']],['Inter Prod', 'Inter Prod - No Bias', 'Ensemble'])
        #fig4.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/SIA_box_A.png', dpi=300)

#new version, showing more aspects of boxplot and ts
if 0 and data_SIAUHH: #dist SIA all month
        marker_styles = ['o','o', '^', 's', 'D', 'p', '*', 'h', 'H', '+', 'x', '|', '_']

        df_SIAUHH_diff=df_SIAUHH.copy()
        df_SIAUHH_diff.dropna(inplace=True)
        products=df_SIAUHH_diff.keys()
        df_SIAUHH_diff['mpm_SIA']=df_SIAUHH_diff.mean(axis=1)
        for product in products:
            if 0:
                df_SIAUHH_diff[product]=df_SIAUHH_diff[product]-df_SIAUHH_diff['mpm_SIA']


        figts, axts = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        inc_index=1
        monthwidth=11
        for tmonth in np.arange(1,13):
            if tmonth==2 or tmonth==9:
                for product in products:
                    df_SIAUHH_diff[product].loc[df_SIAUHH_diff.index.month==tmonth].plot(ax=axts, c=SIA_probs['color'][product], marker=marker_styles[tmonth], label={2:'Feb - ', 9:'Sep - '}[tmonth]+product)
            #if tmonth==2:
            #    for product in products:
            #        df_SIAUHH_diff[product].loc[df_SIAUHH_diff.index.month==tmonth].plot(ax=axts, c=SIA_probs['color'][product], ls='--', legend=False)
            #diff_keys=df_SIAUHH_tmp.keys().drop('OSI_SAF')#.drop('Ens_m')
            #diff_keys=df_SIAUHH.keys()
            df_SIA_tmp=df_SIA_month.loc[df_SIA_month.index.month==tmonth]


            df_SIAUHH_tmp=df_SIAUHH.loc[df_SIAUHH.index.month==tmonth].copy()
            df_SIAUHH_tmp=df_SIAUHH_tmp.join(df_SIA_tmp['batch001i000'], how='inner')
            diff_keys=df_SIAUHH_tmp.keys()#.drop('mpm_SIA')
            df_SIAUHH_tmp.dropna()
            df_SIAUHH_tmp.index=df_SIAUHH_tmp.index.year
            df_SIAUHH_tmp['mpm_SIA']=df_SIAUHH_tmp[diff_keys].mean(axis=1)
            df_SIAUHH_tmp['mpstd_SIA']=df_SIAUHH_tmp[diff_keys].std(axis=1)

            df_SIA_tmp.index=df_SIA_tmp.index.year


            for i, key in enumerate(diff_keys):

                dSIA = df_SIAUHH_tmp[key]-df_SIAUHH_tmp['mpm_SIA']
                #ax1.boxplot(dSIA[np.isnan(dSIA)==0], positions=[monthwidth*tmonth+i], widths=1, labels=[key], whis=(5, 95), sym='')
                bplot=ax1.boxplot(dSIA[np.isnan(dSIA)==0], positions=[monthwidth*tmonth+i], widths=1, labels=[key], whis=(5, 95), sym='', patch_artist=True)
                #ax2.fill_between([monthwidth*tmonth+i-0.5, monthwidth*tmonth+i+0.5], [-1, -1], [1,1], color=SIA_probs['color'][key], alpha=0.3)
                bplot['boxes'][0].set_facecolor(SIA_probs['color'][key])

                bplot=ax2.boxplot(dSIA[np.isnan(dSIA)==0]-np.nanmean(dSIA), positions=[monthwidth*tmonth+i], widths=1, labels=[key], whis=(5, 95), sym='', patch_artist=True)
                #ax2.fill_between([monthwidth*tmonth+i-0.5, monthwidth*tmonth+i+0.5], [-1, -1], [1,1], color=SIA_probs['color'][key], alpha=0.3)
                bplot['boxes'][0].set_facecolor(SIA_probs['color'][key])
                #c=SIA_probs['color'][key]
            if 0:
                dSIA = df_SIA_tmp['batch001i000']-df_SIAUHH_tmp['mpm_SIA']
                #ax1.boxplot(dSIA[np.isnan(dSIA)==0], positions=[monthwidth*tmonth+i+inc_index], widths=1, labels=['batch001i000'], whis=(5, 95), sym='')
                bplot=ax1.boxplot(dSIA[np.isnan(dSIA)==0], positions=[monthwidth*tmonth+i+inc_index], widths=1, labels=['batch001i000'], whis=(5, 95), sym='', patch_artist=True)
                bplot['boxes'][0].set_facecolor('gray')

                bplot=ax2.boxplot(dSIA[np.isnan(dSIA)==0]-np.nanmean(dSIA), positions=[monthwidth*tmonth+i+inc_index], widths=1, labels=['batch001i000'], whis=(5, 95), sym='', patch_artist=True)
                #ax2.fill_between([monthwidth*tmonth+i+inc_index-0.5, monthwidth*tmonth+i+inc_index+0.5], [-1, -1], [1,1], color='k', alpha=0.3)
                bplot['boxes'][0].set_facecolor('gray')
            if 0:
                SIA_tmp_std=df_SIA_tmp.std(axis=1).loc[df_SIA_tmp.index>df_SIAUHH_tmp.index[0]]
                SIA_tmp_std=SIA_tmp_std.loc[SIA_tmp_std.index<df_SIAUHH_tmp.index[-1]].mean()
                #mean SIA std of the period of SIAUHH

                print('Month: {}, STD: {}'.format(tmonth, SIA_tmp_std))
                dist_tmp=np.random.normal(size=10000)*SIA_tmp_std
                bplot=ax2.boxplot(dist_tmp, positions=[monthwidth*tmonth-1.5], widths=1, labels=['Sample Spread'], whis=(5, 95), sym='', patch_artist=True)
                #ax2.fill_between([monthwidth*tmonth+i+inc_index-0.5, monthwidth*tmonth+i+inc_index+0.5], [-1, -1], [1,1], color='k', alpha=0.3)
                bplot['boxes'][0].set_facecolor('darkgray')#which is lighter than gray!
            print('batch001i000 Variance: {}'.format((dSIA[np.isnan(dSIA)==0]-np.nanmean(dSIA)).var()))
            spread_ens=np.mean(np.var(df_SIA_tmp.T, axis=0))
            print('Yearly-mean SIA ensemble spread in [{}]: {:0f}'.format(tmonth, spread_ens))

            if 1:#multi product spread
                dist_tmp=np.random.normal(size=10000)*df_SIAUHH_tmp['mpstd_SIA'].mean()
                bplot=ax2.boxplot(dist_tmp, positions=[monthwidth*tmonth-2.5], widths=1, labels=['Product Spread'], whis=(5, 95), sym='', patch_artist=True)
                #ax2.fill_between([monthwidth*tmonth+i+inc_index-0.5, monthwidth*tmonth+i+inc_index+0.5], [-1, -1], [1,1], color='k', alpha=0.3)
                bplot['boxes'][0].set_facecolor('white')

            if XH=='SH':
                ax1.text(monthwidth*tmonth+2.5, 3.09 , monthnames[tmonth-1], horizontalalignment='center')
                ax2.text(monthwidth*tmonth+2.5, 0.62, monthnames[tmonth-1], horizontalalignment='center')
            elif XH=='NH':
                ax1.text(monthwidth*tmonth+2.5, 2.35 , monthnames[tmonth-1], horizontalalignment='center')
                ax2.text(monthwidth*tmonth+2.5, 0.59, monthnames[tmonth-1], horizontalalignment='center')
        #ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
        #ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
        ax1.set_xticklabels('')
        ax2.set_xticklabels('')

        ax1.set_ylabel(r'SIA - OSI SAF SIA ($\Delta$ SIA), [10$^6$ km$^2$]', fontsize=14)
        ax2.set_ylabel(r'$\Delta$ SIA, No Bias [10$^6$ km$^2$]', fontsize=14)

        if XH=='SH':
            ax1.set_ylim([-2.0, 3.0])
            ax2.set_ylim([-1.1, 1.1])
            axts.set_ylabel(r'SIA [10$^6$ km$^2$]', fontsize=14)
            axts.set_title('Antarctic', fontsize=15)
        elif XH=='NH':
            ax1.set_ylim([-1.2, 1.5])
            ax2.set_ylim([-0.65, 0.65])
            axts.set_ylabel(r'SIA [10$^6$ km$^2$]', fontsize=14)
            axts.set_title('Arctic', fontsize=15)
        ax1.legend()
        ax2.legend()

        if XH=='SH': axts.legend(loc= (0.085, 0.25), ncol=2, fontsize='large')
        axts.set_xlim([df_SIAUHH_diff.index[0], df_SIAUHH_diff.index[-1]])
        axts.set_ylim([0, 20.])
        axts.tick_params(axis='both', labelsize=13)
        figts.tight_layout()
        #fig1.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/diff_SIA_NH_biased.png', dpi=300)
        #fig2.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/diff_SIA_NH.png', dpi=300)
        #fig3.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/diff_SIA_NH_label.png', dpi=300)
        #figts.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/SIA_ts_prods_AA.png', dpi=300)




if 1:#SIA trends
    len_trenplot=12 #12: no yearly mean, 13: with yearly mean
    trend_allmonth = np.zeros((len(df_SIA.keys()),len_trenplot))
    if data_SIAUHH: trend_UHH_allmonth=np.zeros((len(df_SIAUHH.keys()),len_trenplot))
    if data_SIAUHH and trends_rel1981_2010: clim_sia_UHH=np.zeros_like(trend_UHH_allmonth)
    fig_tren, axes_tren= plt.subplots(len_trenplot, 1, sharex=True, figsize=(6,16))
    for tmonth in np.arange(1,len_trenplot+1):
        print('Month: {}'.format(tmonth))
        ax_tren=axes_tren[np.max(tmonth-1, 0)]
        if tmonth==13: #replace by yearly average
            #This is tecnicaly underestimates the ensemble uncertainty, because there is no temp correlation between month in the DS
            df_SIA_tmonth=df_SIA_month.groupby(df_SIA_month.index.year).mean()
        else:
            df_SIA_tmonth=df_SIA_month.loc[df_SIA_month.index.month==tmonth].copy()
            df_SIA_tmonth.index=df_SIA_tmonth.index.year

        #print(df_SIA_tmonth)
        if data_SIAUHH:
            if tmonth==13: #replace by yearly average
                df_SIAUHH_tmonth=df_SIAUHH.groupby(df_SIAUHH.index.year).mean()[:-1]#cut the last because 2025 is not a full year
            else:
                df_SIAUHH_tmonth=df_SIAUHH.loc[df_SIAUHH.index.month==tmonth].copy()
                df_SIAUHH_tmonth.index=df_SIAUHH_tmonth.index.year
            df_SIAUHH_tmonth.dropna(inplace=True)

        if 0:
            fig, ax = plt.subplots()
            ax.set_title('{}'.format(tmonth))
            for key in df_SIAUHH_tmonth.keys():
                df_SIAUHH_tmonth[key].plot()
                #plt.scatter(df_SIAUHH_tmonth[key].index, df_SIA_tmonth[key])

        lrmodel = LinearRegression()
        trends=np.zeros(len(df_SIA_tmonth.keys()))
        trends2016=np.zeros(len(df_SIA_tmonth.keys()))
        if trends_rel1981_2010: clim_sia=np.zeros_like(trends)
        for i, batchi in enumerate(df_SIA_tmonth.keys()):
            x=df_SIA_tmonth.index.to_numpy()
            y=df_SIA_tmonth[batchi].to_numpy()
            if trends_rel1981_2010:
                clim_mask=np.logical_and(x>=1981, x<=2010)
                clim_sia[i]=np.mean(y[clim_mask])

            if data_SIAUHH:
                x_UHH=df_SIAUHH_tmonth.index

            if 1 and data_SIAUHH:#match time periods MC with UHH
                mask=np.array([year in x_UHH for year in x], dtype=bool)
                x=x[mask][np.isnan(y[mask])==0].reshape(-1,1)
                y=y[mask][np.isnan(y[mask])==0].reshape(-1,1)

            else:
                x=x[np.isnan(y)==0].reshape(-1,1)
                y=y[np.isnan(y)==0].reshape(-1,1)

            lrmodel.fit(x, y)#
            if len(np.shape(lrmodel.coef_))==2:
                trends[i]=lrmodel.coef_[0,0].copy()*10.

            else:
                trends[i]=lrmodel.coef_[0].copy()*10.
                print('Shape==2')

            if 0:
                x=df_SIA_tmonth.index.to_numpy()
                mask=x>=0
                y=df_SIA_tmonth[batchi].to_numpy()
                x=x[mask][np.isnan(y[mask])==0].reshape(-1,1)
                y=y[mask][np.isnan(y[mask])==0].reshape(-1,1)

                lrmodel.fit(x, y)#
                if len(np.shape(lrmodel.coef_))==2:
                    trends2016[i]=lrmodel.coef_[0,0].copy()
                else:
                    trends2016[i]=lrmodel.coef_[0].copy()
                #plt.plot(x, lrmodel.predict(x), color='gray', linewidth=1, alpha=0.4)
        #plot2=(df_SIA_tmonth.reset_index()).T.plot(kind='box', ax=ax)

        if trends_rel1981_2010:
            trends=trends/clim_sia*100.
        trend_allmonth[:,tmonth-1]=trends
        #fig.savefig('/home/dusch/Dropbox/Documents/IGS_newcastle2024/presentation/figures/box_sh_comp_forlegend.png', dpi=300)

        #df_SIA_osi_sep

        if data_SIAUHH:
            for i_key, key in enumerate(df_SIAUHH_tmonth.keys()):

                x=df_SIAUHH_tmonth.index.to_numpy()
                y=df_SIAUHH_tmonth[key].to_numpy()
                x=x[np.isnan(y)==0].reshape(-1,1)
                y=y[np.isnan(y)==0].reshape(-1,1)

                lrmodel.fit(x, y)#
                trend_UHH=lrmodel.coef_[0].copy()*10.

                if trends_rel1981_2010:
                    clim_mask=np.logical_and(x>=1981, x<=2010)
                    clim_sia_UHH[i_key, tmonth-1]=np.mean(y[clim_mask])
                    #trend_UHH_allmonth=trend_UHH_allmonth/clim_sia_UHH*100.
                    trend_UHH=trend_UHH/clim_sia_UHH[i_key, tmonth-1]*100.

                ax_tren.plot([trend_UHH,trend_UHH],[0,190], c=SIA_probs['color'][key], ls='--', label=key)
                trend_UHH_allmonth[i_key, tmonth-1]=trend_UHH[0]

                if 0:
                    x=df_SIAUHH_tmonth.index.to_numpy()
                    mask=x>=2016
                    y=df_SIAUHH_tmonth[key].to_numpy()
                    x=x[mask][np.isnan(y[mask])==0].reshape(-1,1)
                    y=y[mask][np.isnan(y[mask])==0].reshape(-1,1)

                    lrmodel.fit(x, y)#
                    trend_UHH2016=lrmodel.coef_[0].copy()


        #plt.figure()
        #hist=plt.hist(trends, bins=20, density=True)
        x_axis=np.linspace(np.min(trends)-0.3, np.max(trends)+0.3, 1000)
        y_axis=np.zeros_like(x_axis)
        if not trends_rel1981_2010:
            std=0.01
        else:
            std=0.1
        for trend in trends:
            y_axis+=1./(std*np.sqrt(2*np.pi))*np.exp(-0.5*(x_axis-trend)**2/std**2)/len(trends)

        print('5 and 95 percentiles: in SIA trend: {:.3f}; {:.3f}'.format(np.quantile(trends, [0.05, 0.95])[0], np.quantile(trends, [0.05, 0.95])[1]))
        print('STD in SIA trend: {:.4f}'.format(np.std(trends)))
        #print(trends)
        if 1:# center distribution on UHH OSI SAF
            ax_tren.fill_between(x_axis-(np.mean(trends)-np.mean(trend_UHH_allmonth[df_SIAUHH_tmonth.keys()=='OSI_SAF', tmonth-1])), np.zeros_like(x_axis), y_axis, color='#757f6d', alpha=0.6, label='Ensemble')
        elif 0:#center distribution on UHH mean
            ax_tren.fill_between(x_axis-(np.mean(trends)-np.mean(trend_UHH_allmonth[:, tmonth-1])), np.zeros_like(x_axis), y_axis, color='#757f6d', alpha=0.6, label='Ensemble')
        else:#do not center on any other estimate
            ax_tren.fill_between(x_axis, np.zeros_like(x_axis), y_axis, color='#757f6d', alpha=0.6, label='Ensemble')

        if not trends_rel1981_2010:
            if tmonth==13:
                ax_tren.set_ylim([0, 40])
            else:
                ax_tren.set_ylim([0, 30])
        else:
            ax_tren.set_ylim([0, 3])
        ax_tren.set_yticks([])
        if tmonth==1 and XH=='NH':
            ax_tren.legend(loc=(0.12, 0.1))
        if tmonth==7:
            ax_tren.set_ylabel('Probability', fontsize=14)
        #if tmonth==13:
        #    ax_tren.set_xticklabels(ax_tren.get_xticks(), fontsize=14)
        ax_tren.text(0, 0.5, monthnames[tmonth-1], verticalalignment='center', horizontalalignment='left', transform=ax_tren.transAxes, fontsize=13)
    if XH=='SH':
        if not trends_rel1981_2010:
            #ax_tren.set_xlim([-0.20, 0.1])#for all month
            ax_tren.set_xlim([-0.345, 0.325])#for Mar, Sep, YM only, Same range as North
    else:
        if not trends_rel1981_2010:
            ax_tren.set_xlim([-0.92, -0.25])

    ax_tren.set_xlabel(r'SIA Trend 1979-2024 [10$^6$ km$^2$/dec.]', fontsize=13)
    if trends_rel1981_2010: ax_tren.set_xlabel(r'SIA Trend 1979-2024 [percent/dec.]', fontsize=13)
    ax_tren.tick_params(axis='both', labelsize=12)

    #ax_tren.plot([0,0],[0,40], c='k', ls=':')
    #plt.legend()
    seaborn.despine(left=True, bottom=False, right=True, fig=fig_tren)
    #ax_tren.savefig('/home/dusch/Dropbox/Documents/IGS_newcastle2024/presentation/figures/histtrend2016.png', dpi=200)
    #fig_tren.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/histtrend_7924_AA_0309ym.png', dpi=300)
    #fig_tren.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/histtrend_rel_7924_A.png', dpi=300)
    #plt.scatter(2012, 6)

    #axes_tren[0].legend(loc='upper left')
    #plt.legend
    if data_SIAUHH:
        #if trends_rel1981_2010: trend_UHH_allmonth=trend_UHH_allmonth/clim_sia_UHH*100.
        print('Trends UHH SIA in Month {}:'.format(tmonth))
        for i in range(np.shape(trend_UHH_allmonth)[0]):
            print('{}    {:.3f}'.format(df_SIAUHH_tmonth.keys()[i],trend_UHH_allmonth[i,tmonth-1]))

    #compare spread in trends between ensamble and UHH
    if 1 and data_SIAUHH:
        #for tmonth in np.range(1,13):
        trend_range_UHH = trend_UHH_allmonth.max(axis=0) - trend_UHH_allmonth.min(axis=0)
        n_bootst=51
        trend_range_i=np.zeros([n_bootst, 12])
        for i in range(n_bootst):
            sample=random.sample(range(n_noise), 3)
            trend_range_i[i,:] = trend_allmonth[sample,:].max(axis=0)-trend_allmonth[sample,:].min(axis=0)
        trend_range=np.median(trend_range_i, axis=0)
        fig, ax = plt.subplots()
        for i in range(n_bootst):
            ax.scatter(trend_range_UHH, trend_range_i[i,:], c='gray', alpha=0.3)
        ax.scatter(trend_range_UHH, trend_range, c='k')
        ax.set_xlim([0,0.16])
        ax.set_ylim([0,0.16])
        if XH=='NH':
            ax.set_title('Arctic', fontsize=15)
        else:
            #ax.set_xlim([0,0.11])
            #ax.set_ylim([0,0.11])
            ax.set_title('Antarctic', fontsize=15)

        ax.set_xlabel('SIA Trend Max-Min, UHH [10$^6$ km$^2$/dec.]', fontsize=13)
        ax.set_ylabel('SIA Trend Max-Min, Ensemble [10$^6$ km$^2$/dec.]', fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        ax.plot([0,1], [0,1], c='k', ls=':')
        print(trend_range/trend_range_UHH)
        fig.tight_layout()
        #fig.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/trends_MAXMIN_AA.png', dpi=300)

if 0 and regional:
    if XH=='NH':
        reg_names={
            '1':'Central_Arctic', '2':'Beaufort', '3':'Chukchi', '4':'E_Siberian', '5':'Laptev', '6':'Kara', '7':'Barents', '8':'E_Greenland',
            '9':'Baffin','10':'St_Lawr','11':'Hudson','12':'Can_Arch','13':'Bering','14':'Okhotsk','15':'Japan','16':'Bohai','17':'Baltic','18':'Gulf_Alaska'
                   }
    if XH=='SH':
        reg_names={
            '1':'Weddell','2':'Indian','3':'Pacific','4':'Ross','5':'BellAmund'
            }
    reg_colors={'1':'steelblue', '2':'orange', '3':'g', '4':'purple', '5':'r', '6':'pink', '7':'saddlebrown', '8':'y', '9':'b', '10':'m', '11':'k', '12':'c', '13':'darkblue', '14':'chocolate', '15':'darkgreen', '16':'orange', '17':'lightgray', '18': 'teal'}


    fig, ax = plt.subplots()
    #if 1:
    #    tmonth=9
    for tmonth in[2,9]:
        for region in regions:
            if tmonth==2:
                #if int(region)>10: ls=':'
                #else: ls='--'
                ls='--'
            else:
                #if int(region)>10: ls='-.'
                #else: ls='-'
                ls='-'
            #for batch_sample in SIA_noise_reg_m[region].keys():
            #    ax.plot(SIA_noise_reg_m[region][batch_sample].loc[SIA_noise_reg_m[region].index.month==tmonth], alpha=0.4, c='gray')
            mean_tmp=SIA_noise_reg_m[region].loc[SIA_noise_reg_m[region].index.month==tmonth].mean(axis=1)
            std_tmp=SIA_noise_reg_m[region].loc[SIA_noise_reg_m[region].index.month==tmonth].std(axis=1)
            line=ax.plot(mean_tmp, label={2:'Feb. ', 9:'Sep. '}[tmonth]+reg_names[region], ls=ls, c=reg_colors[region])
            #ax.fill_between(mean_tmp.loc[mean_tmp.index.month==tmonth].index, mean_tmp-2*std_tmp, mean_tmp+2*std_tmp, color=line[0].get_color(), alpha=0.4)
            quan5=SIA_noise_reg_m[region].loc[SIA_noise_reg_m[region].index.month==tmonth].quantile(0.05,axis=1)
            quan5[quan5<=0]=0.
            quan95=SIA_noise_reg_m[region].loc[SIA_noise_reg_m[region].index.month==tmonth].quantile(0.95,axis=1)
            ax.fill_between(mean_tmp.loc[mean_tmp.index.month==tmonth].index, quan5, quan95, color=line[0].get_color(), alpha=0.4)

        if 0:
            #df_SIA_tmp=df_SIA_month.loc[df_SIA_month.index.month==tmonth]
            mean_tmp=df_SIA_month.loc[df_SIA_month.index.month==tmonth].mean(axis=1)
            std_tmp=df_SIA_month.loc[df_SIA_month.index.month==tmonth].std(axis=1)
            line=ax.plot(mean_tmp, label={2:'Feb. ', 9:'Sep. '}[tmonth]+'Total', ls=ls, c=reg_colors[region])
            #ax.fill_between(mean_tmp.loc[mean_tmp.index.month==tmonth].index, mean_tmp-2*std_tmp, mean_tmp+2*std_tmp, color=line[0].get_color(), alpha=0.4)
            quan5=df_SIA_month.loc[df_SIA_month.index.month==tmonth].quantile(0.05,axis=1)
            quan5[quan5<=0]=0.
            quan95=df_SIA_month.loc[df_SIA_month.index.month==tmonth].quantile(0.95,axis=1)
            ax.fill_between(mean_tmp.loc[mean_tmp.index.month==tmonth].index, quan5, quan95, color=line[0].get_color(), alpha=0.4)

    plt.legend(ncol=2)

    #if tmonth==2:
    #    ax.set_ylabel(r'February SIA [10$^6$ km$^2$]', fontsize=14)
    #elif tmonth==9:
    #    ax.set_ylabel(r'September SIA [10$^6$ km$^2$]', fontsize=14)
    #else:
    ax.set_title('Hem: {}, Month: {}'.format(XH, tmonth))
    ax.set_ylabel('SIA [m km2]')
    ax.set_xlabel('Time', fontsize=14)
    ax.set_xlim([mean_tmp.index[0], mean_tmp.index[-1]])
    #ax.set_ylim([0,9.5])

    #plt.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/second_draft/figures/regional_02SH_legend.png', dpi=300)
    if 0:
        fig, ax = plt.subplots()

        for region in regions:

            #for batch_sample in SIA_noise_reg_m[region].keys():
            #    ax.plot(SIA_noise_reg_m[region][batch_sample], alpha=0.4, c='gray')
            ts_reg_tmp=SIA_noise_reg_m[region].mean(axis=1)
            if int(region)>10: ls='--'
            else: ls='-'
            ax.plot(ts_reg_tmp, label=reg_names[region], ls=ls)
            if region=='1':
                ts_sum_reg=ts_reg_tmp
            else:
                ts_sum_reg+=ts_reg_tmp

        ax.plot(df_SIA_month.mean(axis=1), label='total')
        ax.plot(ts_sum_reg, label='sum')

        plt.legend()


        fig, ax = plt.subplots()
        for region in regions:
            if int(region)>10: ls='--'
            else: ls='-'
            #for batch_sample in SIA_noise_reg_m[region].keys():
            #    ax.plot(SIA_noise_reg_m[region][batch_sample], alpha=0.4, c='gray')
            ax.plot(SIA_noise_reg_m[region].mean(axis=1), label=reg_names[region], ls=ls)
        plt.legend()

    testmonth=9
    fig, ax = plt.subplots()
    for region in regions:
        ax.scatter( SIA_noise_reg_m[region].loc[SIA_noise_reg_m[region].index.month==testmonth].mean(axis=1),SIA_noise_reg_m[region].loc[SIA_noise_reg_m[region].index.month==testmonth].std(axis=1), label=reg_names[region])
    plt.legend()
    ax.set_ylabel(r'SIA Uncertainty [10$^6$ km$^2$]')
    ax.set_xlabel(r'SIA [10$^6$ km$^2$]')
    ax.set_title('Hem: {}, Mon: {}'.format(XH, testmonth))

    fig, ax = plt.subplots()
    for region in regions:
        ax.scatter( SIA_noise_reg_m[region].loc[SIA_noise_reg_m[region].index.month==testmonth].mean(axis=1),SIA_noise_reg_m[region].loc[SIA_noise_reg_m[region].index.month==testmonth].std(axis=1)/SIA_noise_reg_m[region].loc[SIA_noise_reg_m[region].index.month==testmonth].mean(axis=1), label=reg_names[region])
    ax.scatter( df_SIA_month.loc[df_SIA_month.index.month==testmonth].mean(axis=1),df_SIA_month.loc[df_SIA_month.index.month==testmonth].std(axis=1)/df_SIA_month.loc[df_SIA_month.index.month==testmonth].mean(axis=1), label='Total')

    plt.legend()
    ax.set_ylabel(r'Uncertainty / SIA []')
    ax.set_xlabel(r'SIA [10$^6$ km$^2$]')
    ax.set_title('Hem: {}, Mon: {}'.format(XH, testmonth))
