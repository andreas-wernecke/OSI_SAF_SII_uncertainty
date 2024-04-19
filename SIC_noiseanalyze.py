#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:55:12 2022

@author: dusch
"""


#import gdal
import netCDF4 as nc
import numpy as np
#import pandas as pd
from scipy import ndimage, stats, odr
#import skgstat as skg
#from scipy.interpolate import griddata
#from scipy.signal import detrend
import pandas as pd
pd.options.mode.use_inf_as_na = True
#import xarray as xr
import pywt

import os, sys, calendar, csv#, glob#, datetime
from os import path
home=os.getenv('HOME')
#import matplotlib.dates as dates
#import random

genpath=home+'/Dropbox/scripts/'
sys.path.append(genpath)

#import taylorDiagram
#import rasterio, rasterio.mask#, fiona
#import GPy
#import mpl_toolkits
#from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs                   # import projections
import cartopy.feature as cf                 # import features
import scipy.signal, geopandas, shapely, shapely.vectorized, warnings
from scipy.interpolate import interpn
import matplotlib.pyplot as plt

from datetime import date, timedelta, datetime
#from scipy.fftpack import rfft, irfft, fftfreq
#def gausslow(sig, size=np.array([101,101, 11]), mu='none'):
#    sigsp=10.
#    sigtmp=2.
#    mu='none'
#    size=np.array([101,101,11])
#    if mu=='none':
#        mu=(size-1)/2
#    u, v, w = np.meshgrid(np.arange(size[0]), np.arange(size[1]), np.arange(size[2]))
#    #gausslow=np.exp(-((u-mu[0])**2+(v-mu[1])**2)/(2.*sig))*
def iplot(data):
    plt.figure()
    plt.imshow(data)
    plt.colorbar()


#---------------------------------------------------------------------------------------------------
###reading in CCI SIC


lcor_temp = 5.
lcor_sp_km =288.
dx=25.#in km, 50km nominal, corrected for typical arctic ocean (instead of NP)
XH='NH'
comp_cci_ens=False
#data='CCI'
#data='CCI_filt'
data='noise'
n_noise=20
full_year=False

regional=False

if full_year:
    years=[2015]
    months=np.arange(1,13)#13
else:
    years=np.arange(1979, 2024)
    months=[9]

calc_mean_noise=False
#---------------------------------------------------------------------------------------------------

if XH=='NH':
    #NorthSouth='north'
    xh='nh'
elif XH=='SH':
    #NorthSouth='south'
    xh='sh'
    
if data=='CCI':
    paths=['/media/dusch/T7 Shield/SIC/SIC/{:}{:02.0f}/'.format(XH, dx)]
    filterhandle='ESACCI-SEAICE-L4-SICONC-AMSR_{:02.1f}kmEASE2-{:}-'.format(dx, XH)
elif data=='noise':
    paths=[]
    for i in range(n_noise):
        if full_year:
            paths.append('/media/dusch/T7 Shield/SIC/noise/daily/{:}/2015/gauss/{:03d}/'.format(XH,i))
        else:
            #paths.append('/media/dusch/T7 Shield/SIC/noise/daily/{:}/september/gauss/{:03d}/'.format(XH,i))
            paths.append('/media/dusch/T7 Shield/SIC/noise/daily/{}/osi_sep_full/batch001i{:03d}/'.format(XH,i))


    #filterhandle='_lx{:.0f}km_lt{:.0f}d_ice_conc_cdr-v3p0_{:02.1f}km_ease2-{}-'.format(lcor_sp_km, lcor_temp, dx, NorthSouth)
    filterhandle='_lx{:.0f}km_lt{:.0f}d_ice_conc_{}_ease2-{:02.0f}0_cdr-v3p0_'.format(lcor_sp_km, lcor_temp, xh, dx)
    
elif data=='CCI_filt':
    paths=['/media/dusch/T7 Shield/SIC/noise/daily/{:}/2015/filtered/000/'.format(XH)]
    filterhandle='LPfiltered_filtered_lx{:.0f}km_lt{:.0f}dv2_ESACCI-SEAICE-L4-SICONC-AMSR_{:02.1f}kmEASE2-{:}-'.format(lcor_sp_km, lcor_temp, dx, XH)

if XH=='NH':
    regions_file='/media/dusch/T7 Shield/SI_regions_NSIDC/NSIDC-0780_SeaIceRegions_NH_v1.0.shp'
    #EPSG='3411'
elif XH=='SH':
    regions_file='/media/dusch/T7 Shield/SI_regions_NSIDC/NSIDC-0780_SeaIceRegions_SH-NASA_v1.0.shp'
    #EPSG='3412'


if 0:#those file are not in those dir...
    paths=['/media/dusch/T7 Shield/SIC/SIC/NH50/']
    years=np.arange(2002, 2017)
    months=[9]
    filterhandle='_lx{:.0f}km_lt{:.0f}dv2_ESACCI-SEAICE-L4-SICONC-AMSR_{:02.1f}kmEASE2-{:}-'.format(lcor_sp_km, lcor_temp, dx, XH)
# elif 0:
#     paths=[]
#     for i in range(n_noise):
#         paths.append('/media/dusch/AndreasExt/SIC/noise/daily/september/gauss/{:03d}/'.format(i))
#     years=np.arange(2002, 2017)
#     months=[9]
#     filterhandle='_lx{:.0f}km_lt{:.0f}dv2_ESACCI-SEAICE-L4-SICONC-AMSR_{:02.1f}kmEASE2-{:}-'.format(lcor_sp_km, lcor_temp, dx, XH)
# elif 0:
#     n_noise=100
#     paths=[]
#     for i in range(n_noise):
#         paths.append('/media/dusch/AndreasExt/SIC/noise/daily/2015/gauss/{:03d}/'.format(i))
#     years=[2015]
#     months=np.arange(1, 13)
#     filterhandle='_lx{:.0f}km_lt{:.0f}dv2_ESACCI-SEAICE-L4-SICONC-AMSR_{:02.1f}kmEASE2-{:}-'.format(lcor_sp_km, lcor_temp, dx, XH)
elif 0:
    paths=[]
    for i in range(n_noise):
        paths.append('/media/dusch/T7 Shield/SIC/noise/daily/min_max/gauss/{:03d}/'.format(i))
    years=[2015]
    months=np.arange(1, 13)
    filterhandle='_lx{:.0f}km_lt{:.0f}dv2_ESACCI-SEAICE-L4-SICONC-AMSR_{:02.1f}kmEASE2-{:}-'.format(lcor_sp_km, lcor_temp, dx, XH)

elif 0:
    paths=[]
    for i in range(n_noise):
        paths.append('/media/dusch/T7 Shield/SIC/noise/daily/single_corr/gauss/{:03d}/'.format(i))
    filterhandle='_lx{:.0f}km_lt{:.0f}dv2_ESACCI-SEAICE-L4-SICONC-AMSR_{:02.1f}kmEASE2-{:}-'.format(lcor_sp_km, lcor_temp, dx, XH)
elif 0:
    paths=[]
    for i in range(n_noise):
        paths.append('/media/dusch/T7 Shield/SIC/noise/daily/{}/filters_2015/fft/{:03d}/'.format(XH,i))
    filterhandle='_lx{:.0f}km_lt{:.0f}dv2_ESACCI-SEAICE-L4-SICONC-AMSR_{:02.1f}kmEASE2-{:}-'.format(lcor_sp_km, lcor_temp, dx, XH)
elif 0:#testing OSI SAF setup
    paths=[]
    for i in range(n_noise):
        paths.append('/media/dusch/T7 Shield/SIC/SIC/noise/test2/monthly/batch001i{:03d}/'.format(i))
    filterhandle='_lx{:.0f}km_lt{:.0f}d_ice_conc_cdr-v3p0_{:02.1f}km_ease2-north-'.format(lcor_sp_km, lcor_temp, dx)
elif 0:#testing OSI SAF setup september
    paths=[]
    for i in range(n_noise):
        paths.append('/media/dusch/T7 Shield/SIC/noise/daily/NH/osi_sep_full/batch001i{:03d}/'.format(i))
    filterhandle='_lx{:.0f}km_lt{:.0f}d_ice_conc_cdr-v3p0_{:02.1f}km_ease2-north-'.format(lcor_sp_km, lcor_temp, dx)
elif 0:
    paths=[]
    for i in range(n_noise):
        paths.append('/media/dusch/T7 Shield/SIC/noise/daily/{}/filters_2015/wavelet/{:03d}/'.format(XH, i))
    filterhandle='_lx{:.0f}km_lt{:.0f}dv2_ESACCI-SEAICE-L4-SICONC-AMSR_{:02.1f}kmEASE2-{:}-'.format(lcor_sp_km, lcor_temp, dx, XH)

# elif 0:#SH NOISE
#     n_noise=100
#     paths=[]
#     for i in range(n_noise):
#         paths.append('/media/dusch/AndreasExt/SIC/noise/daily/{:}/2015/gauss/{:03d}/'.format(XH,i))
#     years=[2015]
#     months=np.arange(1, 13)
#     filterhandle='_lx{:.0f}km_lt{:.0f}dv2_ESACCI-SEAICE-L4-SICONC-AMSR_{:02.1f}kmEASE2-{:}-'.format(lcor_sp_km, lcor_temp, dx, XH)
# elif 1:#CCI
#     paths=['/media/dusch/AndreasExt/SIC/SIC/{:}{:02.0f}/'.format(XH, dx)]
#     years=[2015]
#     months=np.arange(1, 13)
#     filterhandle='ESACCI-SEAICE-L4-SICONC-AMSR_{:02.1f}kmEASE2-{:}-'.format(dx, XH)


#figure out why mean(SIE_ensamble)>SIE(mean_ensamble)

SIAs_day=[]
SIEs_day=[]
SIAs_week=[]
SIEs_week=[]
SIAs_month=[]
SIEs_month=[]
meanNoise=[]
meanNoise15=[]
meanNoise15_filt=[]

if regional:
    SIAsreg_day={}
    SIEsreg_day={}
    SIAsreg_week={}
    SIEsreg_week={}
    SIAsreg_month={}
    SIEsreg_month={}
    meanNoisereg={}
    meanNoise15reg={}
    meanNoise15reg_filt={}

    sea_ids=np.arange(1,19, dtype=int)
    for sea_id in sea_ids:
        SIAsreg_day[str(sea_id)]=[]
        SIEsreg_day[str(sea_id)]=[]
        SIAsreg_week[str(sea_id)]=[]
        SIEsreg_week[str(sea_id)]=[]
        SIAsreg_month[str(sea_id)]=[]
        SIEsreg_month[str(sea_id)]=[]
        meanNoisereg[str(sea_id)]=[]
        meanNoise15reg[str(sea_id)]=[]
        meanNoise15reg_filt[str(sea_id)]=[]

for nc_dir in paths:#loop over n Samples if NOISE
    print('Folder: '+nc_dir)

    #read data

    SICs=[]
    if data=='CCI' and comp_cci_ens: SIC_sigts=[]
    SICs_date=[]
    status=[]
    SIC_noises=[]

    for year in years:
        print(year)
        for month in months:#
            fns=os.listdir(nc_dir+'{:04d}/{:02d}/'.format(year,month))
            dim=calendar.monthrange(year, month)[1]
            for dom in range(1,dim+1):

                datestr='{:02d}{:02d}'.format(month, dom)
                #print(datestr)
                fn=list(filter(lambda x : filterhandle+'{:}{:}'.format(year,datestr) in x, fns))#list with up to one finename in it
                #SICs=[]
                #SICs_year=[]
                #d_SICs=[]
                #smd_SICs=[]
                #ald_SICs=[]


                    #print(nc_dir+'{:04d}/{:02d}/'.format(year,month)+fileex+str(year)+datestr+'-fv2.1.nc')

                if len(fn)==1:

                    ds=nc.Dataset(nc_dir+'{:04d}/{:02d}/'.format(year,month)+fn[0], "r")
                    #print(ds.variables)
                    if data=='CCI':#original
                        SIC=ds.variables['ice_conc'][:,:]
                        SIC_raw=ds.variables['raw_ice_conc_values'][:,:]
                        SIC[SIC_raw.mask==False]=SIC_raw[SIC_raw.mask==False]
                        xgrid=ds.variables['xc'][:]
                        ygrid=ds.variables['yc'][:]
                    else:
                        SIC=ds.variables['SIC_sample'][:,:]
                        if data=='noise' and calc_mean_noise: SIC_noise=ds.variables['noise'][:,:]
                    if data=='CCI' and comp_cci_ens:total_err=ds.variables['total_standard_error'][:,:]
                    status_tmp=ds.variables['status_flag'][:,:]
                    #smear_err=ds.variables['smearing_standard_error'][:,:]
                    #alg_err=ds.variables['algorithm_standard_error'][:,:]
                    #tempz=ds.variable([])
                    if regional:
                        lon=ds.variables['lon'][:,:]
                        lat=ds.variables['lat'][:,:]

                    ds.close()

                    SICs.append(SIC[0,:,:])
                    status.append(status_tmp[0,:,:])
                    if data=='CCI' and comp_cci_ens:total_err[total_err.mask]=0.
                    if data=='CCI' and comp_cci_ens:SIC_sigts.append(total_err[0,:,:])
                    if data=='noise' and calc_mean_noise:
                        SIC_noise.mask=SIC.mask
                        SIC_noises.append(SIC_noise[0,:,:])
                    #smear_err[smear_err.mask]=0.
                    #smd_SICs.append(smear_err[0,SIC_mask==0])

                    SICs_date.append(datetime(year, month, dom))

                else: print('No file for: '+str(year)+datestr)

    SICs=np.asarray(SICs)
    if data=='CCI' and comp_cci_ens:SIC_sigts=np.asarray(SIC_sigts)
    if data=='noise' and calc_mean_noise:
        SIC_noises=np.asarray(SIC_noises)
        SICs_mask=SIC_noises==-32767
        SIC_noises=np.ma.array(SIC_noises, mask=SICs_mask)
        #meanNoise15=np.asarray(meanNoise15)
    SICs_date=np.asarray(SICs_date)
    status=np.asarray(status)

    SICs_mask=SICs==-32767
    SICs=np.ma.array(SICs, mask=SICs_mask)
    if data=='CCI' and comp_cci_ens:SIC_sigts=np.ma.array(SIC_sigts, mask=SICs_mask)

    if comp_cci_ens:#run with CCI first, then noise
        if data=='CCI':
            CCI_15=np.array(SICs>15, dtype='bool')
            noise_15diff=np.zeros_like(CCI_15, dtype='int')
            SIC_cci=SICs
            SIA_cci=np.sum(np.sum(SIC_cci, axis=1), axis=1)*dx*dx/100.
            SIE_cci=np.sum(np.sum(SIC_cci>15., axis=1), axis=1)*dx*dx
            #SIC_sigts_cci=SIC_sigts
        elif data=='CCI_filt':
            filt_15=np.array(SICs>15, dtype='bool')
            NoiseFilt_15diff=np.zeros_like(filt_15, dtype='int')
            SIC_filt=SICs
            SIA_filt=np.sum(np.sum(SIC_filt, axis=1), axis=1)*dx*dx/100.
            SIE_filt=np.sum(np.sum(SIC_filt>15., axis=1), axis=1)*dx*dx
        else:
            noise_15diff=noise_15diff + np.array(np.array(SICs>15, dtype='bool')!=CCI_15, dtype='int')
            NoiseFilt_15diff=NoiseFilt_15diff + np.array(np.array(SICs>15, dtype='bool')!=filt_15, dtype='int')

    #%%% SIA/SIE
    dateyear=np.zeros(len(SICs_date), dtype=int)
    datemonth=np.zeros(len(SICs_date), dtype=int)
    dateweek=np.zeros(len(SICs_date), dtype=int)
    dateweekday=np.zeros(len(SICs_date), dtype=int)
    dateordinal=np.zeros(len(SICs_date), dtype=int)
    datedom=np.zeros(len(SICs_date), dtype=int)
    for i in range(len(SICs_date)):
        dateyear[i]=SICs_date[i].year
        datemonth[i]=SICs_date[i].year*100+SICs_date[i].month
        dateweek[i]=SICs_date[i].year*100+int(SICs_date[i].strftime("%W") )
        dateweekday[i]=SICs_date[i].weekday()
        dateordinal[i]=SICs_date[i].toordinal()
        datedom[i]=SICs_date[i].day

    if regional:
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
    timeaxweek=[]
    timeaxmonth=[]
    #for year in
    for week in np.unique(dateweek):
        #timeaxweek.append(SICs_date[dateweek==week][0]+timedelta(3))
        if 3 in dateweekday[dateweek==week]:
            timeaxweek.append(SICs_date[dateweek==week][dateweekday[dateweek==week]==3][0])
    for month in np.unique(datemonth):
        timeaxmonth.append(SICs_date[datemonth==month][0]+timedelta(15))



    SIA_tmp=np.sum(np.sum(SICs, axis=1), axis=1)*dx*dx/100.
    SIE_tmp=np.sum(np.sum(SICs>15., axis=1), axis=1)*dx*dx
    SIAs_day.append(SIA_tmp)
    SIEs_day.append(SIE_tmp)
    SIAs_week_tmp=np.ma.array(np.zeros(len(np.unique(dateweek))), mask=np.ones(len(np.unique(dateweek))))
    SIEs_week_tmp=np.ma.array(np.zeros(len(np.unique(dateweek))), mask=np.ones(len(np.unique(dateweek))))
    for iweek in range(len(np.unique(dateweek))):
        SIAs_week_tmp.mask[iweek]=False
        SIEs_week_tmp.mask[iweek]=False
        SIAs_week_tmp[iweek]=np.mean(SIA_tmp[dateweek==np.unique(dateweek)[iweek]])
        SIEs_week_tmp[iweek]=np.mean(SIE_tmp[dateweek==np.unique(dateweek)[iweek]])
        #dateweek
    SIAs_week.append(SIAs_week_tmp[SIAs_week_tmp.mask==0])
    SIEs_week.append(SIEs_week_tmp[SIAs_week_tmp.mask==0])
    SIA_month_tmp=np.ma.array(np.zeros(len(np.unique(datemonth))), mask=np.ones(len(np.unique(datemonth))))
    SIE_month_tmp=np.ma.array(np.zeros(len(np.unique(datemonth))), mask=np.ones(len(np.unique(datemonth))))
    for imonth in range(len(np.unique(datemonth))):
        SIA_month_tmp.mask[imonth]=False
        SIE_month_tmp.mask[imonth]=False
        SIA_month_tmp[imonth]=np.mean(SIA_tmp[datemonth==np.unique(datemonth)[imonth]])
        SIE_month_tmp[imonth]=np.mean(SIE_tmp[datemonth==np.unique(datemonth)[imonth]])
    SIAs_month.append(SIA_month_tmp[SIA_month_tmp.mask==0])
    SIEs_month.append(SIE_month_tmp[SIE_month_tmp.mask==0])

    if data=='noise' and calc_mean_noise:
        meanNoise.append(np.mean(np.mean(SIC_noises, axis=1),axis=1))
        if comp_cci_ens:
            meanNoise15.append(np.asarray([np.mean(SIC_noises[x][np.logical_and(SIC_cci[x]>10, SIC_cci[x]<20)]) for x in range(len(SIC_noises))]))
            meanNoise15_filt.append(np.asarray([np.mean(SIC_noises[x][np.logical_and(SIC_filt[x]>10, SIC_filt[x]<20)]) for x in range(len(SIC_noises))]))
        #meanNoise15.append(np.mean(np.mean(SIC_noises[np.logical_and(SIC>10, SIC<20)], axis=1),axis=1))

    if regional:
        for sea_id in sea_ids:
            SIA_tmp=np.sum(SICs[:,polyid==sea_id], axis=1)*dx*dx/100.
            SIE_tmp=np.sum(SICs[:,polyid==sea_id]>15., axis=1)*dx*dx

            SIAs_week_tmp=np.ma.array(np.zeros(len(np.unique(dateweek))), mask=np.ones(len(np.unique(dateweek))))
            SIEs_week_tmp=np.ma.array(np.zeros(len(np.unique(dateweek))), mask=np.ones(len(np.unique(dateweek))))
            for iweek in range(len(np.unique(dateweek))):
                SIAs_week_tmp.mask[iweek]=False
                SIEs_week_tmp.mask[iweek]=False
                SIAs_week_tmp[iweek]=np.mean(SIA_tmp[dateweek==np.unique(dateweek)[iweek]])
                SIEs_week_tmp[iweek]=np.mean(SIE_tmp[dateweek==np.unique(dateweek)[iweek]])
                #dateweek
            SIA_month_tmp=np.ma.array(np.zeros(len(np.unique(datemonth))), mask=np.ones(len(np.unique(datemonth))))
            SIE_month_tmp=np.ma.array(np.zeros(len(np.unique(datemonth))), mask=np.ones(len(np.unique(datemonth))))
            for imonth in range(len(np.unique(datemonth))):
                SIA_month_tmp.mask[imonth]=False
                SIE_month_tmp.mask[imonth]=False
                SIA_month_tmp[imonth]=np.mean(SIA_tmp[datemonth==np.unique(datemonth)[imonth]])
                SIE_month_tmp[imonth]=np.mean(SIE_tmp[datemonth==np.unique(datemonth)[imonth]])

            SIAsreg_day[str(sea_id)].append(SIA_tmp)
            SIEsreg_day[str(sea_id)].append(SIE_tmp)
            SIAsreg_week[str(sea_id)].append(SIAs_week_tmp[SIAs_week_tmp.mask==0])
            SIEsreg_week[str(sea_id)].append(SIEs_week_tmp[SIAs_week_tmp.mask==0])
            SIAsreg_month[str(sea_id)].append(SIA_month_tmp[SIA_month_tmp.mask==0])
            SIEsreg_month[str(sea_id)].append(SIE_month_tmp[SIE_month_tmp.mask==0])
            if data=='noise' and calc_mean_noise:
                meanNoisereg[str(sea_id)].append(np.mean(np.mean(SIC_noises, axis=1),axis=1))
                if comp_cci_ens:
                    meanNoise15reg[str(sea_id)].append(np.asarray([np.mean(SIC_noises[x][np.logical_and(SIC_cci[x]>10, SIC_cci[x]<20)]) for x in range(len(SIC_noises))]))
                    meanNoise15reg_filt[str(sea_id)].append(np.asarray([np.mean(SIC_noises[x][np.logical_and(SIC_filt[x]>10, SIC_filt[x]<20)]) for x in range(len(SIC_noises))]))
                #meanNoise15.append(np.mean(np.mean(SIC_noises[np.logical_and(SIC>10, SIC<20)], axis=1),axis=1))


SIAs_day=np.asarray(SIAs_day)
SIEs_day=np.asarray(SIEs_day)
SIAs_week=np.ma.asarray(SIAs_week)
SIEs_week=np.ma.asarray(SIEs_week)
SIAs_month=np.asarray(SIAs_month)
SIEs_month=np.asarray(SIEs_month)
if calc_mean_noise:
    meanNoise=np.asarray(meanNoise)
    meanNoise15=np.asarray(meanNoise15)
    meanNoise15_filt=np.asarray(meanNoise15_filt)

if regional:
    for sea_id in sea_ids:
        SIAsreg_day[str(sea_id)]=np.asarray(SIAsreg_day[str(sea_id)])
        SIEsreg_day[str(sea_id)]=np.asarray(SIEsreg_day[str(sea_id)])
        SIAsreg_week[str(sea_id)]=np.asarray(SIAsreg_week[str(sea_id)])
        SIEsreg_week[str(sea_id)]=np.asarray(SIEsreg_week[str(sea_id)])
        SIAsreg_month[str(sea_id)]=np.asarray(SIAsreg_month[str(sea_id)])
        SIEsreg_month[str(sea_id)]=np.asarray(SIEsreg_month[str(sea_id)])

        meanNoisereg[str(sea_id)]=np.asarray(meanNoisereg[str(sea_id)])
        meanNoise15reg[str(sea_id)]=np.asarray(meanNoise15reg[str(sea_id)])
        meanNoise15reg_filt[str(sea_id)]=np.asarray(meanNoise15reg_filt[str(sea_id)])
        #meanNoise15


monthord=np.array([timeaxmonth[i].toordinal() for i in range(len(timeaxmonth))])
dayord=np.array([SICs_date[i].toordinal() for i in range(len(SICs_date))])

if 1:#figure out why mean(SIE_ensamble)>SIE(mean_ensamble)
    #    if 1:#run this first with original data (1) and then with noise data (0)
    if data=='noise' and comp_cci_ens:
        if 0:
            plt.subplot()
            plt.scatter(SIC_cci[noise_15diff!=0], SIC_sigts[noise_15diff!=0], c=noise_15diff[noise_15diff!=0], cmap='viridis')
            plt.colorbar()

            mask_sig0_diff10=np.logical_and(SIC_sigts<1, noise_15diff==np.max(noise_15diff))
            SIC_tmp=SIC_cci.copy()
            SIC_tmp.mask=mask_sig0_diff10==0
        if 1 and calc_mean_noise:
            plt.figure()
            #plt.scatter(SIC_cci[noise_15diff!=0], noise_15diff[noise_15diff!=0])
            plt.hist2d(SIC_cci[noise_15diff!=0], noise_15diff[noise_15diff!=0], bins=[100, 100], cmap='gist_stern')
            plt.colorbar()
            plt.title('Frequency of 15% missmatch with original CCI')
            plt.figure()
            plt.hist2d(SIC_filt[NoiseFilt_15diff!=0], NoiseFilt_15diff[NoiseFilt_15diff!=0], bins=[100, np.max(NoiseFilt_15diff[NoiseFilt_15diff!=0])], cmap='gist_stern')
            plt.colorbar()
            plt.title('Frequency of 15% missmatch with filtered CCI')
            plt.figure()
            res=plt.hist2d(SIC_filt[NoiseFilt_15diff!=0], SIC_sigts[NoiseFilt_15diff!=0], bins=[100, int(np.max(SIC_sigts[NoiseFilt_15diff!=0]))])
            res_w=plt.hist2d(SIC_filt[NoiseFilt_15diff!=0], SIC_sigts[NoiseFilt_15diff!=0], weights=NoiseFilt_15diff[NoiseFilt_15diff!=0], bins=[100, int(np.max(SIC_sigts[NoiseFilt_15diff!=0]))], cmap='gist_stern')
            plt.title('15% missmatch by filtered CCI SIC and sigma(SIC) [count]')
            plt.figure()
            plt.pcolormesh(res_w[1], res_w[2], res_w[0].T/res[0].T, cmap='gist_stern')
            plt.colorbar()
            plt.title('Relative 15% missmatch by filtered CCI SIC and sigma(SIC) [%]')
            plt.figure()
            plt.hist(SIC_filt[NoiseFilt_15diff!=0], cumulative=True, density=True, weights=NoiseFilt_15diff[NoiseFilt_15diff!=0], bins=100)


            fig, axes = plt.subplots(2,2)
            mapa=axes[0,0].pcolormesh(SIC_cci[337,::-1,:],vmin=0, vmax=100)
            plt.colorbar(mapa,ax=axes[0,0])
            mapa=axes[0,1].pcolormesh(SICs[337,::-1,:],vmin=0, vmax=100)
            plt.colorbar(mapa,ax=axes[0,1])
            mapa=axes[1,0].pcolormesh(noise_15diff[337,::-1,:], vmin=30, vmax=70, cmap='PRGn')
            plt.colorbar(mapa,ax=axes[1,0])
            mapa=axes[1,1].pcolormesh(NoiseFilt_15diff[337,::-1,:], vmin=30, vmax=70, cmap='PRGn')
            plt.colorbar(mapa,ax=axes[1,1])
            #I think it might come from noise filtering at strong edges. check filtered files.
            #file /media/dusch/AndreasExt/SIC/SIC/noise/SIC_50km_gauss_CCI_lowpass5_d_T201501_201512_v2.nc
            plt.figure()
            plt.plot(meanNoise.T)
            plt.figure()
            plt.plot(meanNoise15.T)
            plt.title('SIC 10% - 20%, original')
            plt.figure()
            plt.plot(meanNoise15_filt.T)
            plt.title('SIC 10% - 20%, filtered')
            plt.figure()
            plt.pcolormesh(SIC_noises[260,::-1,:],vmin=-30, vmax=30)
            tmp=np.ma.zeros_like(SIC_noises[260,:,:])
            tmp.mask=mask=np.logical_or(SIC_filt[260,::-1,:]<10, SIC_filt[260,::-1,:]>20)
            plt.pcolor(tmp,hatch='x', alpha=0)


if 0:
        timeaxweek_con=np.ma.zeros(len(SICs_date))
        #timeaxweek_con
if 1:
        ticks=np.hstack([np.array([datetime(2015, x,1) for x in range(1,13,2)]),datetime(2016, 1,1)])
        fig, axes=plt.subplots(nrows=1, ncols=2)#, sharex=True, sharey=True,)
        fig.set_figheight(3)
        fig.set_figwidth(15)
        axes[0].plot(SICs_date, SIAs_day[:20,:].T/1e6, c='grey', alpha=0.6)
        axes[0].plot(SICs_date, np.mean(SIAs_day[:20,:], axis=0)/1e6, c='k')
        axes[0].plot(SICs_date, SIAs_day[0,:]/1e6, c='r')
        #axes[0].set_title('Daily')
        axes[0].set_ylabel(r'SIA [m km$^2$]')
        axes[0].set_xlim([SICs_date[0], SICs_date[-1]])
        axes[0].set_xticks(ticks)
        axes[0].set_xticklabels(['{:04d}-{:02d}'.format(x.year, x.month) for x in ticks])

        axes[1].plot(SICs_date, SIEs_day[:20,:].T/1e6, c='grey', alpha=0.6)
        axes[1].plot(SICs_date, np.mean(SIEs_day[:20,:], axis=0)/1e6, c='k')
        axes[1].plot(SICs_date, SIEs_day[0,:].T/1e6, c='r')
        axes[1].set_ylabel(r'SIE [m km$^2$]')
        axes[1].set_xlim([SICs_date[0], SICs_date[-1]])
        #axes[1].set_xlabel('Time')
        axes[1].set_xticks(ticks)
        axes[1].set_xticklabels(['{:04d}-{:02d}'.format(x.year, x.month) for x in ticks])
        #axes[1].set_title('SIE - Daily')
        #plt.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIASIE_unc/paperoutline/figures/'+'STD_2015_288km_5d.png', dpi=300)

        #ticks=np.hstack([np.array([datetime(2015, x) for x in range(1,13,2)]),datetime(2016, 1)])
        SIA_day_mean=np.mean(SIAs_day, axis=0)/1e6
        fig, axes=plt.subplots(nrows=1, ncols=2)#,(2, sharex=True, sharey=True)
        fig.set_figheight(3)
        fig.set_figwidth(15)
        axes[0].plot(SICs_date, (SIAs_day[:20,:]/1e6-SIA_day_mean).T, c='grey', alpha=0.6)
        #axes[0].plot(SICs_date, SIA_day_mean, c='k')
        axes[0].plot(SICs_date, SIAs_day[0,:]/1e6-SIA_day_mean, c='r')
        #axes[0].set_title('Daily')
        axes[0].set_ylabel(r'SIA Anomaly [m km$^2$]')
        axes[0].set_xlim([SICs_date[0], SICs_date[-1]])
        axes[0].set_xticks(ticks)
        axes[0].set_xticklabels(['{:04d}-{:02d}'.format(x.year, x.month) for x in ticks])

        SIE_day_mean = np.mean(SIEs_day, axis=0)/1e6
        axes[1].plot(SICs_date, (SIEs_day[:20,:]/1e6-SIE_day_mean).T, c='grey', alpha=0.6)
        #axes[1].plot(SICs_date, np.mean(SIEs_day[:20,:], axis=0)/1e6, c='k')
        axes[1].plot(SICs_date, SIEs_day[0,:].T/1e6-SIE_day_mean, c='r')
        axes[1].set_ylabel(r'SIE Anomaly [m km$^2$]')
        axes[1].set_xlim([SICs_date[0], SICs_date[-1]])
        #axes[1].set_xlabel('Time')
        axes[1].set_xticks(ticks)
        axes[1].set_xticklabels(['{:04d}-{:02d}'.format(x.year, x.month) for x in ticks])
        #axes[1].set_title('SIE - Daily')
        #plt.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIASIE_unc/paperoutline/figures/'+'Anom_2015_288km_5d.png', dpi=300)


        if full_year:
            fig, axes=plt.subplots(2)
            axes[0].plot(timeaxweek, SIAs_week.T/1e6, c='grey', alpha=0.6)
            axes[0].plot(timeaxweek, np.mean(SIAs_week, axis=0)/1e6, c='k')
            axes[0].set_title('Weekly')
            axes[0].set_ylabel(r'SIA [m km$^2$]')

            axes[1].plot(timeaxweek, SIEs_week.T/1e6, c='grey', alpha=0.6)
            axes[1].plot(timeaxweek, np.mean(SIEs_week, axis=0)/1e6, c='k')
            axes[1].set_ylabel(r'SIE [m km$^2$]')
            axes[1].set_xlabel('Time')
            #axes[1].set_title('SIE - Daily')


            fig, axes=plt.subplots(2)
            axes[0].plot(timeaxmonth, SIAs_month.T/1e6, c='grey', alpha=0.6)
            axes[0].plot(timeaxmonth, np.mean(SIAs_month, axis=0)/1e6, c='k')
            axes[0].set_title('Monthly')
            axes[0].set_ylabel(r'SIA [m km$^2$]')

            #axes[1].plot(SICs_date[15:][::30], SIEs_month.T/1e6, c='grey', alpha=0.6)
            #axes[1].plot(SICs_date[15:][::30], np.mean(SIEs_month, axis=0)/1e6, c='k')
            axes[1].plot(timeaxmonth, SIEs_month.T/1e6, c='grey', alpha=0.6)
            axes[1].plot(timeaxmonth, np.mean(SIEs_month, axis=0)/1e6, c='k')
            axes[1].set_ylabel(r'SIE [m km$^2$]')
            axes[1].set_xlabel('Time')
            #axes[1].set_title('SIE - Daily')

            fig, axes=plt.subplots(2, sharex=True, sharey=True)
            axes[0].fill_between(SICs_date, np.mean(SIAs_day, axis=0)/1e6-np.std(SIAs_day, axis=0)/1e6, np.mean(SIAs_day, axis=0)/1e6+np.std(SIAs_day, axis=0)/1e6, color='gray',  alpha=1., label='daily')
            axes[0].fill_between(timeaxweek, np.mean(SIAs_week, axis=0)/1e6-np.std(SIAs_week, axis=0)/1e6, np.mean(SIAs_week, axis=0)/1e6+np.std(SIAs_week, axis=0)/1e6, color='purple',  alpha=0.4, label='weekly')
            axes[0].fill_between(timeaxmonth, np.mean(SIAs_month, axis=0)/1e6-np.std(SIAs_month, axis=0)/1e6, np.mean(SIAs_month, axis=0)/1e6+np.std(SIAs_month, axis=0)/1e6, color='red',  alpha=0.4, label='weekly')
            axes[0].plot(SICs_date, np.mean(SIAs_day, axis=0)/1e6, c='k')
            axes[0].plot(timeaxweek, np.mean(SIAs_week, axis=0)/1e6, c='purple')
            axes[0].plot(timeaxmonth, np.mean(SIAs_month, axis=0)/1e6, c='r')
            #axes[0].set_title('Daily')
            axes[0].set_ylabel(r'SIA [m km$^2$]')

            axes[1].fill_between(SICs_date, np.mean(SIEs_day, axis=0)/1e6-np.std(SIEs_day, axis=0)/1e6, np.mean(SIEs_day, axis=0)/1e6+np.std(SIEs_day, axis=0)/1e6, color='gray',  alpha=1., label='daily')
            axes[1].fill_between(timeaxweek, np.mean(SIEs_week, axis=0)/1e6-np.std(SIEs_week, axis=0)/1e6, np.mean(SIEs_week, axis=0)/1e6+np.std(SIEs_week, axis=0)/1e6, color='purple',  alpha=0.4, label='weekly')
            axes[1].fill_between(timeaxmonth, np.mean(SIEs_month, axis=0)/1e6-np.std(SIEs_month, axis=0)/1e6, np.mean(SIEs_month, axis=0)/1e6+np.std(SIEs_month, axis=0)/1e6, color='red',  alpha=0.4, label='weekly')
            axes[1].plot(SICs_date, np.mean(SIEs_day, axis=0)/1e6, c='k', label='Mean Ens')
            axes[1].plot(timeaxweek, np.mean(SIEs_week, axis=0)/1e6, c='purple')
            axes[1].plot(timeaxmonth, np.mean(SIEs_month, axis=0)/1e6, c='r')
            if data=='noise' and comp_cci_ens:
                axes[1].plot(SICs_date, SIE_cci/1e6, c='green', label='CCI')
                axes[1].plot(SICs_date, SIE_filt/1e6, c='orange', label='Smoothed')
            axes[1].set_ylabel(r'SIE [m km$^2$]')
            axes[1].set_xlabel('Time')
            #axes[1].set_title('SIE - Daily')

            if data=='noise' and comp_cci_ens:
                pot_up=np.sum(np.sum(np.logical_and(SIC_filt<=15., 15.-SIC_filt<=SIC_sigts), axis=1), axis=1)
                pot_down=np.sum(np.sum(np.logical_and(SIC_filt>=15., SIC_filt-15.<=SIC_sigts), axis=1), axis=1)
                plt.figure()
                plt.plot(SICs_date,pot_up*dx*dx/1e6, label='Potentual noise SIE increase')
                plt.plot(SICs_date,pot_down*dx*dx/1e6, label='Potentual noise SIE decrease')
                plt.plot(SICs_date,(pot_up-pot_down)*dx*dx/1e6, label='Net potentual noise SIE delta')
                plt.plot(SICs_date, np.mean(SIEs_day, axis=0)/1e6 - SIE_filt/1e6, c='k', label='Mean Ens - smoothed SIE')
                plt.plot([SICs_date[0],SICs_date[-1]], [0,0], ls=':', c='k')
                plt.legend()

            fig, axes=plt.subplots(nrows=1, ncols=2)#2, sharex=True, sharey=True)
            fig.set_figheight(3)
            fig.set_figwidth(15)
            axes[0].plot(SICs_date, np.std(SIAs_day, axis=0)/1e6, color='gray',  alpha=1., label='daily')
            axes[0].plot(timeaxweek,np.std(SIAs_week, axis=0)/1e6, color='purple',  alpha=1, label='weekly')
            axes[0].plot(timeaxmonth, np.std(SIAs_month, axis=0)/1e6, color='red',  alpha=1, label='monthly')
            #axes[0].set_title('Daily')
            axes[0].set_xlim([SICs_date[0],SICs_date[-1]])
            axes[0].set_ylabel(r'STD SIA [m km$^2$]')
            axes[0].set_xlim([SICs_date[0],SICs_date[-1]])
            axes[0].set_xticks(ticks)
            axes[0].set_xticklabels(['{:04d}-{:02d}'.format(x.year, x.month) for x in ticks])


            axes[1].plot(SICs_date, np.std(SIEs_day, axis=0)/1e6, color='gray',  alpha=1., label='daily')
            axes[1].plot(timeaxweek, np.std(SIEs_week, axis=0)/1e6, color='purple',  alpha=1., label='weekly')
            axes[1].plot(timeaxmonth, np.std(SIEs_month, axis=0)/1e6, color='red',  alpha=1., label='monthly')

            axes[1].set_xlim([SICs_date[0],SICs_date[-1]])
            #plt.xticks(rotation=22.5)
            axes[1].set_xticks(ticks)
            axes[1].set_xticklabels(['{:04d}-{:02d}'.format(x.year, x.month) for x in ticks])
            axes[1].set_ylabel(r'STD SIE [m km$^2$]')
            #axes[1].set_xlabel('Time')
            axes[0].legend(loc='upper left')

            #plt.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIASIE_unc/paperoutline/figures/'+'STD_X3_2015_288km_5d.png', dpi=300)
            #axes[1].set_title('SIE - Daily')

        else: #not full year = trends
            fig, axes=plt.subplots(2, sharex=True, sharey=True)
            for i in range(20):
                axes[0].scatter(SICs_date, SIAs_day[i,:]/1e6, c='grey', alpha=0.4)
                res=stats.linregress(dayord, SIAs_day[i,:])
                axes[0].plot(SICs_date, (res.intercept+res.slope*dayord)/1e6, 'grey', alpha=0.4)
            axes[0].scatter(SICs_date, np.mean(SIAs_day[:,:], axis=0)/1e6, c='k')
            #axes[0].scatter(SICs_date, SIAs_day[0,:]/1e6, c='r')
            #res=stats.linregress(dayord, SIAs_day[0,:])
            #axes[0].plot(SICs_date, (res.intercept+res.slope*dayord)/1e6, 'r')
            #axes[0].set_title('Daily')
            axes[0].set_ylabel(r'SIA [m km$^2$]')
            #axes[0].set_xlim([SICs_date[0], SICs_date[-1]])
            #axes[0].set_xticks(ticks)

            for i in range(20):
                axes[1].scatter(SICs_date, SIEs_day[i,:]/1e6, c='grey', alpha=0.4)
                res=stats.linregress(dayord, SIEs_day[i,:])
                axes[1].plot(SICs_date, (res.intercept+res.slope*dayord)/1e6, 'grey', alpha=0.4)
            axes[1].scatter(SICs_date, np.mean(SIEs_day[:,:], axis=0)/1e6, c='k')
            #axes[1].scatter(SICs_date, SIEs_day[0,:]/1e6, c='r')
            #res=stats.linregress(dayord, SIEs_day[0,:])
            #axes[1].plot(SICs_date, (res.intercept+res.slope*dayord)/1e6, 'r')
            #axes[1].set_title('Daily')
            axes[1].set_ylabel(r'SIE [m km$^2$]')
            #axes[1].set_xlim([SICs_date[0], SICs_date[-1]])
            #axes[1].set_xticks(ticks)
            #plt.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIASIE_unc/paperoutline/figures/'+'trendTS_288km_5d_DAILY.png', dpi=300)


            fig, ax=plt.subplots(1, sharex=True, sharey=True)
            axes=[ax]
            for i in range(20):#n_noise
                if i==0:
                    axes[0].scatter(timeaxmonth, SIAs_month[i,:]/1e6, c='grey', alpha=0.4, label='Ensemble')
                else:
                    axes[0].scatter(timeaxmonth, SIAs_month[i,:]/1e6, c='grey', alpha=0.4)
                res=stats.linregress(monthord, SIAs_month[i,:])
                axes[0].plot(timeaxmonth, (res.intercept+res.slope*monthord)/1e6, 'grey', alpha=0.4)
            #axes[0].scatter(timeaxmonth, SIAs_month[1,:]/1e6, c='r')
            res=stats.linregress(monthord, SIAs_month[1,:])
            #axes[0].plot(timeaxmonth, (res.intercept+res.slope*monthord)/1e6, 'r')
            axes[0].scatter(timeaxmonth, np.mean(SIAs_month[:,:], axis=0)/1e6, c='k', marker='*', label='Ensemble Mean')
            res=stats.linregress(monthord, np.mean(SIAs_month[:,:], axis=0))
            axes[0].plot(timeaxmonth, (res.intercept+res.slope*monthord)/1e6, 'k')
            #axes[0].set_title('Daily')
            axes[0].set_ylabel(r'September SIA [m km$^2$]')
            #axes[0].set_xlim([timeaxmonth[0], timeaxmonth[-1]])
            #axes[0].set_xticks(ticks)
            #plt.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/first_draft/figures/'+'trend NH09_scat.png', dpi=300)

            fig, ax=plt.subplots(1)
            axes=[ax]
            x_axis_tmp=np.array([timeaxmonth[i].year for i in range(len(timeaxmonth))])
            for i in range(20):#n_noise
                res=stats.linregress(monthord, SIAs_month[i,:])
                axes[0].plot(x_axis_tmp, (res.intercept+res.slope*monthord)/1e6, 'grey', alpha=0.4)
            axes[0].boxplot(SIAs_month[:,:]/1e6, positions=x_axis_tmp)#

            #axes[0].scatter(timeaxmonth, SIAs_month[1,:]/1e6, c='r')
            res=stats.linregress(monthord, SIAs_month[1,:])
            #axes[0].plot(timeaxmonth, (res.intercept+res.slope*monthord)/1e6, 'r')
            #axes[0].scatter(timeaxmonth, np.mean(SIAs_month[:,:], axis=0)/1e6, c='k', marker='*', label='Ensemble Mean')
            res=stats.linregress(monthord, np.mean(SIAs_month[:,:], axis=0))
            axes[0].plot(x_axis_tmp, (res.intercept+res.slope*monthord)/1e6, 'k')
            #axes[0].set_title('Daily')
            axes[0].set_ylabel(r'September SIA [m km$^2$]')
            #axes[0].set_xlim([timeaxmonth[0], timeaxmonth[-1]])
            axes[0].set_xticks(np.arange(1980, 2025, 5))
            axes[0].set_xticklabels(np.arange(1980, 2025, 5))
            #plt.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIA_data/first_draft/figures/'+'trend_NH09_box.png', dpi=300)

            if 0:
                for i in range(20):
                    if i==0:
                        axes[1].scatter(timeaxmonth, SIEs_month[i,:]/1e6, c='grey', alpha=0.4, label='Ensemble')
                    else:
                        axes[1].scatter(timeaxmonth, SIEs_month[i,:]/1e6, c='grey', alpha=0.4)
                    res=stats.linregress(monthord, SIEs_month[i,:])
                    axes[1].plot(timeaxmonth, (res.intercept+res.slope*monthord)/1e6, 'grey', alpha=0.4)
                axes[1].scatter(timeaxmonth, SIEs_month[1,:]/1e6, c='r', label='Selected member')
                res=stats.linregress(monthord, SIEs_month[1,:])
                axes[1].plot(timeaxmonth, (res.intercept+res.slope*monthord)/1e6, 'r')
                axes[1].scatter(timeaxmonth, np.mean(SIEs_month[:,:], axis=0)/1e6, c='k', marker='*', label='Ensemble Mean')
                res=stats.linregress(monthord, np.mean(SIEs_month[:,:], axis=0))
                axes[1].plot(timeaxmonth, (res.intercept+res.slope*monthord)/1e6, 'k')
                #axes[0].set_title('Daily')
                axes[1].set_ylabel(r'SIE [m km$^2$]')
                #axes[0].set_xlim([timeaxmonth[0], timeaxmonth[-1]])
                #axes[0].set_xticks(ticks)
            axes[0].legend()
            #plt.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIASIE_unc/paperoutline/figures/'+'trendTS_288km_5d.png', dpi=300)


        print('SIA Uncertainty [m km²] {:.3f} (daily) {:.3f} (weekly); {:.3f} (monthly)'.format(np.mean(np.std(SIAs_day, axis=0)/1e6), np.mean(np.std(SIAs_week, axis=0)/1e6), np.mean(np.std(SIAs_month, axis=0)/1e6)))
        print('SIE Uncertainty [m km²] {:.3f} (daily) {:.3f} (weekly); {:.3f} (monthly)'.format(np.mean(np.std(SIEs_day, axis=0)/1e6), np.mean(np.std(SIEs_week, axis=0)/1e6), np.mean(np.std(SIEs_month, axis=0)/1e6)))

        print('SIA Uncertainty [m km²] before 1988 {:.3f} (daily), {:.3f} (monthly)'.format(np.mean(np.std(SIAs_day, axis=0)[dateyear<1988]/1e6), np.mean(np.std(SIAs_month, axis=0)[:9]/1e6)))
        print('SIE Uncertainty [m km²] before 1988 {:.3f} (daily), {:.3f} (monthly)'.format(np.mean(np.std(SIEs_day, axis=0)[dateyear<1988]/1e6), np.mean(np.std(SIEs_month, axis=0)[:9]/1e6)))

if 1:
        slopemon=[]
        for i in range(len(paths)):
            res=stats.linregress(monthord, SIAs_month[i,:])
            #res=stats.linregress(dayord, SIAs_day[i,:])
            slopemon.append(res[0])

        #res_mean=stats.linregress(dayord, np.mean(SIAs_day, axis=0))
        slopemon=np.asarray(slopemon)
        save_dir='/home/dusch/data/SIC/noise/npsaves/'
        #np.save(save_dir+'slopes_{}.npy'.format(month), slopemon)
        slopeCCI=-285.27
        SEslopeCCI=16.38#both in km2/day
        if len(slopemon)==1:
            print('SIA trend in 10$^3$ km$^2$/a: {}'.format(slopemon*365./1e3))
            print('Standard error SIA trend in 10$^3$ km$^2$/a: {}'.format(res[4]*365./1e3))
        elif len(slopemon)>1:
            plt.figure()
            plt.hist(slopemon*365./1e3, density=True, histtype='stepfilled', label='Uncertainty Ensemble')
            #plt.plot([slopeCCI*365./1e3, slopeCCI*365./1e3], [0,1], color='k', label='CCI trend')
            #plt.fill_betweenx([0,1], [(slopeCCI-SEslopeCCI)*365./1e3, (slopeCCI-SEslopeCCI)*365./1e3], [(slopeCCI+SEslopeCCI)*365./1e3, (slopeCCI+SEslopeCCI)*365./1e3], \
            #                  color='grey', alpha=0.4, zorder=100)
            #plt.plot([(slopeCCI-SEslopeCCI)*365./1e3, (slopeCCI-SEslopeCCI)*365./1e3], [0,1], color='k', ls='--')
            #plt.plot([(slopeCCI+SEslopeCCI)*365./1e3, (slopeCCI+SEslopeCCI)*365./1e3], [0,1], color='k', ls='--')

            plt.xlabel(r'Trend in September SIA {}-{} [10$^3$ km$^2$/a]'.format(years[0], years[-1]))
            plt.ylabel(r'Frequency density')
            #plt.ylim([0,0.05])
            plt.legend(loc='upper left')
            print('SIA trend mean ensemble in 10$^3$ km$^2$/a: {}'.format(np.mean(slopemon)*365./1e3))
            print('SIA trend std ensemble in 10$^3$ km$^2$/a: {}'.format(np.std(slopemon)*365./1e3))
        #plt.savefig('/home/dusch/Dropbox/Documents/manuscripts/SIASIE_unc/paperoutline/figures/'+'hist_sep_slope.png', dpi=300)

        if 0:
            year_tmp=2012
            dom_min=[]
            for i in range(len(paths)):
                dom_min.append(datedom[dateyear==year_tmp][SIAs_day[i,dateyear==year_tmp]==np.min(SIAs_day[i,dateyear==year_tmp])])
            dom_min=np.asarray(dom_min).flatten()
            #datedom
            #SIA_day


        if 1:
            def f(B, x):
                '''Linear function y = m*x + b'''
                # B is a vector of the parameters.
                # x is an array of the current x values.
                # x is in the same format as the x passed to Data or RealData.
                #
                # Return an array in the same format as y passed to Data or RealData.
                return B[0]*x + B[1]
            linear = odr.Model(f)
            sx=np.zeros_like(monthord)+0.001
            sy=np.std(SIAs_month, axis=0)
            #sy=np.zeros_like(monthord)+0.001
            mydata = odr.Data(monthord,  np.mean(SIAs_month, axis=0), wd=1./sx**2, we=1./sy**2)
            #mydata = odr.Data(monthord,  np.mean(SIAs_month, axis=0), wd=1./sx**2)
            myodr = odr.ODR(mydata, linear, beta0=[1., 2.])
            myoutput = myodr.run()
            myoutput.pprint()
            print(myoutput.beta*365./1e3)
            print(myoutput.sd_beta*365./1e3)

if 0:
    plt.figure()
    #plt.plot(np.sum(np.sum(SIC_sigts, axis=1), axis=1)*50.*50./1e6/100)
    #plt.plot(np.sum(np.sum(noise, axis=1), axis=1)*50.*50./1e6/100)
    #ax=plt.gca()
    #ax.set_ylim([0,3])

    if 1:
        fig, axes=plt.subplots(2, sharex=True, sharey=True)
        axes[0].plot(SICs_date, np.std(SIAs_day, axis=0)/1e6, color='gray',  alpha=1., label='daily')
        axes[0].plot(timeaxweek,np.std(SIAs_week, axis=0)/1e6, color='purple',  alpha=1, label='weekly')
        axes[0].plot(timeaxmonth, np.std(SIAs_month, axis=0)/1e6, color='red',  alpha=1, label='monthly')
        #axes[0].set_title('Daily')
        axes[0].set_ylabel(r'SIA [m km$^2$]')
        axes[0].set_xlim([SICs_date[0],SICs_date[-1]])

        axes[1].plot(SICs_date, np.std(SIEs_day, axis=0)/1e6, color='gray',  alpha=1., label='daily')
        axes[1].plot(timeaxweek, np.std(SIEs_week, axis=0)/1e6, color='purple',  alpha=1., label='weekly')
        axes[1].plot(timeaxmonth, np.std(SIEs_month, axis=0)/1e6, color='red',  alpha=1., label='monthly')
        axes[1].set_xlim([SICs_date[0],SICs_date[-1]])
        axes[1].set_xlim([SICs_date[0],SICs_date[-1]])
        plt.xticks(rotation=22.5)
        axes[1].set_ylabel(r'SIE [m km$^2$]')
        axes[1].set_xlabel('Time')

if 0:#combine all SIE in one graph
    if 0:
        areadir='/mnt/icdc/ice_and_snow/uhh_seaiceareatimeseries/DATA/'
        if XH=='NH': areafn='SeaIceArea__NorthernHemisphere__monthly__UHH__v2019_fv0.01.nc'
        else:        areafn='eaIceArea__SouthernHemisphere__monthly__UHH__v2019_fv0.01.nc'
        ds=nc.Dataset(areadir+areafn, "r")
        SIAUHH_time=ds.variables['time'][:]
        SIA_osi=ds.variables['osisaf'][:]
        SIA_hado=ds.variables['HadISST_orig'][:]
        SIA_nt=ds.variables['nsidc_nt'][:]
        SIA_walsh=ds.variables['walsh'][:]
        ds.close()

    start_time=date(1850,1,15)#added 15 days because time always points to the first of the month by default
    if 0:
        time_tmp=np.asarray([start_time+timedelta(SIAUHH_time.data[i]) for i in np.nonzero(np.isnan(SIAUHH_time.data)==0)[0]])
        SIAUHH_time=time_tmp

    extentdir='/media/dusch/AndreasExt/SIC/SIC/SIE/'
    if XH=='NH': extentfn='SIE_observations_nh_v2023_fv0.01.nc'#HadISST SIE from quentin (like icdc SIA)
    else:        extentfn='SIE_observations_sh_v2023_fv0.01.nc'
    ds=nc.Dataset(extentdir+extentfn, "r")
    SIEUHH_time=ds.variables['time'][:]
    SIE_hado=ds.variables['HadISST_orig'][:]
    SIE_hadnsidc=ds.variables['HadISST_nsidc'][:]
    ds.close()
    time_tmp=np.asarray([start_time+timedelta(SIEUHH_time.data[i]) for i in np.nonzero(np.isnan(SIEUHH_time.data)==0)[0]])
    SIEUHH_time=time_tmp

    if XH=='NH': extentfn='N_seaice_extent_daily_v3.0.csv'#NOAA SIE from https://noaadata.apps.nsidc.org/NOAA/G02135
    else:        extentfn='S_seaice_extent_daily_v3.0.csv'

    SIE_NOAA=[]
    time_NOAA=[]
    with open(extentdir+extentfn, mode='r') as csvfile:
        ds = csv.reader(csvfile)
        lc=0
        for row in ds:
            if lc <= 1:
                lc+=1
                print(', '.join(row))
            else:
                SIE_NOAA.append(float(row[3]))
                time_NOAA.append(date(int(row[0]),int(row[1]),int(row[2])))
    SIE_NOAA=np.asarray(SIE_NOAA)
    time_NOAA=np.asarray(time_NOAA)

    fig, axes=plt.subplots(2, sharex=True, sharey=True)
    #axes[0].fill_between(SICs_date, np.mean(SIAs_day, axis=0)/1e6-np.std(SIAs_day, axis=0)/1e6, np.mean(SIAs_day, axis=0)/1e6+np.std(SIAs_day, axis=0)/1e6, color='red',  alpha=1., label='daily')
    #axes[0].fill_between(timeaxweek, np.mean(SIAs_week, axis=0)/1e6-np.std(SIAs_week, axis=0)/1e6, np.mean(SIAs_week, axis=0)/1e6+np.std(SIAs_week, axis=0)/1e6, color='purple',  alpha=0.4, label='weekly')
    axes[0].fill_between(timeaxmonth, np.mean(SIAs_month, axis=0)/1e6-2.*np.std(SIAs_month, axis=0)/1e6, np.mean(SIAs_month, axis=0)/1e6+2.*np.std(SIAs_month, axis=0)/1e6, color='gray',  alpha=0.4, label=r'$\pm$2$\sigma$')
    axes[0].plot(SICs_date, np.mean(SIAs_day, axis=0)/1e6, c='k', label='Ensemble')
    axes[0].plot(SICs_date, SIA_cci/1e6, c='orange', label='CCI')
    axes[0].plot(SICs_date, SIA_filt/1e6, c='green',label='Smoothed')
    #axes[0].plot(timeaxweek, np.mean(SIAs_week, axis=0)/1e6, c='purple')
    axes[0].plot(timeaxmonth, np.mean(SIAs_month, axis=0)/1e6, c='k', ls=':', label='Ensemble')
    if 0:
        axes[0].plot(SIAUHH_time, SIA_osi,  ls=':', label='OSI SAF')
        axes[0].plot(SIAUHH_time, SIA_hado, c='r', ls=':', label='Had_o')
        axes[0].plot(SIAUHH_time, SIA_nt, ls=':', label='NSIDC_nt')
        axes[0].plot(SIAUHH_time, SIA_walsh, ls=':', label='Walsh')
    #axes[0].set_title('Daily')
    axes[0].set_ylabel(r'SIA [m km$^2$]')
    axes[0].set_xlim(SICs_date[0],SICs_date[-1])
    axes[0].legend()
    #axes[1].fill_between(SICs_date, np.mean(SIEs_day, axis=0)/1e6-np.std(SIEs_day, axis=0)/1e6, np.mean(SIEs_day, axis=0)/1e6+np.std(SIEs_day, axis=0)/1e6, color='red',  alpha=1., label='daily')
    #axes[1].fill_between(timeaxweek, np.mean(SIEs_week, axis=0)/1e6-np.std(SIEs_week, axis=0)/1e6, np.mean(SIEs_week, axis=0)/1e6+np.std(SIEs_week, axis=0)/1e6, color='purple',  alpha=0.4, label='weekly')
    axes[1].fill_between(timeaxmonth, np.mean(SIEs_month, axis=0)/1e6-2.*np.std(SIEs_month, axis=0)/1e6, np.mean(SIEs_month, axis=0)/1e6+2.*np.std(SIEs_month, axis=0)/1e6, color='gray',  alpha=0.4, label=r'$\pm$2$\sigma$')
    axes[1].plot(SICs_date, np.mean(SIEs_day, axis=0)/1e6, c='k', label='Ensemble')
    #axes[1].plot(timeaxweek, np.mean(SIEs_week, axis=0)/1e6, c='purple')
    axes[1].plot(timeaxmonth, np.mean(SIEs_month, axis=0)/1e6, ls=':', c='k',label='Ensemble')
    if 0:
        axes[1].plot(SIEUHH_time, SIE_hado, c='r', ls=':', label='Had_o')
        axes[1].plot(SIEUHH_time, SIE_hadnsidc, ls=':', label='Had_nsidc')
    axes[1].plot(time_NOAA, SIE_NOAA, label='NOAA')
    if data=='noise' and comp_cci_ens:
        axes[1].plot(SICs_date, SIE_cci/1e6, c='orange', label='CCI')
        axes[1].plot(SICs_date, SIE_filt/1e6, c='green', label='Smoothed')
    axes[1].set_ylabel(r'SIE [m km$^2$]')
    axes[1].set_xlabel('Time')
    axes[1].legend()





















