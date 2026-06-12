from pydl.goddard.astro import gcirc
import math
import subprocess
from astropy.io import fits
from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import os
import warnings
from numpy import inf
warnings.filterwarnings("ignore")
from PyAstronomy import pyasl
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.sdss import SDSS
from astropy.coordinates import Angle
import heapq
import pandas as pd
#import seaborn as sns
from scipy.interpolate import interp1d
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tck
import matplotlib.patches as mpatches
from astropy.modeling.models import custom_model
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
#from scipy import asarray as ar,exp
from scipy import interpolate
from scipy.interpolate import BSpline, splrep, splev
from scipy.stats import ks_2samp
import matplotlib.ticker as mticker
from scipy.integrate import quad
from matplotlib.offsetbox import AnchoredText
from random import randrange, uniform
import random
from glob import glob
import glob
import sys
#sys.path.insert(1, '/home/sapna/Proj1_stack/');
#import sap_modules
from scipy.ndimage import gaussian_filter1d
import matplotlib.gridspec as gridspec
from scipy import integrate
from statsmodels.stats.proportion import proportion_confint
from spectres import spectres
from linetools.spectra.xspectrum1d import XSpectrum1D
from astropy import units as u


######################

################
class switch(object):
    value = None
    def __new__(class_, value):
        class_.value = value
        return True

def case(*args):
    return any((arg == switch.value for arg in args))
###############



AND = np.logical_and
OR  = np.logical_or
SH  = np.shape
nullfmt   = NullFormatter()
#################### Initial parameters ###########

arcsec2rad = 4.84814e-6
cm_2_Mpc = 3.086e24
M_sun   = 1.989e33
pi = 3.141592653589793
Ho = 71
cosmo = FlatLambdaCDM(H0=Ho, Om0=0.3,Tcmb0=2.725)

##############
def open_calibrate_fits(filename):
    data, header = fits.getdata(filename, ext=1,header=True)
    wave   = data.field('WAVE')
    flux   = data.field('FLUX')
    eflux  = data.field('ERROR')
    model  = data.field('Conti_spline')
    #reg1   = ((wave > 1210) & (wave < 1220))     ###((wave > 1198) & (wave < 1202))  |  | ((wave > 1300) & (wave < 1308))
    return wave, flux, eflux, model


##################
def get_AOD(w_obs,fl_n,ion_wv,ion_os,zabs):
    v = 299792.46*(w_obs/((1.0 + zabs)*ion_wv)  - 1.0)
    Const_fact = (2.654e-15)*ion_os*ion_wv
    tau = np.log(1.0/fl_n)

    id1 = np.where(fl_n >= 1)[0]
    id0 = np.where(fl_n <= 0)[0]
    if np.size(id1) >= 1: tau[id1] = 1e-6
    if np.size(id0) >= 1: tau[id0] = 1e6

    N_app = tau/Const_fact
    ##########
    idv = np.where(np.logical_and(v > -50, v < 50))[0]
    #print('Integrated N between 50 km/s velocity range:', np.log10(integrate.simpson(N_app[idv], v[idv], even='first')))
    ###########
    return v, N_app

####==================================

def plot_expected_lines_ver1(name_q_loc,zcl,v1_abs, v2_abs,pairid_loc,cnf_val, v_cent):
    ion_wav = [1215.6701, 1025.7223, 972.5368, 949.7431, 937.8035, 1031.927, 1037.616,  1036.3367, 1193.2897, 1206.500 ] #,  1190.4158, 1260.4221,1 334.5323, ]
    ion_nm  = ['lya',     'lyb',     'lyg',    'lyd',    'lye',    'OVI',    'OVI'   ,  'CII',     'SiII',    'SiIII'  ] #,  'SiII',    'SiII'   , 'CII',     ]
    ion_os  = [0.4164,     0.07912,  0.029,    0.01394,  0.007799,  0.1329,   0.06609,  0.1231,    0.4991,    1.669    ]
    ##############
    dir_data_file = 'New_Data/'+name_q_loc+'/'
    filename = glob.glob(dir_data_file+'*_conti_spectres.fits')
    [wav,flux,sig,conti_sp,flg_em] = open_calibrate_fits(filename[0])
    id_spec = np.where( flg_em == 0)[0]
    w_cos,f_cos,e_cos,c_cos = wav[id_spec],flux[id_spec],sig[id_spec], conti_sp[id_spec]
    ################        
    fnt_sz = 15
    lb_sz = 15
    lwd=2
    l_major,l_minor = 8,4
    ##############
    fig= plt.figure(figsize=(14, 10))
    fig.subplots_adjust(left=0.05, bottom=0.08, right=0.97, top=0.9, wspace=0.12, hspace=0.0)
    
    gs0 = fig.add_gridspec(1, 2,  width_ratios=[5.0,3.0],height_ratios=[2])
    
    gs00 = gs0[0].subgridspec(10, 1)

    ########### for apparent optical depth #######
    gs01 = gs0[1].subgridspec(3, 1)
    axs01 = fig.add_subplot(gs01[1, 0])
    c_aod = ['k','r','b','g','magenta']
    alp_aod = [0.9,0.6,0.5,0.4,0.3]
    axs01.set_ylabel('log N$_{a}$(v)',fontsize=fnt_sz)
    axs01.set_xlabel('V$_{LOS-abs}$',fontsize=fnt_sz)
    axs01.set_xticks([-0.1,0.0,0.1])
    ###########

    for ii in range(len(ion_wav)): 
        axs00 = fig.add_subplot(gs00[ii, 0])
        vel = 299792.46*(w_cos/((1.0 + zcl)*ion_wav[ii])  - 1.0)
        axs00.step(vel/1000.0,f_cos/c_cos,color='darkblue',lw=1.0,label=ion_nm[ii]+str(int(ion_wav[ii])))
        axs00.legend(frameon=False,loc='lower right',prop={'size': fnt_sz-5})
        axs00.set_xlim(-0.6, 0.6)
        axs00.set_ylim(0.0,1.2)
        #axs00.set_yticks([0.25,0.5,0.75])
        axs00.axhline(1.0,color='k',ls='-.')
        axs00.axvline(0.0,color='k',ls='-.')
        axs00.axvline(v1_abs/1000.0,color='magenta',ls='-.')
        axs00.axvline(v2_abs/1000.0,color='magenta',ls='-.')
        axs00.axvline(v_cent/1000.0,color='red',ls='-.',lw=0.5)
        axs00.tick_params(axis="x", labelsize=lb_sz)
        axs00.tick_params(axis="y", labelsize=lb_sz)
        axs00.tick_params(axis='both',which='major',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=6)
        axs00.tick_params(axis='both',which='minor',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=3)    
        plt.setp(axs00.spines.values(), linewidth=lwd)

        ################
        if ii < 5:
            wabs = (v_cent/299792.46 + 1.0)*((1.0 + zcl)*ion_wav[ii])
            zabs = (wabs/ion_wav[ii] - 1.0)
            v_aod, N_aod = get_AOD(w_cos,f_cos/c_cos,ion_wav[ii],ion_os[ii],zabs)
            v_aod, N_aod = v_aod[np.isnan(N_aod) == False]/1000.0, N_aod[np.isnan(N_aod) == False]

            id_line = np.where( AND(v_aod > -0.55, v_aod < 0.55))[0]
            if np.size(id_line) > 3:
                axs01.step(v_aod[id_line], N_aod[id_line],color=c_aod[ii],lw=1.0,label=ion_nm[ii], alpha= alp_aod[ii])
                lg1 = axs01.legend(frameon=False,prop={'size': fnt_sz-4}, ncol=5,bbox_to_anchor=(1.0,1.2))
                #axs01.set_ylim(np.median(N_aod[id_line])-10, np.median(N_aod[id_line])+10)
                axs01.tick_params(axis="x", labelsize=lb_sz)
                axs01.tick_params(axis="y", labelsize=lb_sz)
                axs01.tick_params(axis='both',which='major',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=6)
                axs01.tick_params(axis='both',which='minor',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=3)
                plt.setp(axs01.spines.values(), linewidth=lwd)
        #############
    str_title = name_q_loc+';  zcl='+str(zcl)+';  pairid = '+str(pairid_loc)+';  CNF = '+str(cnf_val)
    fig.text(0.15,0.93,str_title,fontsize=15)
    fig.text(0.34, 0.03,  'LOS Velocity '+r'( 10$^{3}$ km s$^{-1}$ )', va='center', ha='center',fontsize=fnt_sz+4)
    fig.text(0.02, 0.5, 'Normalized Flux', va='center', ha='center', rotation='vertical',fontsize=fnt_sz+4)
    #plt.show()
    ###########===============================


    
############==========================
def get_R500(z_c,M500_c):  
    M500 = M500_c*10**14 * M_sun                              #### in gm
    ro_c = (cosmo.critical_density(z_c)).value                #### in g/cm^3
    R500_t   =  ( (M500/(500*ro_c))*(3.0/(4.0*pi)) )**(1/3)   #### in cm
    return R500_t/cm_2_Mpc                                    #### in Mpc
############==========================
def get_velocity(z1,z2):
    beta = ( (1+z1)**2 - (1+z2)**2 ) /  ( (1+z1)**2 + (1+z2)**2 )
    velocity = beta * 299792.46
    return  velocity
#############
def delvz(z1,z2): 
    beta = ((1.+z2)**2. - (1.+z1)**2.)/((1.+z2)**2. + (1.+z1)**2.) 
    vout = beta*2.997e5   
    return vout



##########################################################################

def get_composite(infile,w_ion):
        data, header = fits.getdata(infile, ext=1,header=True)
        w            = data.field('wave')[0]
        mean_f       = data.field('mean_flux')[0]
        med_f        = data.field('med_flux')[0]
        N_q_id       = data.field('N_q')[0]
        mean_f_rnd   = data.field('MEAN_FLUX_RND')[0]
        med_f_rnd    = data.field('MED_FLUX_RND')[0]
        mean_bs      = data.field('MEAN_FLUX_BS')[0]
        med_bs       = data.field('MED_FLUX_BS')[0]
        N_q_contri   = data.field('ID_Q_U')[0]
        #std_mean_bs = np.std(data.field('MEAN_FLUX_BS')[0],axis=1)
        #std_med_bs = np.std(data.field('MED_FLUX_BS')[0] ,axis=1)
        vel = 299792.46*(w/w_ion - 1)
        return w,mean_f,med_f,vel,N_q_id.astype('int'),mean_f_rnd,med_f_rnd,mean_bs,med_bs,N_q_contri

#############
def calcew(wl,fl,err,lmts,z=0.0):
    pix = np.where((wl >= lmts[0]) & (wl <= lmts[1]))   
    pixp = tuple(np.array(pix)+1)  
    ew = sum((1.-fl[pix])*(wl[pixp]-wl[pix])) 
    sigew = np.sqrt(sum( (err[pix]*(wl[pixp]-wl[pix]))**2 )) 
    return ew,sigew
###################
def running_median(datx,daty,daty_err,bin_size=21):
    xvals,yvals_w,yvals_unw = [],[],[]
    for j in range(len(datx)):
        if j+bin_size < len(datx):
            xvals.append( np.mean(datx[j:j+bin_size]) )
            yvals_w.append( np.average(daty[j:j+bin_size], weights=1.0/daty_err[j:j+bin_size]**2) )
            yvals_unw.append( np.median(daty[j:j+bin_size]) )
        elif j+bin_size >= len(datx):
            k = j
            bin_size = len(datx) - k
            xvals.append( np.mean(datx[j:j+bin_size]) )
            yvals_w.append( np.average(daty[j:j+bin_size], weights=1.0/daty_err[j:j+bin_size]**2) )
            yvals_unw.append( np.median(daty[j:j+bin_size]) )
            k = k-1
    return np.array(xvals),np.array(yvals_w),np.array(yvals_unw)
##############
def fit_ploy2D(xvar,yvar,yvar_er):
    @custom_model
    def poly_fit2(x, a1=0.0,a2=0.0,a3=1.0):
        return  a1*x**2 + a2*x + a3
    fitter = fitting.LevMarLSQFitter()
    p_init = poly_fit2(a1=0.0,a2=0.0,a3=1.0)  
    p = fitter(p_init,xvar,yvar,weights=1.0/yvar_er**2) 
    return p
###################

def fit_ploy3D(xvar,yvar,yvar_er):
    @custom_model
    def poly_fit2(x, a1=0.0,a2=0.0,a3=0.0,a4=1.0):
        return  a1*x**3 + a2*x**2 + a3*x + a4
    fitter = fitting.LevMarLSQFitter()
    p_init = poly_fit2( a1=0.0,a2=0.0,a3=0.0,a4=1.0)  
    p = fitter(p_init,xvar,yvar,weights=1.0/yvar_er**2) 
    return p
############################
def line1(x, N1,a,b):
    return N1*np.exp(-0.5*( ( x - a )/b )**2) ###N1*np.exp(-0.5*( ( x - (1.0+ a)*2796.3543 )/b )**2)
   

###########################

def fit_ploy1D(xvar,yvar,yvar_er):
    @custom_model
    def poly_fit1(x, a1=0.0,a2=1.0):
        return  a1*x + a2
    fitter = fitting.LevMarLSQFitter()
    p_init = poly_fit1(a1=0.0,a2=1.0)
    p = fitter(p_init,xvar,yvar,weights=1.0/yvar_er**2)
    return p

############################################################

    
######################==================

def resample_spec(wav, flux, eflux, conti, npix):

    #wv_array = np.arange(wav[0], wav[-1], (wav[-1] - wav[0])/int(np.size(wav)/5.0))   ###np.arange(900.000, 1188.014, 0.014) * u.AA  # (Start wv, end wv, delta lambda)
    #sp1 = XSpectrum1D.from_file('/Users/frcashman/Dropbox/P2/SMCX-1/h_d17501_nvo_ufix.fits')

    #sp2 = sp1.rebin(wv_array, do_sig=True, grow_bad_sig=True)
    #sp2.write_to_ascii('/Users/frcashman/Dropbox/P2/SMCX-1/smcx1_bin.txt', overwrite=True)

    
    regrid = np.arange(wav[0], wav[-1], (wav[-1] - wav[0])/int(np.size(wav)/npix))

    n_flux_re, n_eflux_re = flux/conti, eflux/conti


    f_re, er_re  = spectres(regrid, wav, n_flux_re , spec_errs= n_eflux_re, verbose=False)
    
    wav,flux,eflux = regrid, f_re, er_re

    # plt.step(wav,flux)
    # plt.axhline(1.0,color='r')
    # plt.show()

    return wav,flux,eflux


def get_vLSR_corr(ra_deg, dec_deg):

    print(ra_deg, dec_deg)
    sky_coord = SkyCoord(ra = ra_deg * u.degree, dec = dec_deg * u.degree, frame='icrs')
    galactic_coord = sky_coord.galactic

    # galactic_latitude = galactic_coord.b.degree
    # galactic_longitude = galactic_coord.l.degree

    latr   = galactic_coord.b.radian  
    longr  = galactic_coord.l.radian  

    vcorr  = 9.0 * np.cos(longr) * np.cos(latr) + 12.0 * np.sin(longr) * np.cos(latr) + 7.0 * np.sin(latr)

    return vcorr

    #v=v+vcorr


###############
def over_plot_ions(name_q, wav_f, nflux_f, neflux_f, vcorr,ion_nm_gv, ion_wav_gv, ion_nm, ion_wav, ion_os):
    

    #ion_nm_gv, ion_wav_gv = ['SiII'], [1190.2]
    ##############    
    w_cos,f_cos, e_cos = wav_f, nflux_f, neflux_f

    
    ################        
    fnt_sz = 15
    lb_sz = 15
    lwd=2
    l_major,l_minor = 8,4
    ##############
    
    for jj in range(1):

        fig, axs = plt.subplots(1,1, sharex=True,sharey=True,figsize=(16,8))
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06,  top=0.95, wspace=0.05, hspace=0.05)

        fig.text(0.52, 0.02,  'V$_{LSR}$  '+r'(km s$^{-1}$ )', va='center', ha='center',fontsize=fnt_sz+2)   #10$^{3}$
        fig.text(0.025, 0.5, 'Normalized Flux', va='center', ha='center', rotation='vertical',fontsize=fnt_sz+2) 
        plt.suptitle(name_q, fontsize = 18)


        axs.set_xlim(-150, 550)
        axs.set_xticks([-100,0, 100, 200, 300, 400, 500])
        axs.set_ylim(0.0,1.2)
        axs.axhline(1.0,color='k',ls='-.')
        ################
        #for ii in range(N_lines):
        for kk in range(0,len(ion_nm_gv)):   ##

            print(ion_nm_gv[kk], ion_wav_gv[kk])
            cond1 = (ion_nm_gv[kk] == ion_nm)
            cond2 = [abs(num - ion_wav_gv[kk]) == min(abs(x - ion_wav_gv[kk]) for x in ion_wav) for num in ion_wav]


            id_l = np.where( AND(cond1, cond2))[0]



            vel = 299792.46*( (w_cos/ion_wav[id_l])  - 1.0)    ##+ vcorr

            # # # # #print(ion_wav_gv[kk])
            
            if (ion_wav_gv[kk] == 1304):
                w_shift = (-10.0/299792.46 + 1.0)*ion_wav[id_l]
                w_cos_tmp = w_cos*(ion_wav[id_l]/w_shift)
                vel = 299792.46*( (w_cos_tmp /ion_wav[id_l])  - 1.0)    ##+ vcorr
                np.savetxt(name_q+'/VPFIT/spec_2pix_1304_-10kmps.txt',np.c_[w_cos_tmp,f_cos, e_cos], fmt='%s')


                #w1526,f1526, e1526 = np.loadtxt(name_q+'/VPFIT/spec_2pix_1304.txt',unpack=True,usecols=[0,1,2],skiprows=0)
                #vel = 299792.46*( (w1526 /ion_wav[id_l])  - 1.0)    ##+ vcorr

                
            tt_str = ion_nm[id_l][0]+r" $\lambda$"+str(int(ion_wav[id_l][0]))
            axs.step(vel,f_cos,lw=1.0, label = tt_str)

            
            #anc2 = AnchoredText(ion_nm[id_l]+r" $\lambda$"+str(int(ion_wav[id_l])), loc="lower right", frameon=False,prop=dict(fontsize=12,color='r'))
            #axs.add_artist(anc2)


        leg  = axs.legend(frameon=False, fontsize=12, loc='lower right')
        

        # axs.axhline(1.0,color='k',ls='-.')    
        # axs.minorticks_on()
        # axs.tick_params(axis="both", labelsize= 10)
        # axs.tick_params(axis='both',which='major',direction='in', bottom=True, top=True, left=True, right=True,width=1,length=12)
        # axs.tick_params(axis='both',which='minor',direction='in', bottom=True, top=True, left=True, right=True,width=1,length=4)
        # plt.setp(axs.spines.values(), linewidth=1.0)
        plt.show()
                

    ###########===============================



def get_low_ion_stack_plot():

    npix = 0
    #filename = 'Codes/required_lines_FUV_for_stack.txt'
    filename = 'Codes/required_important_lines_stack_ver1.txt'
    
    df_req_line = pd.read_csv(filename,  delim_whitespace=1)
    idx_l = (df_req_line.loc_l  !=  -1)
    df_req_line = df_req_line[idx_l]
    

    nm_lns, wav_lns, f_lns, N_lns, loc_lns =  df_req_line.name.values, df_req_line.w_rest.values, df_req_line.f_osc.values, df_req_line.Night_flg.values, (df_req_line.loc_l.values).astype('int')
    
    
    
    ######+======================
    df_file = pd.read_csv('notes_obj_half.txt', delim_whitespace=1)
    dirname = df_file.dirname.values
    grism_flg = df_file.grism_flg.values
    wLIF1_SIC2, w130_LIF1, w130_LIF2, w160_130 = df_file.W_LIF1_SIC2A.values, df_file.W_G130_LIF1.values , df_file.W_G130_LIF2.values , df_file.W_G160_G130.values

    ra_q, dec_q = df_file.ra.values, df_file.dec.values
    ###########====================


    #for ii in range(len(dirname)):
    for ii in range(0,1):

        w_cos_full,f_cos_full, e_cos_full = np.loadtxt(dirname[ii]+'/VPFIT/spec_2pix.txt',unpack=True,usecols=[0,1,2],skiprows=0)
        ion_gv, wav_gv = np.loadtxt(dirname[ii]+'/VPFIT/lines1.pf',unpack=True,usecols=[0],skiprows=0, dtype='str'), np.loadtxt(dirname[ii]+'/VPFIT/lines1.pf',unpack=True,usecols=[1],skiprows=0)
        
        vcorr = get_vLSR_corr(ra_q[ii], dec_q[ii])

        over_plot_ions(dirname[ii],w_cos_full,f_cos_full,e_cos_full,vcorr,ion_gv, wav_gv, nm_lns, wav_lns,f_lns)

        
            
    
######################==================
#check_candidate_absorbers_new()
#check_low_ion_lines()
get_low_ion_stack_plot()



