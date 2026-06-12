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
import seaborn as sns
from scipy.interpolate import interp1d
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tck
import matplotlib.patches as mpatches
from astropy.modeling.models import custom_model
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
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


from VoigtFit.container.lines import show_transitions
from VoigtFit.funcs.voigt import Voigt   ###, convolve_numba
from scipy.signal import fftconvolve, gaussian


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

##################

def get_velocity(z1,z2):
    beta = ( (1+z1)**2 - (1+z2)**2 ) /  ( (1+z1)**2 + (1+z2)**2 )
    velocity = beta * 299792.46
    return  velocity
#############
def delvz(z1,z2): 
    beta = ((1.+z2)**2. - (1.+z1)**2.)/((1.+z2)**2. + (1.+z1)**2.) 
    vout = beta*2.997e5   
    return vout


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

    #print(ra_deg, dec_deg)
    sky_coord = SkyCoord(ra = ra_deg * u.degree, dec = dec_deg * u.degree, frame='icrs')
    galactic_coord = sky_coord.galactic

    # galactic_latitude = galactic_coord.b.degree
    # galactic_longitude = galactic_coord.l.degree

    latr   = galactic_coord.b.radian  
    longr  = galactic_coord.l.radian  

    vcorr  = 9.0 * np.cos(longr) * np.cos(latr) + 12.0 * np.sin(longr) * np.cos(latr) + 7.0 * np.sin(latr)

    return vcorr

    #v=v+vcorr

###########========
from scipy.special import wofz

# def voigt(x, mu, sigma, gamma):
#     z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
#     return wofz(z).real / (sigma * np.sqrt(2 * np.pi))



def get_profile(ion_wav, pars, wl_arr = None, vel_arr = None, 
                    resolution = None, redshift = None, **kwargs):

    """
    Get absorption line profile for specified ion_wavelength

     Parameters
     ----------
     ion_wav: `str`
         ion_wavelength to get profile for
     pars: `dict`
         parameters for line profile
         must include "v", "b", and "logN"
     resolution: `number`, optional, must be keyword
         instrument resolution for convolution in km/s
         defualt to 20 km/s
     redshift: `number`, optional, must be keyword
         system redshift, default to 0
     wl_arr: `list-like`, optional, must be keyword
         array of wavelengths to compute spectrum to
     vel_arr: `list_like`, optional, must be keyword
         array of velocities to compute spectrum to
     """

    # find the right transition

    line_list_file = '/Users/smishra/vpfit/atom_plot.dat'
    
    wl_cen,f_ij,gamma = np.loadtxt(line_list_file,unpack=True,usecols=[1,2,3],comments='#',skiprows=1)
    names = np.loadtxt(line_list_file,unpack=True,usecols=[0],comments='#',dtype='str',skiprows=1)
    wl_cen_int = [int(wavelength) for wavelength in wl_cen]
    name_club =  np.array([f"{name}_{wavelength}" for name, wavelength in zip(names, wl_cen_int)])
    
    #df_atom = pd.read_csv('/Users/smishra/vpfit/atom_plot.dat',  delim_whitespace=1)
    #names, wl_cen, f_ij, gamma = df_atom.ion_nm.values, df_atom.ion_wav.values, df_atom.ion_fos.values, df_atom.ion_gamma.values
    #wl_cen_int = [int(wavelength) for wavelength in wl_cen]
    #name_club =  np.array([f"{name}_{wavelength}" for name, wavelength in zip(names, wl_cen_int)])

    
    l0, f, gam =  wl_cen[name_club == ion_wav], f_ij[name_club == ion_wav], gamma[name_club == ion_wav]


    print(ion_wav,l0, f, gam)
    #### from /Users/smishra/anaconda3/lib/python3.1/site-packages/VoigtFit/static/linelist.dat
    # atomic_data = show_transitions(ion = ion_wav.split("_")[0])
    # names = [at[0] for at in atomic_data]
    # match = np.array(names) == ion_wav
    # _, _, l0, f, gam, _ = np.array(atomic_data)[match][0]

    
    # check for resolution
    if resolution == None:
        resolution = 20. #km/s for COS

    # check redshift
    if redshift == None:
        redshift = 0.

    v = pars["v"]
    b = pars["b"]
    logN = pars["logN"]


    #print('Sapna',v,b, logN, ion_wav, l0, f, gam)

    
    if ((type(v) == float) | (type(v) == np.float64)):
        v = list([v])
        b = list([b])
        logN = list([logN])

    #l_center = l0 * (redshift + 1.)
    #wl_arr = (vel_arr / 299792.46 * l_center) + l_center

    
    # # # find wl_line
    # # if vel_arr == None:
    # #     if wl_arr == None:
    # #         # use default wavelength range of +-500 km/s
    # #         wl_arr = np.arange(l_center - .01*250, l_center +0.01*250, 0.01)
    # #     vel_arr = (wl_arr - l_center)/l_center*(speed_of_light.to(u.km/u.s).value)
    # # elif wl_arr == None:
    # #     wl_arr = (vel_arr / (speed_of_light.to(u.km/u.s).value) * l_center) + l_center
    
    
    tau = np.zeros_like(vel_arr)

    for (vv, bb, NN) in zip(v, b, logN):
        tau += Voigt(wl_arr, l0, f, 10**NN, 1.e5*bb, gam, z = vv/299792.46 )

        
    # Compute profile
    profile_int = np.exp(-tau)

    # # convolve with instrument profile
    # if isinstance(resolution, float):
    #     pxs = np.diff(wl_arr)[0] / wl_arr[0] * 299792.46
    #     sigma_instrumental = resolution / 2.35482 / pxs
    #     LSF = gaussian(len(wl_arr) // 2, sigma_instrumental)
    #     LSF = LSF/LSF.sum()
    #     profile = fftconvolve(profile_int, LSF, 'same')
    # else:
    #     print('H')
    #     profile = voigt.convolve_profile(profile_int, resolution)

    
    out = {
        "wl":wl_arr,
        "vel":vel_arr,
        "spec":profile_int
        }

    return out


def plot_low_ion_stack_ver2(name_q, vcorr, ion_nm, ion_wav, ion_os, spec_file, fitted_file):

    

    print(ion_nm, ion_wav, ion_os)

    
    nm_to_append   = np.array(['SiIV', 'SiIV', 'CIV', 'CIV'])
    wav_to_append  = np.array([1393.76018, 1402.77291, 1548.204, 1550.77755])
    os_to_append   = np.array([0.513, 0.254, 0.1899, 0.09475])

    ion_nm = np.concatenate([ion_nm, nm_to_append])
    ion_wav = np.concatenate([ion_wav, wav_to_append])
    ion_os = np.concatenate([ion_os, os_to_append])
    
    df_fit = pd.read_csv(fitted_file,  delim_whitespace=1)


    w_cos_full,f_cos_full = np.loadtxt(spec_file,unpack=True,usecols=[0,1],skiprows=0)


    # vel_full = 299792.46*( (w_cos_full/ion_wav[id_ion])  - 1.0)  + vcorr
    # id_v = (vel_full > -150) & ( vel_full < 550)
    # vel, w_cos, f_cos = vel_full[id_v==True], w_cos_full[id_v==True], f_cos_full[id_v==True]

    # plt.step(vel,f_cos,color='darkblue',lw=1.0 )
    # ion_pars = {"v":[254.8119, 318.5526, 367.7530], "b":[19.19260, 22.6366, 13.36989] , "logN":[]}

    N_lines = len(ion_wav)

    P = 3
    Q = int(N_lines / P) + (N_lines % P != 0)
    #Q = Q

    
    vel_list_dk, b_list_dk, N_list_dk = np.loadtxt(name_q+'/DK_line_list.txt',unpack=True,usecols=[0,2,4],skiprows=0)
    ion_list_dk = np.loadtxt(name_q+'/DK_line_list.txt',unpack=True,usecols=[6],skiprows=0, dtype='str')     


    
    v_siiv_dk,b_siiv_dk,N_siiv_dk = vel_list_dk[ion_list_dk == 'SiIV'],b_list_dk[ion_list_dk == 'SiIV'], N_list_dk[ion_list_dk == 'SiIV']
    v_civ_dk,b_civ_dk,N_civ_dk  = vel_list_dk[ion_list_dk == 'CIV'], b_list_dk[ion_list_dk == 'CIV'], N_list_dk[ion_list_dk == 'CIV']

    ################        
    fnt_sz = 15
    lb_sz = 15
    lwd=2
    l_major,l_minor = 8,4
    ##############

    
    for jj in range(1):

        fig, axs = plt.subplots(P,Q, sharex=True,sharey=True,figsize=(16,10))
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06,  top=0.95, wspace=0.05, hspace=0.05)

        fig.text(0.52, 0.02,  'V$_{LSR}$  '+r'(km s$^{-1}$ )', va='center', ha='center',fontsize=fnt_sz+2)   #10$^{3}$
        fig.text(0.025, 0.5, 'Normalized Flux', va='center', ha='center', rotation='vertical',fontsize=fnt_sz+2)
         
        plt.suptitle(name_q, fontsize = 18)
        axs = axs.ravel()
        
        num_plots = P * Q

        ################
        #for ii in range(N_lines):
        for ii in range(0, num_plots):

            plt_loc = ii
            
            if jj*num_plots + ii < N_lines:

                id_ion = jj*num_plots + ii

                
                axs[plt_loc].set_xlim(-100, 600)
                axs[plt_loc].set_xticks([0, 100, 200, 300, 400,500])
                axs[plt_loc].set_ylim(0.0,1.5)   
                axs[plt_loc].axhline(1.0,color='k',ls='-.')
                anc2 = AnchoredText(ion_nm[id_ion]+r" $\lambda$"+str(int(ion_wav[id_ion])), loc="lower right", frameon=False,prop=dict(fontsize=12,color='r'))
                axs[plt_loc].add_artist(anc2)
                    
                    
                vel_full = 299792.46*( (w_cos_full/ion_wav[id_ion])  - 1.0) 
                
                id_v = (vel_full > -250) & ( vel_full < 650)

                vel, w_cos, f_cos = vel_full[id_v==True], w_cos_full[id_v==True], f_cos_full[id_v==True]
                
                
                axs[plt_loc].step(vel + vcorr ,f_cos,color='darkblue',lw=1.0 )
                
                ######################=======================================================
                if ( (ion_nm[id_ion] != 'SiIV') & (ion_nm[id_ion] != 'CIV') ):
                    
                    id_match =  np.where( ion_nm[id_ion]  == ion_list_dk)[0]
                    
                    if np.size(id_match) > 1:
                        for tt in range(len(id_match)):
                            axs[plt_loc].axvline(vel_list_dk[id_match][tt],ymin=0.0, ymax=0.4,color='magenta',ls='-.',lw=1.2)
                    elif np.size(id_match) == 1:
                        axs[plt_loc].axvline(vel_list_dk[id_match],color='magenta',ls='-.',lw=1.2)

                        
                    ion_pars = {"v":[], "b":[], "logN":[]}
                    n_comp = len(df_fit.z[df_fit.name == ion_nm[id_ion]].values)
                    tag = ion_nm[id_ion]+"_"+str(int(ion_wav[id_ion])) 

                    
                    for kk in range(n_comp):
                        
                        z_lmc = df_fit.z[df_fit.name == ion_nm[id_ion]].iloc[kk]
                        z_lmc_err = df_fit.z_err[df_fit.name == ion_nm[id_ion]].iloc[kk]

                        V1 =  299792.46 * (    ( (1.0 + z_lmc)**2 - 1.0 ) / ( (1.0 + z_lmc)**2 + 1.0) )
                        V1_err = 299792.46 * (    ( (1.0 + z_lmc_err)**2 - 1.0 ) / ( (1.0 + z_lmc_err)**2 + 1.0) )

                        b1 =  df_fit.b[df_fit.name == ion_nm[id_ion]].iloc[kk]
                        N1 =  df_fit.N[df_fit.name == ion_nm[id_ion]].iloc[kk]
                        
                        ion_pars_single = {"v":[V1], "b":[b1], "logN":[N1]}
                        profile_single = get_profile(tag, ion_pars_single, wl_arr= w_cos, vel_arr = vel)
                        
                        axs[plt_loc].plot(profile_single["vel"]+ vcorr, profile_single["spec"], color='c',lw=1.5)
                        axs[plt_loc].axvline(V1+ vcorr, ymin=0.5, ymax=0.9, color='c', linestyle='-',lw=1.5)
                        
                        ion_pars["v"].append(V1)
                        ion_pars["b"].append(b1)
                        ion_pars["logN"].append(N1)
                    #######################=======================================

                    profile = get_profile(tag, ion_pars, wl_arr= w_cos, vel_arr = vel)
                    #axs[plt_loc].plot(profile["vel"]+ vcorr, profile["spec"], color='r',lw=3.0)
                    
                #############=======================================================================
                   
                if (ion_nm[id_ion] == 'SiIV'):

                    ion_pars = {"v":[], "b":[], "logN":[]}
                    n_comp = len(df_fit.z[df_fit.name == ion_nm[id_ion]].values)
                    tag = ion_nm[id_ion].split('_')[0]+"_"+str(int(ion_wav[id_ion]))

                    for bb2 in range(len(v_siiv_dk)):

                        V1 =  v_siiv_dk[bb2]       
                        b1 =  b_siiv_dk[bb2] 
                        N1 =  N_siiv_dk[bb2] 
                        
                        #ion_pars_single_siiv = {"v":[V1], "b":[b1], "logN":[N1]}
                        #profile_single_siiv  = get_profile(tag, ion_pars_single, wl_arr= w_cos, vel_arr = vel)
                        #axs[plt_loc].plot(profile_single_siiv["vel"], profile_single_siiv["spec"], color='y',lw=1.0)
                    
                        ion_pars["v"].append(V1)
                        ion_pars["b"].append(b1)
                        ion_pars["logN"].append(N1)
                        axs[plt_loc].axvline(v_siiv_dk[bb2], ymin=0.0, ymax=0.4,color='y',ls='-.',lw=2)

                    
                    profile_siiv = get_profile(tag, ion_pars, wl_arr= w_cos, vel_arr = vel)
                    axs[plt_loc].plot(profile_siiv["vel"], profile_siiv["spec"], color='y',lw=3.0)   ### not added vcorr as DK as already added


                if (ion_nm[id_ion] == 'CIV'):


                    ion_pars = {"v":[], "b":[], "logN":[]}
                    n_comp = len(df_fit.z[df_fit.name == ion_nm[id_ion]].values)
                    tag = ion_nm[id_ion]+"_"+str(int(ion_wav[id_ion])) 

                    for bb2 in range(len(v_civ_dk)):

                        V1 =  v_civ_dk[bb2]       
                        b1 =  b_civ_dk[bb2] 
                        N1 =  N_civ_dk[bb2] 
                        
                        #ion_pars_single_civ = {"v":[V1], "b":[b1], "logN":[N1]}
                        #profile_single_civ  = get_profile(tag, ion_pars_single, wl_arr= w_cos, vel_arr = vel)
                        #axs[plt_loc].plot(profile_single_civ["vel"], profile_single_civ["spec"], color='g',lw=1.0)
                    
                        ion_pars["v"].append(V1)
                        ion_pars["b"].append(b1)
                        ion_pars["logN"].append(N1)
                        axs[plt_loc].axvline(v_civ_dk[bb2], ymin=0.0, ymax=0.4,color='g',ls='-.',lw=2)

                    profile_civ = get_profile(tag, ion_pars, wl_arr= w_cos, vel_arr = vel)
                    axs[plt_loc].plot(profile_civ["vel"], profile_civ["spec"], color='g',lw=3.0)  ### not added vcorr as DK as already added


                axs[plt_loc].axhline(0.1, color='k',ls='-.')
                axs[plt_loc].axvline(150.0, color='orange',ls='-.')
                axs[plt_loc].minorticks_on()
                axs[plt_loc].tick_params(axis="both", labelsize= 10)
                axs[plt_loc].tick_params(axis='both',which='major',direction='in', bottom=True, top=True, left=True, right=True,width=1,length=12)
                axs[plt_loc].tick_params(axis='both',which='minor',direction='in', bottom=True, top=True, left=True, right=True,width=1,length=4)
                plt.setp(axs[plt_loc].spines.values(), linewidth=1.0)

            else:
                axs[plt_loc].set_axis_off()

    ###########===============================



def get_low_ion_stack_plot():

    npix = 2
    #filename = 'Codes/required_lines_FUV_for_stack.txt'
    filename = 'Codes/required_important_lines_stack_ver1.txt'

    df_req_line = pd.read_csv(filename,  delim_whitespace=1)
    idx_l = (df_req_line.loc_l  !=  -1)
    df_req_line = df_req_line[idx_l]
    
    nm_lns, nm_lns_full   =  df_req_line.name.values, df_req_line.name_full.values 
    wav_lns, f_lns, N_lns, loc_lns = df_req_line.w_rest.values, df_req_line.f_osc.values, df_req_line.Night_flg.values, (df_req_line.loc_l.values).astype('int')
    
    ##########======================
    # df_file = pd.read_csv('notes_obj_half.txt', delim_whitespace=1)
    # dirname = df_file.dirname.values
    # grism_flg = df_file.grism_flg.values
    # wLIF1_SIC2, w130_LIF1, w130_LIF2, w160_130 = df_file.W_LIF1_SIC2A.values, df_file.W_G130_LIF1.values , df_file.W_G130_LIF2.values , df_file.W_G160_G130.values
    # ra_q, dec_q = df_file.ra.values, df_file.dec.values


    df_lmc = pd.read_csv('LMC_sightlines_line_info.txt', delim_whitespace=1)
    dirname = df_lmc.dirname.values
    ra_q, dec_q = df_lmc.ra.values, df_lmc.dec.values
    line_tag = ['SiII1260','SiII1193','SiII1190','SiII1526','SiII1304','SiIII1206','SII1253','SII1250','OI1302','AlII1670','FeII1144','FeII1608', 'SiI1562', 'SiI1631', 'NiII1370', 'NiII1317', 'NI1199']  #'CII1334','CII1036','FeIII1122',

    
   
    #for ii in range(len(dirname)):
    for ii in range(0,1):

        spec_file = dirname[ii]+'/VPFIT/spec_2pix.txt'
        fitted_file = dirname[ii]+'/VPFIT/fitted_param.txt' ###fitted_param_Si-b-tied.txt' #'
        vcorr = get_vLSR_corr(ra_q[ii], dec_q[ii])
        

        df_fit = pd.read_csv(fitted_file,  delim_whitespace=1)
        w_cos_full,f_cos_full = np.loadtxt(spec_file,unpack=True,usecols=[0,1],skiprows=0)
   
        
        #for jj in range(len(line_tag)):
        for jj in range(0,1):


            ion_nm, ion_wav, ion_os = nm_lns[nm_lns_full == line_tag[jj]], wav_lns[nm_lns_full == line_tag[jj]], f_lns[nm_lns_full == line_tag[jj]]

            vel_full = 299792.46*( (w_cos_full/ion_wav)  - 1.0)

            id_v = (vel_full > -250) & ( vel_full < 650)

            vel, w_cos, f_cos = vel_full[id_v==True], w_cos_full[id_v==True], f_cos_full[id_v==True]

            plt.step(vel + vcorr ,f_cos,color='darkblue',lw=1.0 )
            plt.show()
            
          

        

        #     line_fit_name, line_fit_wav, line_fit_fos = [], [], []
            
        #     for jj in range(len(line_tag)):

        #         #print(line_tag[jj], df_lmc[line_tag[jj]].iloc[ii])
        #         if df_lmc[line_tag[jj]].iloc[ii] == 'Y':
        #             line_fit_name.append( nm_lns[nm_lns_full == line_tag[jj]])
        #             line_fit_wav.append( wav_lns[nm_lns_full == line_tag[jj]])
        #             line_fit_fos.append( f_lns[nm_lns_full == line_tag[jj]])

        #     line_fit_name = np.concatenate(line_fit_name, axis=0)
        #     line_fit_wav  = np.concatenate(line_fit_wav, axis=0)
        #     line_fit_fos  = np.concatenate(line_fit_fos, axis=0)

        #     spec_file = dirname[ii]+'/VPFIT/spec_2pix.txt'
        #     fitted_file = dirname[ii]+'/VPFIT/fitted_param.txt' ###fitted_param_Si-b-tied.txt' #'

        #     v

        #     #print(line_fit_name)
        #     plot_low_ion_stack_ver2(dirname[ii], vcorr, line_fit_name, line_fit_wav,line_fit_fos, spec_file, fitted_file)              
            
        #     ################
        #     pdf.savefig()
        #     plt.show()
        #     plt.clf()
        #     plt.cla()
        #     plt.close()
        # ######===================

    
######################==================
get_low_ion_stack_plot()
