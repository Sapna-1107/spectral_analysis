import pyforest
import sys
#sys.path.insert(1, '/home/sapnamisra/GalaxyQSO/Codes/'); from gaussfold import gaussfold

from pydl.goddard.astro import gcirc
import math
import subprocess
from astropy.io import fits
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
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import heapq
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy import interpolate
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tck
from random import randrange, uniform
import random
from scipy.stats import skew
from scipy.integrate import quad
from matplotlib.ticker import NullFormatter,FormatStrFormatter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gmean

from spectres import spectres

######################
AND = np.logical_and
OR  = np.logical_or
SH  = np.shape
#################### Initial parameters ###########

###################
def calcew(wl,fl,cnt,lmts,z=0.0):
    pix = np.where((wl >= lmts[0]) & (wl <= lmts[1]))   
    pixp = tuple(np.array(pix)+1)  
    ew = sum((cnt - fl[pix])*(wl[pixp]-wl[pix])) 
    #sigew = np.sqrt(sum( (err[pix]*(wl[pixp]-wl[pix]))**2 ))
    return ew #,sigew
###############







###################
def zhu2013_eq5(w,z):
    g0,alp_g,z_g,beta_g = 0.63, 5.38, 0.41, 2.97
    w_star0, alp_w, z_w, beta_w = 0.33, 1.21, 2.24, 2.43
    g_z = g0*( ( (1.0 + z)**alp_g )/ (1.0 + (z/z_g)**beta_g))
    w_star_z = w_star0 * (  ((1.0+z)**alp_w)/ (1.0 + (z/z_w)**beta_w ))
    return g_z* np.exp(-w/w_star_z)



def get_N_abs_dN_dz(zemi_q):
    lmin, lmax = 4000.0,9000.0
    vprox      = 5000.0
    c_speed = 299792.46
    ion_wav = [2796.3543,1548.2041]

    for ii in range(len(ion_wav)):
        lam_prox = (vprox/c_speed)*ion_wav*(1+zemi_q)
        z_low    = max([lmin/ion_wav-1.0, (1+zemi_q)*1215.67/ion_wav -1.0])
        z_up     = min([lmax/ion_wav-1.0, ((1+zemi_q)*ion_wav - lam_prox)/ion_wav-1.0])
        dN_dz_MgII = quad(zhu_eq5, 1.2,1.8, args=(z_avg[ii]))[0] 



from astropy.modeling.models import custom_model
from astropy.modeling import models, fitting



###################
def fit_guass_2796(xvar_fit,yvar_fit,ht_in,sig_in):
    @custom_model
    def gauss_mgii2796(x, N=0.0,b=1.0,Cn=1.0):
        return  Cn  - N*np.exp(-0.5*( ( x - 2796.3543 )/b )**2) 
    g_init = gauss_mgii2796( N=ht_in,b=sig_in,Cn=1.0)  ##models.Gaussian1D(amplitude=-100., mean=2796.3543, stddev=2.23)   ##
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, xvar_fit,yvar_fit)
    return g


######################
def line1(x, N,b):
        return N*np.exp(-0.5*( ( x - 2796.3543 )/b )**2)


def line2(x, N,b):
        return N*np.exp(-0.5*( ( x - 1215.6701 )/b )**2)

    
##########################    

def get_lambda_from_b(b_line):

    b_all = np.sqrt( b_line**2 + 17**2)
    FWTM = np.sqrt( 8 * np.log(2) ) * b_all
    del_lambda = (FWTM * 1450.0 ) / 299792.46
    return del_lambda



def simulated_spectrum(wave,flg_random_abs = 0, flg_lya_abs= 0, SNR_l= 0 ,SNR_u = 0, seed_spec = 0, continumm_level= 0.0, v_sigma = 0.0):

    df = pd.read_csv('danforth_lya_system.txt', delim_whitespace=1)
    idx = (df.logN > 12.9); df = df[idx]

   
    random.seed(seed_spec+1); SNR_sdss = uniform(SNR_l,SNR_u)
    sigma_noise = continumm_level/SNR_sdss
    np.random.seed(seed_spec+2); noise_std = np.random.normal(0.0,sigma_noise, len(wave))      #### 0.0*wave + 1.0

    
    flux = np.zeros(len(wave)) + continumm_level
    flux = flux + noise_std

    # ######################===================================================================================
    # if flg_random_abs == 1:

    #     random.seed(seed_spec+3); N_abs = random.randint(1,5)
    #     for ll in range(N_abs):

    #         random.seed(seed_spec+3); id_dan_r = random.randint(0, len(df)-1)
    #         random.seed(seed_spec+4); ew_lya_r = (df['EW'].iloc[id_dan_r] / 1000.0)  
    #         random.seed(seed_spec+5); sigma_lya_r =  get_lambda_from_b( df['b'].iloc[id_dan_r])
    #         ht_lya_r =   ew_lya_r / ( (sigma_lya_r) * np.sqrt(2.0 * math.pi) )


    #         random.seed(seed_spec+6+ll); wave_abs = uniform(1200.0, 1230.0)

    #         #print(wave_abs)


    #         flux = flux - (ht_lya_r)* np.exp( -0.5*( ( wave - wave_abs ) /sigma_lya_r  )**2 )  \
    #                     - (ht_lya_r/5.26)*np.exp( -0.5*( ( wave - (wave_abs - 189.9478) ) /sigma_lya_r  )**2 ) \
    #                     - (ht_lya_r/14.36)*np.exp( -0.5*( ( wave - (wave_abs- 243.1333 ) ) /sigma_lya_r  )**2 ) \
    #                     - (ht_lya_r/8.9)*np.exp(-0.5*( ( wave - (wave_abs - 265.927) ) /sigma_lya_r  )**2 )
    # ################==============================================================================================

    
    if flg_lya_abs == 1:

        #print(df['zabs'].iloc[id_dan], df['Sig'].iloc[id_dan], df['EW'].iloc[id_dan], df['b'].iloc[id_dan], df['logN'].iloc[id_dan], np.median(np.diff(wave)) )
        #Const_fact = (2.654e-15)* 1215.6701 * 0.416400
        #1.0 * np.exp(- 10**df['logN'].iloc[id_dan] / Const_fact )

        random.seed(seed_spec+3); id_dan = random.randint(0, len(df)-1)  #1
        random.seed(seed_spec+4); ew_lya = (df['EW'].iloc[id_dan] / 1000.0)  
        random.seed(seed_spec+5); sigma_lya =  get_lambda_from_b( df['b'].iloc[id_dan])

        #np.random.seed(seed_spec+4); ew_lya = float(np.random.normal(0.271,0.1, 1))
        #random.seed(seed_spec+5); sigma_lya =  float(get_lambda_from_b( np.random.normal(35,5, 1)))

        ht_lya =   ew_lya / ( (sigma_lya) * np.sqrt(2.0 * math.pi) )

        #random.seed(seed_spec+6); w_loc = uniform( 1215.6701 - 1215.6701*1000.0/299792.46, 1215.6701 + 1215.6701*1000.0/299792.46)

        np.random.seed(seed_spec+6); w_loc =  np.random.normal(1215.6701, 1215.6701*v_sigma/299792.46)
        
        flux = flux - (ht_lya)* np.exp( -0.5*( ( wave - w_loc ) / sigma_lya  )**2 ) 


        print('Given EW:',ew_lya , 'Given sigma:' , sigma_lya, 'Given ht: ',ht_lya, 'Given W_loc:', w_loc )

        
        #####################
        f_lya = lambda x, ht1, w0, b: 1.0 - ht1 * np.exp(-0.5*( ( x - w0 )/b )**2)      #fa*(x/2.0)**a

        line_lya = lambda x, ht1, w0, b: ht1 * np.exp(-0.5*( ( x - w0 )/b )**2 )

                                                         

        #start = (ht_lya, w_loc, sigma_lya)
        #popt, pcov = curve_fit(f_lya, wave,flux , sigma = noise_std, p0 = start,absolute_sigma=True)  ##; perr = np.sqrt(np.diag(pcov)) #,  

        #print('Given EW:',ew_lya , 'Observed EW:' , calcew(wave,flux,[1210, 1220],z=0.0), 'Observed line-fit EW: ', quad(line_lya, 1210,1220, args=( popt[0], popt[1], popt[2]))[0] )

        # #############
        # plt.plot(wave, flux)
        # plt.plot(wave, f_lya(wave,*popt), color='r')
        # plt.show()
        # ###########################

    if flg_lya_abs == 1:
        return wave,flux,noise_std,(wave*0.0 + 1.0/np.std(noise_std)),w_loc,ew_lya,sigma_lya,ht_lya
    else:
        return wave,flux,noise_std, (wave*0.0 + 1.0/np.std(noise_std))
        
    
        
        
#######################
def do_binning(wav_big,flux_big,sig_big,SNR_big,m_wav):

    #SNR_big = flux_big / sig_big
    
    N1 = len(m_wav)
    mean_arr, med_arr, wmean_arr, cmean_arr, gmean_arr  = np.zeros(N1), np.zeros(N1), np.zeros(N1), np.zeros(N1), np.zeros(N1)
    

    for j in range(N1-1):
        id_fnd = np.where(AND( wav_big >=  m_wav[j] , wav_big < m_wav[j+1] ))
        if np.size(id_fnd) >= 3:
            
            med_arr[j]          = np.median(flux_big[id_fnd])
            mean_arr[j]         = np.mean(flux_big[id_fnd])
            wmean_arr[j]        = np.average(flux_big[id_fnd], weights = SNR_big[id_fnd])


            id_g = np.where( AND(flux_big[id_fnd] > 0.0, flux_big[id_fnd] != 0.0) )[0]
            if np.size(id_g) > 1:
                gmean_arr[j]        = gmean(flux_big[id_fnd][id_g])
            ###
            sigclip = sigma_clip(flux_big[id_fnd], sigma=5.0, maxiters=10)
            cmean_arr[j]  = np.mean( flux_big[id_fnd][sigclip.mask == False] )
            
        else:
            continue


    ################
    wav_tmp      =  m_wav[0:N1-1]
    med_tmp      =  med_arr[0:N1-1]
    mean_tmp     =  mean_arr[0:N1-1]
    wmean_tmp    = wmean_arr[0:N1-1]
    cmean_tmp    = cmean_arr[0:N1-1]
    gmean_tmp    = gmean_arr[0:N1-1]
    
    #plt.plot(wav_tmp,mean_tmp)
    #plt.show()
    
    return wav_tmp,med_tmp,mean_tmp,wmean_tmp,cmean_tmp,gmean_tmp


########################

    
def stack_main_simulation(N_sim, SNR_l=0.0, SNR_u=0.0, CF_lya = 0.0, v_sigma = 0.0):

    w_loc_a,ew_lya_a,sigma_lya_a,ht_lya_a = [],[],[],[]
    wav_all,flux_all,sig_all,SNR_all = [],[],[],[]
    id_all_bs = []
    continumm_level = 1.0

    ###############################
    id_all = np.arange(0,N_sim,1)
    cov_frac_lya = int(CF_lya * N_sim)
    np.random.seed(111); rnd_lya_id = np.random.choice(id_all,replace=False,size=cov_frac_lya)  ###replase=False no repeated entries
    Lya_flg = np. zeros((N_sim), dtype=int)
    Lya_flg[rnd_lya_id] = 1

    #######################

    C0 = 91.0
    gamma = 1.24
    dNdz = C0*(1.0 + 0.2)**gamma
    Dz_mean = (2000.0/299792.46)  ##*1.179
    lambda_p =  dNdz*Dz_mean
    CF_u = np.exp(-lambda_p)
    
    cov_frac_uncor = int((CF_u)*N_sim)   ## Zhu+2013: dN/dz = 0.71 at z~0.5 (for EW>0.2A),Delta_z = 0.18; N=dN_dz*Delta_z=0.13
    np.random.seed(222); rnd_uncor_id = np.random.choice(id_all,replace=False,size=cov_frac_uncor)  ###replase=False no repeated entries
    flg_random_abs = np. zeros((N_sim), dtype=int)
    flg_random_abs[rnd_uncor_id] = 0
    ###################
    
    wmin,wmax,reso  = 1200,1230, 0.0330811   #####this is for the 3-pix smooth spectrum of COS    #### 0.0299072
    wave1 = np.arange(wmin,wmax,reso)

    #np.random.seed(22211)
    
    for k in range(N_sim):

         
        if Lya_flg[k] ==1:
            wave2,flux,sig,snr_t,w_c,ew,sd,ht = simulated_spectrum(wave1, flg_random_abs = flg_random_abs[k] , flg_lya_abs = Lya_flg[k] , \
                                                             SNR_l = SNR_l, SNR_u = SNR_u, seed_spec = 99900+k, continumm_level= 1.0, v_sigma = v_sigma) 
            w_loc_a.append(w_c); ew_lya_a.append(ew); sigma_lya_a.append(sd); ht_lya_a.append(ht)
            
        else:
            wave2,flux,sig,snr_t = simulated_spectrum(wave1, flg_random_abs = flg_random_abs[k] , flg_lya_abs = Lya_flg[k] , SNR_l = SNR_l, SNR_u = SNR_u, seed_spec = 99900+k, continumm_level= 1.0, v_sigma = v_sigma)
            
        wav_all.append(wave2)
        flux_all.append(flux)
        sig_all.append(sig)
        id_all_bs.append(np.zeros(len(wave2),dtype='int')+k)
        SNR_all.append(snr_t)
       

    ##################################
    
    wave_all,flux_all, sig_all, SNR_all = np.ravel(wav_all),np.ravel(flux_all), np.ravel(sig_all), np.ravel(SNR_all)
    id_all_bs = np.ravel(id_all_bs)

    wave,md,mn,wmn,cmn,gmn = do_binning(wave_all,flux_all,sig_all,SNR_all,m_wav)

    ############ for bootstrap ==============
    BS_itr = 200
    
    ew_md_bs, ew_mn_bs, ew_wmn_bs, ew_cmn_bs, ew_gmn_bs = np.zeros(BS_itr),np.zeros(BS_itr),np.zeros(BS_itr),np.zeros(BS_itr),np.zeros(BS_itr)

    vel_tol = 500.0
    w_l,w_u = 1215.6701 - 1215.6701*vel_tol/299792.46, 1215.6701 + 1215.6701*vel_tol/299792.46

    for ii in range(BS_itr):

        #print(ii)
        bs_num = random.choices(range(0, N_sim), k= N_sim)
        result = []
        
        for b in bs_num:
            result.extend(np.where(id_all_bs == b)[0])
    
        flux_loc = flux_all[result]
        wave_loc = wave_all[result]
        sig_loc  = sig_all[result]
        SNR_loc  = SNR_all[result]

        w_tmp,md_tmp,mn_tmp,wmn_tmp,cmn_tmp,gmn_tmp = do_binning(np.array(wave_loc),np.array(flux_loc), np.array(sig_loc), np.array(SNR_loc),m_wav)
        ew_md_bs[ii]  = calcew(w_tmp,md_tmp, np.median(md_tmp),   [w_l,w_u],z=0.0)
        ew_mn_bs[ii]  = calcew(w_tmp,mn_tmp, np.median(mn_tmp), [w_l,w_u],z=0.0)
        ew_wmn_bs[ii] = calcew(w_tmp,wmn_tmp,np.median(wmn_tmp),[w_l,w_u],z=0.0)
        ew_cmn_bs[ii] = calcew(w_tmp,cmn_tmp,np.median(cmn_tmp),[w_l,w_u],z=0.0)
        ew_gmn_bs[ii] = calcew(w_tmp,gmn_tmp,np.median(gmn_tmp),[w_l,w_u],z=0.0)

        #print(ew_md_bs[ii],ew_mn_bs[ii],ew_wmn_bs[ii],ew_cmn_bs[ii],ew_gmn_bs[ii])
        
    ######################################

    #print(round(np.std(ew_md_bs),3),round(np.std(ew_mn_bs),3),round(np.std(ew_wmn_bs),3),round(np.std(ew_cmn_bs),3),round(np.std(ew_gmn_bs),3))
    # return wave, md, str(int(SNR_med)), ew, ew_err
    return wave,md,mn,wmn,cmn,gmn,np.array(ew_lya_a), 299792.46*(np.array(w_loc_a)/1215.6701 - 1.0),\
        round(np.std(ew_md_bs),3),round(np.std(ew_mn_bs),3),round(np.std(ew_wmn_bs),3),round(np.std(ew_cmn_bs),3),round(np.std(ew_gmn_bs),3)

           
def get_EW_exp_stack(w,md,mn,wmn,cmn,gmn,ew, CF_lya):

    sigclip = sigma_clip(ew, sigma=5.0, maxiters=10)
        
    ew_md_s,  ew_md_exp   = round(calcew(w,md,np.median(md),[w_l,w_u],z=0.0) ,3),  round(CF_lya*np.median(ew),3)
    ew_mn_s,  ew_mn_exp   = round(calcew(w,mn,np.median(mn),[w_l,w_u],z=0.0) ,3),  round(CF_lya*np.mean(ew),3)
    ew_cmn_s, ew_cmn_exp  = round(calcew(w,cmn,np.median(cmn),[w_l,w_u],z=0.0),3), round(CF_lya * np.mean( ew[sigclip.mask == False] ),3)
    ew_gmn_s, ew_gmn_exp  = round(calcew(w,gmn,np.median(gmn),[w_l,w_u],z=0.0),3), round(CF_lya*gmean(ew),3)
    ew_wmn_s,  ew_wmn_exp   = round(calcew(w,wmn,np.median(wmn),[w_l,w_u],z=0.0) ,3),  round(CF_lya*np.mean(ew),3)

    
    return ew_md_s,ew_md_exp,ew_mn_s,ew_mn_exp,ew_cmn_s,ew_cmn_exp,ew_gmn_s,ew_gmn_exp,ew_wmn_s,ew_wmn_exp

#########################

def sim1():
    CF_lya = 0.21
    N_sim = 1000
    v_sigma = 444.0

    w1,md1,mn1,wmn1,cmn1,gmn1,ew1,vcent1,sigew_md1,sigew_mn1,sigew_wmn1,sigew_cmn1,sigew_gmn1 = stack_main_simulation(N_sim, SNR_l = 0.5, SNR_u = 4.9, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md1_s,ew_md1_exp,ew_mn1_s,ew_mn1_exp,ew_cmn1_s,ew_cmn1_exp,ew_gmn1_s,ew_gmn1_exp,ew_wmn1_s,ew_wmn1_exp = get_EW_exp_stack(w1,md1,mn1,wmn1,cmn1,gmn1,ew1,CF_lya)
    
    w2,md2,mn2,wmn2,cmn2,gmn2,ew2,vcent2,sigew_md2,sigew_mn2,sigew_wmn2,sigew_cmn2,sigew_gmn2 = stack_main_simulation(N_sim, SNR_l = 5, SNR_u = 9.9, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md2_s,ew_md2_exp,ew_mn2_s,ew_mn2_exp,ew_cmn2_s,ew_cmn2_exp,ew_gmn2_s,ew_gmn2_exp,ew_wmn2_s,ew_wmn2_exp = get_EW_exp_stack(w2,md2,mn2,wmn2,cmn2,gmn2,ew2,CF_lya)
    
    w3,md3,mn3,wmn3,cmn3,gmn3,ew3,vcent3,sigew_md3,sigew_mn3,sigew_wmn3,sigew_cmn3,sigew_gmn3 = stack_main_simulation(N_sim, SNR_l = 10, SNR_u = 14.99, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md3_s,ew_md3_exp,ew_mn3_s,ew_mn3_exp,ew_cmn3_s,ew_cmn3_exp,ew_gmn3_s,ew_gmn3_exp,ew_wmn3_s,ew_wmn3_exp = get_EW_exp_stack(w3,md3,mn3,wmn3,cmn3,gmn3,ew3,CF_lya)
    
    w4,md4,mn4,wmn4,cmn4,gmn4,ew4,vcent4,sigew_md4,sigew_mn4,sigew_wmn4,sigew_cmn4,sigew_gmn4 = stack_main_simulation(N_sim, SNR_l = 15, SNR_u = 19.99, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md4_s,ew_md4_exp,ew_mn4_s,ew_mn4_exp,ew_cmn4_s,ew_cmn4_exp,ew_gmn4_s,ew_gmn4_exp,ew_wmn4_s,ew_wmn4_exp = get_EW_exp_stack(w4,md4,mn4,wmn4,cmn4,gmn4,ew4,CF_lya)
    
    w5,md5,mn5,wmn5,cmn5,gmn5,ew5,vcent5,sigew_md5,sigew_mn5,sigew_wmn5,sigew_cmn5,sigew_gmn5 = stack_main_simulation(N_sim, SNR_l = 20, SNR_u = 24.99, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md5_s,ew_md5_exp,ew_mn5_s,ew_mn5_exp,ew_cmn5_s,ew_cmn5_exp,ew_gmn5_s,ew_gmn5_exp,ew_wmn5_s,ew_wmn5_exp = get_EW_exp_stack(w5,md5,mn5,wmn5,cmn5,gmn5,ew5,CF_lya)



def plot1():

    CF_lya = 0.21
    N_sim = 715
    v_sigma = 436.0

    w1,md1,mn1,wmn1,cmn1,gmn1,ew1,vcent1,sigew_md1,sigew_mn1,sigew_wmn1,sigew_cmn1,sigew_gmn1 = stack_main_simulation(N_sim, SNR_l = 0.5, SNR_u = 4.9, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md1_s,ew_md1_exp,ew_mn1_s,ew_mn1_exp,ew_cmn1_s,ew_cmn1_exp,ew_gmn1_s,ew_gmn1_exp,ew_wmn1_s,ew_wmn1_exp = get_EW_exp_stack(w1,md1,mn1,wmn1,cmn1,gmn1,ew1,CF_lya)
    
    w2,md2,mn2,wmn2,cmn2,gmn2,ew2,vcent2,sigew_md2,sigew_mn2,sigew_wmn2,sigew_cmn2,sigew_gmn2 = stack_main_simulation(N_sim, SNR_l = 5, SNR_u = 9.9, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md2_s,ew_md2_exp,ew_mn2_s,ew_mn2_exp,ew_cmn2_s,ew_cmn2_exp,ew_gmn2_s,ew_gmn2_exp,ew_wmn2_s,ew_wmn2_exp = get_EW_exp_stack(w2,md2,mn2,wmn2,cmn2,gmn2,ew2,CF_lya)
    
    w3,md3,mn3,wmn3,cmn3,gmn3,ew3,vcent3,sigew_md3,sigew_mn3,sigew_wmn3,sigew_cmn3,sigew_gmn3 = stack_main_simulation(N_sim, SNR_l = 10, SNR_u = 14.99, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md3_s,ew_md3_exp,ew_mn3_s,ew_mn3_exp,ew_cmn3_s,ew_cmn3_exp,ew_gmn3_s,ew_gmn3_exp,ew_wmn3_s,ew_wmn3_exp = get_EW_exp_stack(w3,md3,mn3,wmn3,cmn3,gmn3,ew3,CF_lya)
    
    w4,md4,mn4,wmn4,cmn4,gmn4,ew4,vcent4,sigew_md4,sigew_mn4,sigew_wmn4,sigew_cmn4,sigew_gmn4 = stack_main_simulation(N_sim, SNR_l = 15, SNR_u = 19.99, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md4_s,ew_md4_exp,ew_mn4_s,ew_mn4_exp,ew_cmn4_s,ew_cmn4_exp,ew_gmn4_s,ew_gmn4_exp,ew_wmn4_s,ew_wmn4_exp = get_EW_exp_stack(w4,md4,mn4,wmn4,cmn4,gmn4,ew4,CF_lya)
    
    w5,md5,mn5,wmn5,cmn5,gmn5,ew5,vcent5,sigew_md5,sigew_mn5,sigew_wmn5,sigew_cmn5,sigew_gmn5 = stack_main_simulation(N_sim, SNR_l = 20, SNR_u = 24.99, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md5_s,ew_md5_exp,ew_mn5_s,ew_mn5_exp,ew_cmn5_s,ew_cmn5_exp,ew_gmn5_s,ew_gmn5_exp,ew_wmn5_s,ew_wmn5_exp = get_EW_exp_stack(w5,md5,mn5,wmn5,cmn5,gmn5,ew5,CF_lya)


    
    fig, axs = plt.subplots(2,2, sharex=True,figsize=(18,8))
    plt.subplot(1,2,1)

    plt.plot(w1,md1/np.median(md1),c='b', label='SNR_c1: '+str(int( 1.0/ np.std(md1)))+'; EW1: '+str(ew_md1_s)+'; EW1_exp: '+str(ew_md1_exp)+'; Diff %: '+str( int(100.0*abs(ew_md1_exp - ew_md1_s)/ ew_md1_exp)))
    plt.plot(w2,md2/np.median(md2),c='r', label='SNR_c2: '+str(int( 1.0/ np.std(md2)))+'; EW2: '+str(ew_md2_s)+'; EW2_exp: '+str(ew_md2_exp)+'; Diff %: '+str( int(100.0*abs(ew_md2_exp - ew_md2_s)/ ew_md2_exp)))
    plt.plot(w3,md3/np.median(md3),c='g', label='SNR_c3: '+str(int( 1.0/ np.std(md3)))+'; EW3: '+str(ew_md3_s)+'; EW3_exp: '+str(ew_md3_exp)+'; Diff %: '+str( int(100.0*abs(ew_md3_exp - ew_md3_s)/ ew_md3_exp)))
    plt.plot(w4,md4/np.median(md4),c='y', label='SNR_c4: '+str(int( 1.0/ np.std(md4)))+'; EW4: '+str(ew_md4_s)+'; EW4_exp: '+str(ew_md4_exp)+'; Diff %: '+str( int(100.0*abs(ew_md4_exp - ew_md4_s)/ ew_md4_exp)))
    plt.plot(w5,md5/np.median(md5),c='c', label='SNR_c5: '+str(int( 1.0/ np.std(md5)))+'; EW5: '+str(ew_md5_s)+'; EW5_exp: '+str(ew_md5_exp)+'; Diff %: '+str( int(100.0*abs(ew_md5_exp - ew_md5_s)/ ew_md5_exp)))
    
    
    plt.axhline(1.0,color='k',ls='-.')
    plt.axvline(w_l,color='k',ls='-.')
    plt.axvline(w_u,color='k',ls='-.')
    plt.title('Median')
    plt.legend(frameon=False,fontsize=10)
    
    plt.subplot(1,2,2)
    
    plt.plot(w1,mn1/np.median(mn1),c='b', label='SNR_c1: '+str(int( 1.0/ np.std(mn1)))+'; EW1: '+str(ew_mn1_s)+'; EW1_exp: '+str(ew_mn1_exp)+'; Diff %: '+str( int(100.0*abs(ew_mn1_exp - ew_mn1_s)/ ew_mn1_exp)))
    plt.plot(w2,mn2/np.median(mn2),c='r', label='SNR_c2: '+str(int( 1.0/ np.std(mn2)))+'; EW2: '+str(ew_mn2_s)+'; EW2_exp: '+str(ew_mn2_exp)+'; Diff %: '+str( int(100.0*abs(ew_mn2_exp - ew_mn2_s)/ ew_mn2_exp)))
    plt.plot(w3,mn3/np.median(mn3),c='g', label='SNR_c3: '+str(int( 1.0/ np.std(mn3)))+'; EW3: '+str(ew_mn3_s)+'; EW3_exp: '+str(ew_mn3_exp)+'; Diff %: '+str( int(100.0*abs(ew_mn3_exp - ew_mn3_s)/ ew_mn3_exp)))
    plt.plot(w4,mn4/np.median(mn4),c='y', label='SNR_c4: '+str(int( 1.0/ np.std(mn4)))+'; EW4: '+str(ew_mn4_s)+'; EW4_exp: '+str(ew_mn4_exp)+'; Diff %: '+str( int(100.0*abs(ew_mn4_exp - ew_mn4_s)/ ew_mn4_exp)))
    plt.plot(w5,mn5/np.median(mn5),c='c', label='SNR_c5: '+str(int( 1.0/ np.std(mn5)))+'; EW5: '+str(ew_mn5_s)+'; EW5_exp: '+str(ew_mn5_exp)+'; Diff %: '+str( int(100.0*abs(ew_mn5_exp - ew_mn5_s)/ ew_mn5_exp)))
    
    plt.axhline(1.0,color='k',ls='-.')
    plt.axvline(w_l,color='k',ls='-.')
    plt.axvline(w_u,color='k',ls='-.')
    plt.title('Mean')
    plt.legend(frameon=False,fontsize=10)


    # # plt.subplot(2,2,3)
    # # plt.plot(w1,cmn1/np.median(cmn1),c='b', label='SNR_c1: '+str(int( 1.0/ np.std(cmn1)))+'; EW1: '+str(ew_cmn1_s)+'; EW1_exp: '+str(ew_cmn1_exp)+'; Diff %: '+str( int(100.0*abs(ew_cmn1_exp - ew_cmn1_s)/ ew_cmn1_exp)))
    # # plt.plot(w2,cmn2/np.median(cmn2),c='r', label='SNR_c2: '+str(int( 1.0/ np.std(cmn2)))+'; EW2: '+str(ew_cmn2_s)+'; EW2_exp: '+str(ew_cmn2_exp)+'; Diff %: '+str( int(100.0*abs(ew_cmn2_exp - ew_cmn2_s)/ ew_cmn2_exp)))
    # # plt.plot(w3,cmn3/np.median(cmn3),c='g', label='SNR_c3: '+str(int( 1.0/ np.std(cmn3)))+'; EW3: '+str(ew_cmn3_s)+'; EW3_exp: '+str(ew_cmn3_exp)+'; Diff %: '+str( int(100.0*abs(ew_cmn3_exp - ew_cmn3_s)/ ew_cmn3_exp)))
    # # plt.plot(w4,cmn4/np.median(cmn4),c='y', label='SNR_c4: '+str(int( 1.0/ np.std(cmn4)))+'; EW4: '+str(ew_cmn4_s)+'; EW4_exp: '+str(ew_cmn4_exp)+'; Diff %: '+str( int(100.0*abs(ew_cmn4_exp - ew_cmn4_s)/ ew_cmn4_exp)))
    # # plt.plot(w5,cmn5/np.median(cmn5),c='c', label='SNR_c5: '+str(int( 1.0/ np.std(cmn5)))+'; EW5: '+str(ew_cmn5_s)+'; EW5_exp: '+str(ew_cmn5_exp)+'; Diff %: '+str( int(100.0*abs(ew_cmn5_exp - ew_cmn5_s)/ ew_cmn5_exp)))
    
    
    # # plt.axhline(1.0,color='k',ls='-.')
    # # plt.axvline(w_l,color='k',ls='-.')
    # # plt.axvline(w_u,color='k',ls='-.')
    # # plt.title('5-sigma clipped mean')
    # # plt.legend(frameon=False,fontsize=10)


    # # plt.subplot(2,2,4)
    # # plt.plot(w1,gmn1/np.median(gmn1),c='b', label='SNR_c1: '+str(int( 1.0/ np.std(gmn1)))+'; EW1: '+str(ew_gmn1_s)+'; EW1_exp: '+str(ew_gmn1_exp)+'; Diff %: '+str( int(100.0*abs(ew_gmn1_exp - ew_gmn1_s)/ ew_gmn1_exp)))
    # # plt.plot(w2,gmn2/np.median(gmn2),c='r', label='SNR_c2: '+str(int( 1.0/ np.std(gmn2)))+'; EW2: '+str(ew_gmn2_s)+'; EW2_exp: '+str(ew_gmn2_exp)+'; Diff %: '+str( int(100.0*abs(ew_gmn2_exp - ew_gmn2_s)/ ew_gmn2_exp)))
    # # plt.plot(w3,gmn3/np.median(gmn3),c='g', label='SNR_c3: '+str(int( 1.0/ np.std(gmn3)))+'; EW3: '+str(ew_gmn3_s)+'; EW3_exp: '+str(ew_gmn3_exp)+'; Diff %: '+str( int(100.0*abs(ew_gmn3_exp - ew_gmn3_s)/ ew_gmn3_exp)))
    # # plt.plot(w4,gmn4/np.median(gmn4),c='y', label='SNR_c4: '+str(int( 1.0/ np.std(gmn4)))+'; EW4: '+str(ew_gmn4_s)+'; EW4_exp: '+str(ew_gmn4_exp)+'; Diff %: '+str( int(100.0*abs(ew_gmn4_exp - ew_gmn4_s)/ ew_gmn4_exp)))
    # # plt.plot(w5,gmn5/np.median(gmn5),c='c', label='SNR_c5: '+str(int( 1.0/ np.std(gmn5)))+'; EW5: '+str(ew_gmn5_s)+'; EW5_exp: '+str(ew_gmn5_exp)+'; Diff %: '+str( int(100.0*abs(ew_gmn5_exp - ew_gmn5_s)/ ew_gmn5_exp)))
    
    
    # # plt.axhline(1.0,color='k',ls='-.')
    # # plt.axvline(w_l,color='k',ls='-.')
    # # plt.axvline(w_u,color='k',ls='-.')
    # # plt.title('GMEAN')
    # # plt.legend(frameon=False,fontsize=10)


def plot2_which_method():

    CF_lya = 0.21
    N_sim = 700
    v_sigma = 436.0


    w5,md5,mn5,wmn5,cmn5,gmn5,ew5,vcent5,sigew_md5,sigew_mn5,sigew_wmn5,sigew_cmn5,sigew_gmn5 = stack_main_simulation(N_sim, SNR_l = 20, SNR_u = 24.99, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md5_s,ew_md5_exp,ew_mn5_s,ew_mn5_exp,ew_cmn5_s,ew_cmn5_exp,ew_gmn5_s,ew_gmn5_exp,ew_wmn5_s,ew_wmn5_exp = get_EW_exp_stack(w5,md5,mn5,wmn5,cmn5,gmn5,ew5,CF_lya)

    print(ew_wmn5_s,ew_wmn5_exp)
    
    w1,md1,mn1,wmn1,cmn1,gmn1,ew1,vcent1,sigew_md1,sigew_mn1,sigew_wmn1,sigew_cmn1,sigew_gmn1 = stack_main_simulation(N_sim, SNR_l = 0.5, SNR_u = 4.9, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md1_s,ew_md1_exp,ew_mn1_s,ew_mn1_exp,ew_cmn1_s,ew_cmn1_exp,ew_gmn1_s,ew_gmn1_exp,ew_wmn1_s,ew_wmn1_exp = get_EW_exp_stack(w1,md1,mn1,wmn1,cmn1,gmn1,ew1,CF_lya)
    
    w2,md2,mn2,wmn2,cmn2,gmn2,ew2,vcent2,sigew_md2,sigew_mn2,sigew_wmn2,sigew_cmn2,sigew_gmn2 = stack_main_simulation(N_sim, SNR_l = 5, SNR_u = 9.9, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md2_s,ew_md2_exp,ew_mn2_s,ew_mn2_exp,ew_cmn2_s,ew_cmn2_exp,ew_gmn2_s,ew_gmn2_exp,ew_wmn2_s,ew_wmn2_exp = get_EW_exp_stack(w2,md2,mn2,wmn2,cmn2,gmn2,ew2,CF_lya)
    
    w3,md3,mn3,wmn3,cmn3,gmn3,ew3,vcent3,sigew_md3,sigew_mn3,sigew_wmn3,sigew_cmn3,sigew_gmn3 = stack_main_simulation(N_sim, SNR_l = 10, SNR_u = 14.99, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md3_s,ew_md3_exp,ew_mn3_s,ew_mn3_exp,ew_cmn3_s,ew_cmn3_exp,ew_gmn3_s,ew_gmn3_exp,ew_wmn3_s,ew_wmn3_exp = get_EW_exp_stack(w3,md3,mn3,wmn3,cmn3,gmn3,ew3,CF_lya)
    
    w4,md4,mn4,wmn4,cmn4,gmn4,ew4,vcent4,sigew_md4,sigew_mn4,sigew_wmn4,sigew_cmn4,sigew_gmn4 = stack_main_simulation(N_sim, SNR_l = 15, SNR_u = 19.99, CF_lya= CF_lya, v_sigma = v_sigma)
    ew_md4_s,ew_md4_exp,ew_mn4_s,ew_mn4_exp,ew_cmn4_s,ew_cmn4_exp,ew_gmn4_s,ew_gmn4_exp,ew_wmn4_s,ew_wmn4_exp = get_EW_exp_stack(w4,md4,mn4,wmn4,cmn4,gmn4,ew4,CF_lya)
    
    label_arr = ['0 <= SNR < 5', '5 <= SNR < 10', '10 <= SNR < 15', '15 <= SNR < 20', '20 <= SNR < 25']
    

    title_str = 'Median EW$_{input}$='+str(round(ew_md1_exp,3))+'; Mean EW$_{input}$='+str(round(ew_mn1_exp,3))


    fnt_sz = 16
    lwd= 3
    
    fig, axs = plt.subplots(2,2, sharey=True,sharex=True,figsize=(11,9))
    fig.subplots_adjust(left=0.1, bottom=0.07, right=0.95, top=0.97, wspace=0.02, hspace=0.16)


    fig.text(0.5, 0.025,  'Restframe wavelength ($\AA$)', va='center', ha='center',fontsize=fnt_sz+2)
    fig.text(0.03, 0.5, 'Normalized flux', va='center', ha='center', rotation='vertical',fontsize=fnt_sz+2)

        
    #axs1.suptitle(title_str,fontsize=20)
    axs1 = plt.subplot(2,2,1)

    axs1.plot(w1,md1/np.median(md1),c='b', label=label_arr[0]+'; EW1: '+str(ew_md1_s)+'$\pm$'+str(sigew_md1),alpha= 0.2)
    axs1.plot(w2,md2/np.median(md2),c='r', label=label_arr[1]+'; EW2: '+str(ew_md2_s)+'$\pm$'+str(sigew_md2),alpha= 0.5)
    axs1.plot(w3,md3/np.median(md3),c='g', label=label_arr[2]+'; EW3: '+str(ew_md3_s)+'$\pm$'+str(sigew_md3),alpha= 0.7)
    axs1.plot(w4,md4/np.median(md4),c='y', label=label_arr[3]+'; EW4: '+str(ew_md4_s)+'$\pm$'+str(sigew_md4),alpha= 0.9)
    axs1.plot(w5,md5/np.median(md5),c='c', label=label_arr[4]+'; EW5: '+str(ew_md5_s)+'$\pm$'+str(sigew_md5),alpha= 1.0)
    axs1.axhline(1.0,color='k',ls='-.')
    #axs1.axvline(w_l,color='k',ls='-.')
    #axs1.axvline(w_u,color='k',ls='-.')
    axs1.set_title('Median', fontsize=fnt_sz)
    axs1.legend(frameon=False,fontsize=12,loc='upper center')
    axs1.minorticks_on()
    axs1.tick_params(axis="both", labelsize=fnt_sz-2)
    axs1.tick_params(which='major', length=10, width=1.5, direction='inout')
    axs1.tick_params(which='minor', length=5, width=1.5, direction='in')
    axs1.tick_params(axis='both',which='major',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=12)
    axs1.tick_params(axis='both',which='minor',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=8)
    plt.setp(axs1.spines.values(), linewidth=3)
    

    
    axs2 = plt.subplot(2,2,2)
    axs2.plot(w1,mn1/np.median(mn1),c='b', label=label_arr[0]+'; EW1: '+str(ew_mn1_s)+'$\pm$'+str(sigew_mn1),alpha= 0.2)
    axs2.plot(w2,mn2/np.median(mn2),c='r', label=label_arr[1]+'; EW2: '+str(ew_mn2_s)+'$\pm$'+str(sigew_mn2),alpha= 0.5)
    axs2.plot(w3,mn3/np.median(mn3),c='g', label=label_arr[2]+'; EW3: '+str(ew_mn3_s)+'$\pm$'+str(sigew_mn3),alpha= 0.7)
    axs2.plot(w4,mn4/np.median(mn4),c='y', label=label_arr[3]+'; EW4: '+str(ew_mn4_s)+'$\pm$'+str(sigew_mn4),alpha= 0.9)
    axs2.plot(w5,mn5/np.median(mn5),c='c', label=label_arr[4]+'; EW5: '+str(ew_mn5_s)+'$\pm$'+str(sigew_mn5),alpha= 1.0)
    axs2.axhline(1.0,color='k',ls='-.')
    #axs2.axvline(w_l,color='k',ls='-.')
    #axs2.axvline(w_u,color='k',ls='-.')
    axs2.set_title('Mean', fontsize=fnt_sz)
    axs2.legend(frameon=False,fontsize=12,loc='upper center')
    axs2.minorticks_on()
    axs2.tick_params(axis="both", labelsize=fnt_sz-2)
    axs2.tick_params(which='major', length=10, width=1.5, direction='inout')
    axs2.tick_params(which='minor', length=5, width=1.5, direction='in')
    axs2.tick_params(axis='both',which='major',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=12)
    axs2.tick_params(axis='both',which='minor',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=8)
    plt.setp(axs2.spines.values(), linewidth=3)


    axs3 = plt.subplot(2,2,3)
    axs3.plot(w1,wmn1/np.median(wmn1),c='b', label=label_arr[0]+'; EW1: '+str(ew_wmn1_s)+'$\pm$'+str(sigew_wmn1),alpha= 0.2)
    axs3.plot(w2,wmn2/np.median(wmn2),c='r', label=label_arr[1]+'; EW2: '+str(ew_wmn2_s)+'$\pm$'+str(sigew_wmn2),alpha= 0.5)
    axs3.plot(w3,wmn3/np.median(wmn3),c='g', label=label_arr[2]+'; EW3: '+str(ew_wmn3_s)+'$\pm$'+str(sigew_wmn3),alpha= 0.7)
    axs3.plot(w4,wmn4/np.median(wmn4),c='y', label=label_arr[3]+'; EW4: '+str(ew_wmn4_s)+'$\pm$'+str(sigew_wmn4),alpha= 0.9)
    axs3.plot(w5,wmn5/np.median(wmn5),c='c', label=label_arr[4]+'; EW5: '+str(ew_wmn5_s)+'$\pm$'+str(sigew_wmn5),alpha= 1.0)
    axs3.axhline(1.0,color='k',ls='-.')
    #axs3.axvline(w_l,color='k',ls='-.')
    #axs3.axvline(w_u,color='k',ls='-.')
    axs3.set_title('SNR Weighted-MEAN', fontsize=fnt_sz)
    axs3.legend(frameon=False,fontsize=12,loc='upper center')
    axs3.minorticks_on()
    axs3.tick_params(axis="both", labelsize=fnt_sz-2)
    axs3.tick_params(which='major', length=10, width=1.5, direction='inout')
    axs3.tick_params(which='minor', length=5, width=1.5, direction='in')
    axs3.tick_params(axis='both',which='major',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=12)
    axs3.tick_params(axis='both',which='minor',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=8)
    plt.setp(axs3.spines.values(), linewidth=3)
    

    axs4 = plt.subplot(2,2,4)
    axs4.plot(w1,cmn1/np.median(cmn1),c='b', label=label_arr[0]+'; EW1: '+str(ew_cmn1_s)+'$\pm$'+str(sigew_cmn1),alpha= 0.2)
    axs4.plot(w2,cmn2/np.median(cmn2),c='r', label=label_arr[1]+'; EW2: '+str(ew_cmn2_s)+'$\pm$'+str(sigew_cmn2),alpha= 0.5)
    axs4.plot(w3,cmn3/np.median(cmn3),c='g', label=label_arr[2]+'; EW3: '+str(ew_cmn3_s)+'$\pm$'+str(sigew_cmn3),alpha= 0.7)
    axs4.plot(w4,cmn4/np.median(cmn4),c='y', label=label_arr[3]+'; EW4: '+str(ew_cmn4_s)+'$\pm$'+str(sigew_cmn4),alpha= 0.9)
    axs4.plot(w5,cmn5/np.median(cmn5),c='c', label=label_arr[4]+'; EW5: '+str(ew_cmn5_s)+'$\pm$'+str(sigew_cmn5),alpha= 1.0)
    axs4.axhline(1.0,color='k',ls='-.')
    #axs4.axvline(w_l,color='k',ls='-.')
    #axs4.axvline(w_u,color='k',ls='-.')
    axs4.set_title('5-sigma clipped mean', fontsize=fnt_sz)
    axs4.legend(frameon=False,fontsize=12,loc='upper center')
    axs4.minorticks_on()
    axs4.tick_params(axis="both", labelsize=fnt_sz-2)
    axs4.tick_params(which='major', length=10, width=1.5, direction='inout')
    axs4.tick_params(which='minor', length=5, width=1.5, direction='in')
    axs4.tick_params(axis='both',which='major',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=12)
    axs4.tick_params(axis='both',which='minor',direction='in', bottom=True, top=True, left=True, right=True,width=lwd,length=8)
    plt.setp(axs4.spines.values(), linewidth=3)

    ############################
    plt.savefig('simulated_SNR_effect_with_BS.png')
    plt.show()



    
##############################



def simulate_single(df1,  seed_spec=0, flg_lya_abs=0, SNR = 0, v_sigma = 0.0 ):

    #df5 = pd.read_csv('942_all_absorbers_catalog.txt', delim_whitespace=1)
    #idx5 = (df5.sigma_l > 3.0) & ( df5.vcent_lya > -500.0) & ( df5.vcent_lya < 500.0); df5 = df5[idx5]

    velocity = np.arange(-6000, 6000, 2.45)  # Array of velocities in km/s
    wave = (velocity/299792.46 + 1.0) * w_ion

    n_pix = np.median(np.diff(wave))
    
    
    continumm_level = 1.0
    flux = np.zeros(len(wave)) + continumm_level


    sigma_noise = continumm_level/SNR;  noise_std = np.random.normal(0.0,sigma_noise, len(wave))
    
    if flg_lya_abs == 1:

        random.seed(seed_spec+1); id_dan = random.randint(0, len(df1)-1)
        
        #np.random.seed(seed_spec+1); b_rnd = np.random.choice(df1['b'])
        #np.random.seed(seed_spec+2); logN_rnd = np.random.choice(df1['logN'])

        #np.random.seed(seed_spec+3); b_rnd = np.random.normal(30,5,1)[0]

        #print(b_rnd)
        b_rnd = df1['b'].iloc[id_dan]

        
        logN_rnd = df1['logN'].iloc[id_dan]

        ew_lya = 10**logN_rnd * ( (w_ion**2* f_ion)/1.13e20)
        sigma_lya = ( (np.sqrt( 2 * np.log(2) ) * b_rnd ) * 1450.0 ) / 299792.46   ###

        np.random.seed(seed_spec+3); w_loc = np.random.normal(1215.6701, 1215.6701*v_sigma/299792.46)
        
        
        #ew_lya = df5.ew_lya.iloc[id_dan]
        #width =  abs( df5.v2_lya.iloc[id_dan] - df5.v1_lya.iloc[id_dan])
        #sigma_lya =  (width*w_ion ) / 299792.46   ###1450.0
    
        ht_lya =   ew_lya / ( (sigma_lya) * np.sqrt(2.0 * math.pi) )

        flux = flux - (ht_lya)* np.exp( -0.5*( ( wave - w_loc ) / sigma_lya  )**2 ) 

        f_lya = lambda x, ht1, w0, b: 1.0 - ht1 * np.exp(-0.5*( ( x - w0 )/b )**2)      #fa*(x/2.0)**a
        line_lya = lambda x, ht1, w0, b: ht1 * np.exp(-0.5*( ( x - w0 )/b )**2 )

        start = (ht_lya, w_loc, sigma_lya)
        popt, pcov = curve_fit(f_lya, wave,flux , sigma = noise_std, p0 = start,absolute_sigma=True)  ##; perr = np.sqrt(np.diag(pcov)) #,  
        print(w_loc,ht_lya,sigma_lya,'Given EW:',ew_lya , 'Observed EW:', 'Observed line-fit EW: ', quad(line_lya, 1210,1220, args=( popt[0], popt[1], popt[2]))[0] )
        ##calcew(wave,flux,flux*0.0 + 1.0, [1212, 1217],z=0.0),

        N_dan, b_dan = logN_rnd , b_rnd
    else:

        N_dan, b_dan, ew_lya,w_loc = -1, -1, -1, 99999

    ############ convolution with COS =============
    delta_lambda = 1450.0 * 17.7 / 299792.46  ### resolution of COS  ###2 * np.sqrt(2 * np.log(2)) 
    convolved_flux = gaussian_filter1d(flux, delta_lambda/n_pix)


    ##### add noise to the data ==============

    
    flux_n = convolved_flux + noise_std

    ############ resample the data ================
    
    regrid = np.arange(wave[0], wave[-1], (wave[-1] - wave[0])/int(np.size(wave)/3.0))
    spec_resample, spec_errs_resample = spectres(regrid, wave, flux_n, spec_errs=abs(noise_std),verbose=False)
    wave_res,flux_res,eflux_res = regrid,spec_resample, spec_errs_resample

    #plt.step(wave,flux,color='b')
    #plt.step(wave,convolved_flux,color='r')
    #plt.step(wave1,flux1,color='g')
    #plt.show()

    return wave_res,flux_res, N_dan, b_dan, ew_lya,w_loc 
    

def simulate_realistic():

    C14 = 25    ## 1
    beta = 1.65 ##0.01

    N1_l , N1_u = 12.5, 13
    
    dN_dz = 260.0*0.2  ###C14 * (10**N1_l / 10**14)**(-(beta-1))  - C14 * (10**N1_u/10**14)**(-(beta-1))   #    #### for 12.6 - 12.8    #

    df1 = pd.read_csv('danforth_lya_system.txt', delim_whitespace=1)
    idx1 = ( df1.logN > N1_l)  & ( df1.logN < N1_u); df1 = df1[idx1]


    plt.hist(df1['logN'])
    plt.show()
    
    ########################
    df2 = pd.read_csv('lya_flg_with_SNR_942.txt', delim_whitespace=1)
    delz_tot = sum(df2.del_z.values)

    N_sim = len(df2)
    
    ew_inj, b_inj, N_inj, w_inj = np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim), np.zeros(N_sim)
    wav_all,flux_all,SNR_all = [],[],[]
    ##########
    detection_rate_lya =  int( dN_dz * delz_tot) ###int( dN_dz * np.median(df2.del_z.values) * N_sim)     ###

    id_all = np.arange(0,N_sim,1)
    np.random.seed(111); rnd_lya_id = np.random.choice(id_all,replace=False,size= detection_rate_lya)  ###replase=False no repeated entries
    Lya_flg = np. zeros((N_sim), dtype=int)
    Lya_flg[rnd_lya_id] = 1


    # #print(np.median(df2.z_cl))
    # print(len(df1),delz_tot,dN_dz, detection_rate_lya, np.sum(Lya_flg))
    
    # ########################
    # for ii in range(N_sim):
    # #for ii in range(0,100):

    #     #random.seed(2222+ii); SNR1 = df2.SNR_lya.iloc[ii]   #uniform(1,30)  ### 

    #     print(ii)
        
    #     wav,flux,N_tmp,b_tmp,ew_tmp,w_tmp = simulate_single(df1, seed_spec=ii, flg_lya_abs=Lya_flg[ii], SNR = df2.SNR_lya.iloc[ii], v_sigma = 500)

    #     N_inj[ii],b_inj[ii],ew_inj[ii],w_inj[ii] = N_tmp,b_tmp,ew_tmp,w_tmp 

    #     wav_all.append(wav)
    #     flux_all.append(flux)
    #     SNR_all.append( np.zeros(len(wav)) + df2.SNR_lya.iloc[ii])

    # wave_all,flux_all,SNR_all = np.ravel(wav_all),np.ravel(flux_all), np.ravel(SNR_all)
    # wave,md,mn,wmn,cmn,gmn = do_binning(wave_all,flux_all,SNR_all*0.0,SNR_all,m_wav)

    # ew = calcew(wave,wmn,np.median(wmn),[1212, 1217],z=0.0)

    # plt.plot(wave,wmn,color='b')
    # plt.axhline(1.0,color='k')
    # plt.savefig('simulation_effect.pdf')
    # plt.show()
    
    #print(np.std(  299792.46*(w_inj[w_inj < 99999]/w_ion - 1.0)) )

    #print(b_inj)
    #plt.hist(N_inj,bins=40)
    #plt.show()
    
    
    #print(delz_tot)
    
    #logN = 12
    #print( (10**logN) * ((w_ion**2* f_ion)/1.13e20))
    #plt.hist(df1.b, bins=50)
    #plt.show()
#######################



######################################################

f_lya = lambda x, ht1, w0, b: 1.0 - ht1 * np.exp(-0.5*( ( x - w0 )/b )**2)      #fa*(x/2.0)**a
vel_tol = 500.0
w_l,w_u = 1215.6701 - 1215.6701*vel_tol/299792.46, 1215.6701 + 1215.6701*vel_tol/299792.46

wmin1,wmax1,reso1  = 1200.0,1230,0.5
m_wav = np.arange(wmin1,wmax1,reso1)

w_ion, f_ion = 1215.6701, 0.4165

###################

plot2_which_method()   
#simulate_realistic()
