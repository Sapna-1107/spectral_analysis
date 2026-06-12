import glob, os, fnmatch, timeit, subprocess, time
import numpy as np
from matplotlib import pylab as plt
from astropy.io import fits
from scipy.interpolate import interp1d
import pandas as pd
from astropy.table import Table

AND = np.logical_and



def get_spec(infile):
    #data, header = fits.getdata(infile, ext=0,header=True)
    #hdul = fits.open(infile)
    #cols = hdul[1].columns
    #print(cols)
    # w   = data.field('WAVE')
    # f   = data.field('FLUX')
    # f_e = data.field('ERROR')
    # f_e_u = data.field('FLUXERR_UP')
    # f_e_d = data.field('FLUXERR_DOWN')
    # snr = data.field('SN')

    w, f, f_e = np.loadtxt(infile , unpack=True,usecols=[0,1,2],skiprows=1)
    snr = f / f_e
    
    return w,f,f_e,snr
############

def get_wdivide(w1,f1,snr1,w2,f2,snr2):
    
    w_low, w_up = w2[w2 < w1[len(w1)-1]][0], w1[w1 > w2[0]][len(w1[w1 > w2[0]])-1]
    id1 = np.where( AND(w1 > w_low, w1 < w_up))
    id2 = np.where( AND(w2 > w_low, w2 < w_up))
    
    snr_int2 = interp1d(w2[id2], snr2[id2], kind='cubic',fill_value="extrapolate")
    
    f = snr1[id1]
    g = snr_int2(w1[id1])
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()

    if np.size(idx) > 0:
        idc_med = int(np.median(idx))
        w_divide =  w1[id1][idc_med]

    else:
        snr_comm_arr =  np.array([np.median(snr1[id1]), np.median(snr2[id2])])
        w_arr_comm   =  np.array([w1[-1], w2[1]])
        id_sn_max    =  np.argmax(snr_comm_arr)

        w_divide     =  w_arr_comm[id_sn_max]

        


    #plt.axvline(x=w_low)
    #plt.axvline(x=w_up)


      
    return w_divide

            

#pwd = os.getcwd()
#dirfile = 'dir_file.txt'
#dirname = np.loadtxt(dirfile,usecols=[0],dtype='str',skiprows=9)

df_file = pd.read_csv('notes_obj_half.txt', delim_whitespace=1)
#print(df_file)

dirname = df_file.dirname.values
grism_flg = df_file.grism_flg.values


#for ii in range(len(dirname)):
for ii in range(0,1):

    tag_search = dirname[ii]+'/'+dirname[ii]+'*_spec-*'

    print( grism_flg[ii])
    
    if grism_flg[ii] == 1:

        print(dirname[ii])

        file1, file2, file3, file4, file5  = glob.glob(tag_search+'G160M'), glob.glob(tag_search+'G130M') , glob.glob(tag_search+'LIF2'), glob.glob(tag_search+'LIF1'), glob.glob(tag_search+'SIC2A')
        
        w1,f1,e1,snr1 = get_spec(file1[0])        
        w2,f2,e2,snr2 = get_spec(file2[0])
        w3,f3,e3,snr3 = get_spec(file3[0])
        w4,f4,e4,snr4 = get_spec(file4[0])
        w5,f5,e5,snr5 = get_spec(file5[0])

        w1,f1,e1,snr1=w1[ e1 > 0.0],f1[e1> 0.0],e1[e1 > 0.0],snr1[ e1 > 0.0]
        w2,f2,e2,snr2=w2[ e2 > 0.0],f2[e2> 0.0],e2[e2 > 0.0],snr2[ e2 > 0.0]
        w3,f3,e3,snr3=w3[ e3 > 0.0],f3[e3> 0.0],e3[e3 > 0.0],snr3[ e3 > 0.0]
        w4,f4,e4,snr4=w4[ e4 > 0.0],f4[e4> 0.0],e4[e4 > 0.0],snr4[ e4 > 0.0]
        w5,f5,e5,snr5=w5[ e5 > 0.0],f5[e5> 0.0],e5[e5 > 0.0],snr5[ e5 > 0.0]


        fig = plt.figure(figsize=(20,8))
        

        plt.plot(w1,f1,c='y',label='G160; SN = '+str(round(np.median(snr1),2)))
        plt.plot(w2,f2,c='b',label='G130; SN = '+str(round(np.median(snr2),2)))
        plt.plot(w3,f3,c='g',label='LIF2; SN = '+str(round(np.median(snr3),2)))
        plt.plot(w4,f4,c='r',label='LIF1; SN = '+str(round(np.median(snr4),2)))
        plt.plot(w5,f5,c='magenta',label='SIC2A; SN = '+str(round(np.median(snr5),2)))

        plt.legend()
        
        w_divide1 = get_wdivide(w2,f2,snr2,w1,f1,snr1)   ## between G130 and G160
        w_divide2 = get_wdivide(w3,f3,snr3,w2,f2,snr2)   ## between G130 and LIF2
        w_divide3 = get_wdivide(w5,f5,snr5,w4,f4,snr4)   ## LIF1 SIC2

        plt.title(dirname[ii]+';  W_G130_G160 = '+str(w_divide1)+';  W_G130_LIF2 = '+str(w_divide2)+';  W_LIF1_SIC2A = '+str(w_divide3),fontsize=16)
        ############
        reg1, reg2, reg3, reg4, reg5 = (w1 > w_divide1), (w2 >= w_divide2) & (w2 <= w_divide1), (w3 < w_divide2), (w4 >= w_divide3), (w5 < w_divide3)
        w_final = np.concatenate((w5[reg5] , w4[reg4], w3[reg3], w2[reg2], w1[reg1]), axis = 0)
        f_final = np.concatenate((f5[reg5] , f4[reg4], f3[reg3], f2[reg2], f1[reg1]), axis = 0)
        e_final = np.concatenate((e5[reg5] , e4[reg4], e3[reg3], e2[reg2], e1[reg1]), axis = 0)

        # outfile = dirname[ii]+'/'+dirname[ii]+'_spec_full_combined.fits'

        # tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='WAVE', format='E',array=w_final), fits.Column(name='FLUX', format='E',array=f_final), \
        #         fits.Column(name='ERROR', format='E',array=e_final), \
        #         fits.Column(name='w_G130_G160', format='E',array=[w_divide1]), fits.Column(name='w_LIF2_G130', format='E',array=[w_divide2]),\
        #         fits.Column(name='w_SIC2_LIF1', format='E',array=[w_divide3]) ])

        # tbhdu.writeto(outfile,overwrite=True)

         ###############
        plt.axvline(x=w_divide1,lw=2,color='black',ls='--')
        plt.axvline(x=w_divide2,lw=2,color='black',ls='--')
        plt.axvline(x=w_divide3,lw=2,color='black',ls='--')
        plt.show()
    ############

    elif grism_flg[ii] == 2:

        print(dirname[ii])

        file1, file2, file3, file4  = glob.glob(tag_search+'G160M'), glob.glob(tag_search+'G130M') , glob.glob(tag_search+'LIF1'), glob.glob(tag_search+'SIC2A')
        
        w1,f1,e1,snr1 = get_spec(file1[0])        
        w2,f2,e2,snr2 = get_spec(file2[0])
        w3,f3,e3,snr3 = get_spec(file3[0])
        w4,f4,e4,snr4 = get_spec(file4[0])

        w1,f1,e1,snr1=w1[ e1 > 0.0],f1[e1> 0.0],e1[e1 > 0.0],snr1[ e1 > 0.0]
        w2,f2,e2,snr2=w2[ e2 > 0.0],f2[e2> 0.0],e2[e2 > 0.0],snr2[ e2 > 0.0]
        w3,f3,e3,snr3=w3[ e3 > 0.0],f3[e3> 0.0],e3[e3 > 0.0],snr3[ e3 > 0.0]
        w4,f4,e4,snr4=w4[ e4 > 0.0],f4[e4> 0.0],e4[e4 > 0.0],snr4[ e4 > 0.0]


        fig = plt.figure(figsize=(20,8))
        plt.plot(w1,f1,c='y',label='G160; SN  = '+str(round(np.median(snr1),2)))
        plt.plot(w2,f2,c='b',label='G130; SN  = '+str(round(np.median(snr2),2)))
        plt.plot(w3,f3,c='r',label='LIF1; SN  = '+str(round(np.median(snr3),2)))
        plt.plot(w4,f4,c='magenta',label='SIC2A; SN = '+str(round(np.median(snr4),2)))
        plt.legend()
        
        w_divide1 = get_wdivide(w2,f2,snr2,w1,f1,snr1)   ## between G130 and G160
        w_divide3 = get_wdivide(w4,f4,snr4,w3,f3,snr3)   ## LIF1 SIC2

        plt.title(dirname[ii]+';  W_G130_G160 = '+str(w_divide1)+';  W_LIF1_SIC2A = '+str(w_divide3),fontsize=16)
        ############
        reg1, reg2, reg3, reg4, reg5 = (w1 > w_divide1), (w2 <= w_divide1), (w3 >= w_divide3), (w4 < w_divide3)
        w_final = np.concatenate((w4[reg4], w3[reg3], w2[reg2], w1[reg1]), axis = 0)
        f_final = np.concatenate((f4[reg4], f3[reg3], f2[reg2], f1[reg1]), axis = 0)
        e_final = np.concatenate((e4[reg4], e3[reg3], e2[reg2], e1[reg1]), axis = 0)

        # outfile = dirname[ii]+'/'+dirname[ii]+'_spec_full_combined.fits'

        # tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='WAVE', format='E',array=w_final), fits.Column(name='FLUX', format='E',array=f_final), \
        #         fits.Column(name='ERROR', format='E',array=e_final), \
        #         fits.Column(name='w_G130_G160', format='E',array=[w_divide1]), fits.Column(name='w_SIC2_LIF1', format='E',array=[w_divide3]) ])

        # tbhdu.writeto(outfile,overwrite=True)

        ###############
        plt.axvline(x=w_divide1,lw=2,color='black',ls='--')
        plt.axvline(x=w_divide3,lw=2,color='black',ls='--')
        plt.show()
    ############
    
    elif grism_flg[ii] == 3:
        print(dirname[ii])

        file1, file2, file3  = glob.glob(tag_search+'G160M'), glob.glob(tag_search+'G130M'), glob.glob(tag_search+'LIF1')
        
        w1,f1,e1,snr1 = get_spec(file1[0])        
        w2,f2,e2,snr2 = get_spec(file2[0])
        w3,f3,e3,snr3 = get_spec(file3[0])
        
        w1,f1,e1,snr1=w1[ e1 > 0.0],f1[e1> 0.0],e1[e1 > 0.0],snr1[ e1 > 0.0]
        w2,f2,e2,snr2=w2[ e2 > 0.0],f2[e2> 0.0],e2[e2 > 0.0],snr2[ e2 > 0.0]
        w3,f3,e3,snr3=w3[ e3 > 0.0],f3[e3> 0.0],e3[e3 > 0.0],snr3[ e3 > 0.0]

        fig = plt.figure(figsize=(20,8))
        plt.plot(w1,f1,c='y',label='G160; SN = '+str(round(np.median(snr1),2)))
        plt.plot(w2,f2,c='b',label='G130; SN = '+str(round(np.median(snr2),2)))
        plt.plot(w3,f3,c='r',label='LIF1; SN = '+str(round(np.median(snr3),2)))
        plt.legend()
        
        w_divide1 = get_wdivide(w2,f2,snr2,w1,f1,snr1)   ## between G130 and G160

        plt.title(dirname[ii]+';  W_G130_G160 = '+str(w_divide1),fontsize=16)
        
        ############
        reg1, reg2, reg3  = (w1 > w_divide1), (w2 <= w_divide1), (w3 < w3[-1])
        w_final = np.concatenate((w3[reg3], w2[reg2], w1[reg1]), axis = 0)
        f_final = np.concatenate((f3[reg3], f2[reg2], f1[reg1]), axis = 0)
        e_final = np.concatenate((e3[reg3], e2[reg2], e1[reg1]), axis = 0)

        outfile = dirname[ii]+'/'+dirname[ii]+'_spec_full_combined.fits'

        # tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='WAVE', format='E',array=w_final), fits.Column(name='FLUX', format='E',array=f_final), \
        #         fits.Column(name='ERROR', format='E',array=e_final), \
        #         fits.Column(name='w_G130_G160', format='E',array=[w_divide1]) ])
        # tbhdu.writeto(outfile,overwrite=True)
        ###############
        plt.axvline(x=w_divide1,lw=2,color='black',ls='--')
        plt.show()
    ############
    
    elif grism_flg[ii] == 4:

        print(tag_search)
        print(dirname[ii])
        
        file1, file2  = glob.glob(tag_search+'G160M'), glob.glob(tag_search+'G130M') 
        
        w1,f1,e1,snr1 = get_spec(file1[0])        
        w2,f2,e2,snr2 = get_spec(file2[0])

        w1,f1,e1,snr1=w1[ e1 > 0.0],f1[e1> 0.0],e1[e1 > 0.0],snr1[ e1 > 0.0]
        w2,f2,e2,snr2=w2[ e2 > 0.0],f2[e2> 0.0],e2[e2 > 0.0],snr2[ e2 > 0.0]
        w_divide1 = get_wdivide(w2,f2,snr2,w1,f1,snr1)   ## between G130 and G160

        ############
        reg1, reg2 = (w1 > w_divide1), (w2 <= w_divide1)
        w_final = np.concatenate( (w2[reg2], w1[reg1]), axis = 0)
        f_final = np.concatenate( (f2[reg2], f1[reg1]), axis = 0)
        e_final = np.concatenate( (e2[reg2], e1[reg1]), axis = 0)

        outfile = dirname[ii]+'/'+dirname[ii]+'_spec_full_combined.fits'

        # tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='WAVE', format='E',array=w_final), fits.Column(name='FLUX', format='E',array=f_final), \
        #         fits.Column(name='ERROR', format='E',array=e_final), fits.Column(name='w_G130_G160', format='E',array=[w_divide1]) ])
        # tbhdu.writeto(outfile,overwrite=True)

        # df2 = pd.DataFrame(list(zip(w_final,f_final,e_final,w_divide1)), columns =['wave','flux','error', 'w_G130_G160'])
        # t = Table.from_pandas(df2)
        # t.write(outfile,overwrite=True)
    
        #####================
        fig = plt.figure(figsize=(20,8))

        plt.title(dirname[ii]+';  W_G130_G160 = '+str(w_divide1),fontsize=16)
        plt.plot(w1,f1,c='y',label='G160; SN = '+str(round(np.median(snr1),2)))
        plt.plot(w2,f2,c='b',label='G130; SN = '+str(round(np.median(snr2),2)))
        plt.axvline(x=w_divide1,lw=2,color='black',ls='--')

        print(w_divide1)
        plt.legend()
        plt.show()
    
    else:
        continue

    
    #plt.step(w_final,f_final,c='grey',lw=3,label='Full', alpha=0.4)

    
   
