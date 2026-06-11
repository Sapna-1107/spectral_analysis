;;;;===============
function mask_emission,wav_t,flux_t,sig_t,conti_t,zem
  id_em_final = [0.0]
  emission_line = (1.0 + zem)*[1215.6701,1398.266545,1549.49265,1908.734]  
   ;; ;;================for sky line================
    for jj = 0, n_elements(emission_line)-1 do begin
       id_em = where( wav_t ge (emission_line(jj) - 20.0) and  wav_t le (emission_line(jj) + 20.0), n_em)  ;;;inspired from Srinivasan
       if n_em gt 0 then id_em_final = [id_em_final,id_em]
    endfor

    if n_elements(id_em_final) gt 1 then begin
       id_em_final = id_em_final[1:*]
       remove,id_em_final,wav_t
       remove,id_em_final,flux_t
       remove,id_em_final,sig_t
       remove,id_em_final,conti_t
    endif

    masked = {w: wav_t, f: flux_t, e: sig_t,c: conti_t }
  return, masked
end
;;;==================

function mask_bad_reg,wav_t,flux_t,sig_t,conti_t, file_qso_old_t
  bb =  mrdfits(file_qso_old_t,1,h,/silent)

  w_ot = bb.wave & f_ot = bb.flux & e_ot = bb.error
  ;wplot,w_ot,f_ot,psym=2
  
  id_e0 = where( e_ot eq 0, n_e0)

  if n_e0 gt 3 then begin
     dim1 = n_elements(id_e0)-1
     dim2 = n_elements(id_e0)-2

     idd_jump = id_e0[1:dim1] - id_e0[0:dim2]

     id_gap = where( idd_jump gt 1, n_gap)

     if n_gap ge 1 then begin
        l_wave_jump =[id_e0(0), id_e0(id_gap+1)]   
        u_wave_jump =[id_e0(id_gap), id_e0(dim1)]
     endif else begin
        l_wave_jump = id_e0(0)  
        u_wave_jump = id_e0(dim1) 
     endelse

  endif

  
  if n_elements(l_wave_jump) gt 0 then begin

  ;mask_reg_low =  [ 1198, 1208, 1300,[w_ot(l_wave_jump)]]
  ;mask_reg_up  =  [ 1202, 1224, 1308,[w_ot(u_wave_jump)]]
  
   mask_reg_low =  [1133.8, 1134.6, 1144.6, 1152.4, 1190.0, 1192.9, 1198, 1206.1, 1208, 1250.2, 1253.4, 1259.1, 1260,   1300, 1334.1, 1335.2, 1393.3, 1402.3, 1526.2, 1547.7, 1550.3, 1607.9, 1670.2, [w_ot(l_wave_jump)]]
   mask_reg_up  =  [1134.5, 1135.4, 1145.3, 1153.2, 1190.8, 1193.7, 1202, 1206.9, 1224, 1251,   1254.2, 1259.9, 1260.8, 1308, 1335,   1336.1, 1394.2, 1403.2, 1527.2, 1548.7, 1551.3, 1609, 1671.3,   [w_ot(u_wave_jump)]]

  endif else begin

   ;mask_reg_low =  [ 1198, 1208, 1300]
   ;mask_reg_up  =  [ 1202, 1224, 1308]
   
   mask_reg_low =  [ 1133.8, 1134.6, 1144.6, 1152.4, 1190.0, 1192.9, 1198, 1206.1, 1208, 1250.2, 1253.4, 1259.1, 1260,   1300, 1334.1, 1335.2, 1393.3, 1402.3, 1526.2, 1547.7, 1550.3, 1607.9, 1670.2]
   mask_reg_up  =  [ 1134.5, 1135.4, 1145.3, 1153.2, 1190.8, 1193.7, 1202, 1206.9, 1224, 1251,   1254.2, 1259.9, 1260.8, 1308, 1335,   1336.1, 1394.2, 1403.2, 1527.2, 1548.7, 1551.3, 1609, 1671.3]
   
  endelse

  id_bad_reg = [0]
   ;; ;;================for sky line================
    for jj = 0, n_elements(mask_reg_low)-1 do begin
       id_bad = where( wav_t ge mask_reg_low(jj) and  wav_t le mask_reg_up(jj), n_bad)
       if n_bad gt 0 then id_bad_reg = [id_bad_reg, id_bad]
    endfor
    
    if n_elements(id_bad_reg) gt 1 then begin
       id_bad_reg = id_bad_reg[1:*]
       remove,id_bad_reg,wav_t
       remove,id_bad_reg,flux_t
       remove,id_bad_reg,sig_t
       remove,id_bad_reg,conti_t
    endif
    masked = {w: wav_t, f: flux_t, e: sig_t,c: conti_t }
  return, masked
end


;;;;;;;;;;============================= efficient way =====================
function get_matched_ids_2, x_all,x_rnd,bootstrapid=bootstrp_num
 aa= x_all  
 bb= x_rnd  
 naa = n_elements(aa)
 nbb = n_elements(bb)
 ;;===================
 ;hist_a=histogram(aa,/l64,min=0, max=max(bb)+1, rev=RI)
 hist_a=histogram(aa,/l64,min=0, max=max(bb)+1, rev=RI)
 res=lonarr(total(hist_a[bb], /int), /nozero)

 pos=0l
 for j=0l,nbb-1l do begin
    print,bootstrp_num,j
    value=bb[j]
    count=hist_a[value]
    if count gt 0 then begin
       res[pos:pos+count-1]= RI[RI[value] : RI[value+1]-1]
       pos += count
    endif
 endfor
  return,res
end

;;;========================================================

function do_binning,wav_big,flux_big,sig_big,id_big,siglevel=sig_tmp
  Common CS1, ion_name,ion_wav_c,wmin,wmax, reso,SNR_req,vel_tol,del_v

  SNR_big = sig_big   ;;;;;flux_big/sig_big
  
  n_arr=  fix((wmax - wmin )/reso)
  m_wav = wmin + (wmax - wmin )*findgen(n_arr)/n_arr
  N1 = n_elements(m_wav)
  
  mean_arr        = fltarr(N1) - 9999.9999
  med_arr         = fltarr(N1) - 9999.9999
  N_q_arr         = fltarr(N1) - 9999.9999
  percentile      = fltarr(2,N1) - 9999.9999
  N_arr           = fltarr(N1) - 9999.9999
  std_arr         = fltarr(N1) - 9999.9999
  err_mean_arr    = fltarr(N1) - 9999.9999
  gmean_arr       = fltarr(N1) - 9999.9999
  tau_mean_arr    = fltarr(N1) - 9999.9999
  tau_med_arr     = fltarr(N1) - 9999.9999
  tau_gmean_arr   = fltarr(N1) - 9999.9999

  wmean_arr       = fltarr(N1) - 9999.9999
  mean_clip       = fltarr(N1) - 9999.9999
  
 for j=0L,N1-2L do begin
   
   id_fnd = where( wav_big gt  m_wav(j) and wav_big le m_wav(j+1), n_999)   ;;;;and finite(flux_big) ne 0
  
   if n_999 ge 3 then begin

      MEANCLIP, flux_big(id_fnd), mean_tmp, sigma_tmp, SUBS = idclipped,CLIPSIG=sig_tmp  ;;;;;=, MAXITER=, CONVERGE_NUM=, /VERBOSE, /DOUBLE ]
      mean_arr(j)      = mean( flux_big(id_fnd(idclipped)) )
      med_arr(j)       = median( flux_big(id_fnd(idclipped)), /EVEN)
      id_pix           = id_big(id_fnd(idclipped))      
      N_q_arr(j)       = n_elements(id_pix[UNIQ(id_pix, SORT(id_pix))])
      percentile_tmp   = PERCENTILES(flux_big(id_fnd(idclipped)), CONFLIMIT=0.68)
      percentile(0,j)  = percentile_tmp[0]  ;;;cgPercentiles(flux_big(id_fnd(idclipped)), Percentiles=[0.68])
      percentile(1,j)  = percentile_tmp[1]
      N_arr(j)         = n_999
      std_arr(j)       = sigma_tmp          ;;;;stdev( flux_big(id_fnd(idclipped)) )
      err_mean_arr(j)  = std_arr(j)/sqrt(N_arr(j))

      ;;;=========
      id_pos = where(flux_big(id_fnd(idclipped)) gt 0)
      gmean_arr(j)     = gmean(flux_big(id_fnd(idclipped(id_pos))))
      tau_mean_arr(j)  = mean( 1.0/flux_big(id_fnd(idclipped(id_pos))))
      tau_med_arr(j)   = median( 1.0/flux_big(id_fnd(idclipped(id_pos))),  /EVEN)
      tau_gmean_arr(j) = 1.0 ;;;gmean( alog(1.0/flux_big(id_fnd(idclipped(id_pos)))))

      wmean_arr(j)       = w_mean( flux_big(id_fnd(idclipped)),  SNR_big(id_fnd(idclipped))  ) ;1.0/sig_big(id_fnd(idclipped))^2
      MEANCLIP, flux_big(id_fnd(idclipped)), mean_tmp2, sigma_tmp2, SUBS = idclipped_2,CLIPSIG=5.0
      mean_clip(j) = mean( flux_big(id_fnd(idclipped(idclipped_2))) )
      
   endif else continue
   
  endfor
 
   wav_tmp      =  m_wav[0:N1-2] 
   med_tmp      =  med_arr[0:N1-2]  
   mean_tmp     =  mean_arr[0:N1-2]
   N_q_tmp      =  N_q_arr[0:N1-2]
   percent_tmp  =  percentile[*,0:N1-2]
   std_tmp      =  std_arr[0:N1-2]
   N_tmp        =  N_arr[0:N1-2]
   err_mean_tmp =  err_mean_arr[0:N1-2]

   gmean_tmp     = gmean_arr[0:N1-2]
   tau_mean_tmp  = tau_mean_arr[0:N1-2]
   tau_med_tmp   = tau_med_arr[0:N1-2]
   tau_gmean_tmp = tau_gmean_arr[0:N1-2]
   
   wmean_arr_tmp = wmean_arr[0:N1-2]
   mean_clip_tmp = mean_clip[0:N1-2]
   
   struc = {WAVE:wav_tmp, med_flux: med_tmp, mean_flux: mean_tmp, N_q: N_q_tmp, Percentile68: percent_tmp, std_flux:std_tmp, N_pix:N_tmp, $
            err_mean:err_mean_tmp, gmean_flux: gmean_tmp, med_tau: tau_med_tmp, mean_tau: tau_mean_tmp, tau_gmean: tau_gmean_tmp, wmean_flux: wmean_arr_tmp, $
            cmean_flux: mean_clip_tmp}

   ;if save_flg eq 1 then mwrfits,struc,out_file_tmp,/create

  return, struc
end

function stack_bootstrap,id_all,wav_r_all,flux_all,sig_all,siglevel=sig_tmp

  common CS, dir_data,dir_pwd,dir_out,ra_q,dec_q,zem_q,SNR_q,NUM_q,name_q,grism_flg,ra_cl,dec_cl,zemi_cl,M500,R500,q_cl_vel,r_phy,r_phy_R500,cat,pair_id,ll_wav,ll_flg,zemi_cl_org,zem_org,lya_flg,lyb_flg,ovia_flg,ovib_flg,civa_flg,R200,RNORM_200,SNR_lya,snr_lfr,ex500,CF_abs
  
  Common CS1, ion_name,ion_wav_c,wmin,wmax, reso,SNR_req,vel_tol,del_v
  
  nbootstrap = 200 
  ;;=====================
  n_arr=  fix((wmax - wmin )/reso)
  m_wav = wmin + (wmax - wmin )*findgen(n_arr)/n_arr
  N1 = n_elements(m_wav)
  ;;;;================for final storing ===========
  mean_final  =  fltarr(nbootstrap,N1-1) - 9999.9999
  med_final   =  fltarr(nbootstrap,N1-1) - 9999.9999

  gmean_final = fltarr(nbootstrap,N1-1) - 9999.9999

  wmean_final = fltarr(nbootstrap,N1-1) - 9999.9999
  cmean_final = fltarr(nbootstrap,N1-1) - 9999.9999

  
  mean_tau_final   =  fltarr(nbootstrap,N1-1) - 9999.9999
  med_tau_final   =  fltarr(nbootstrap,N1-1) - 9999.9999
  gmean_tau_final =  fltarr(nbootstrap,N1-1) - 9999.9999
  ;;;;=============================================

  nmin = 0L                        
  nmax = n_elements(name_q)     
  ;; nmin = 83766L
  ;; nmax = 6    
  tmp_a = wav_r_all
  
  for ii=0,nbootstrap-1 do begin
   randomNumbers = RANDOMU(999.99+ii, nmax)
   sortedRandomNumbers = SORT(randomNumbers)
   arr_rnd = nmin + ULONG64( nmax*randomNumbers)
   arr_rnd = pair_id(arr_rnd)
   id_f = get_matched_ids_2(id_all, arr_rnd,bootstrapid=ii)
   outfile1 = strcompress(dir_out+'composite_full-30K-2sig-51pix-bootstrap_'+string(ii)+'.fits',/remove_all)
   print,outfile1
   str_m = do_binning(wav_r_all(id_f),flux_all(id_f),sig_all(id_f),id_all(id_f),siglevel=sig_tmp)
   mean_final(ii,*) = str_m.mean_flux
   med_final(ii,*)  = str_m.med_flux
   gmean_final(ii,*)  = str_m.gmean_flux
   med_tau_final(ii,*)  = str_m.med_tau
   mean_tau_final(ii,*)  = str_m.mean_tau  
   gmean_tau_final(ii,*)  = str_m.tau_gmean

   wmean_final(ii,*)  = str_m.wmean_flux
   cmean_final(ii,*)  = str_m.cmean_flux
  endfor
  
  str_BS_final = {WAVE:m_wav[0:N1-2], mean_flux_BS: mean_final, med_flux_BS: med_final, gmean_flux_BS:gmean_final, $
                  med_tau_BS:med_tau_final , mean_tau_BS:mean_tau_final, gmean_tau_BS:gmean_tau_final, wmean_flux_BS: wmean_final, cmean_flux_BS:cmean_final }

  return,str_BS_final
end
;;;;;=======================

function get_master_wave_flux_arr

  common CS, dir_data,dir_pwd,dir_out,ra_q,dec_q,zem_q,SNR_q,NUM_q,name_q,grism_flg,ra_cl,dec_cl,zemi_cl,M500,R500,q_cl_vel,r_phy,r_phy_R500,cat,pair_id,ll_wav,ll_flg,zemi_cl_org,zem_org,lya_flg,lyb_flg,ovia_flg,ovib_flg,civa_flg,R200,RNORM_200,SNR_lya,snr_lfr,ex500,CF_abs
  
    Common CS1, ion_name,ion_wav_c,wmin,wmax, reso,SNR_req,vel_tol,del_v

  ;readcol,'contributed_file.txt',name_q,ll_wav,ll_wav_end,zem_q,zemi_cl,grism_flg,f='(x,A,f,f,f,f,I)'
  ;pair_id = indgen(N_cos)

  N_cos = n_elements(name_q)

  wav_r_all = [0.0d]  &  flux_all  = [0.0d]  & id_all = [0] & sig_all = [0.0d]  

  name_in = strarr(N_cos) + 'NF'
  outfile_in = strarr(N_cos) + 'NF'
  wav_in_s = fltarr(N_cos)
  wav_in_e = fltarr(N_cos)
  wav_in_s_r = fltarr(N_cos)
  wav_in_e_r = fltarr(N_cos)
  zem_in_q = fltarr(N_cos)
  zem_in_cl = fltarr(N_cos)
  gr_in     = intarr(N_cos)

  n1 = [''] & zq1 = [0.] & zcl1 = [0.] & f_gt1 = [0.]
  lll = 0
  lya_flg = strarr(N_cos) + 'Lya_Out'

  delz_lya = fltarr(N_cos) - 99999.999

  ;SNR_lya_tmp = fltarr(N_cos) - 99999.999

  tt = 0
  for k=0L, N_cos-1L do begin
  ;for k=0, 50 do begin
     
  ;print,k,name_q(k)

  ;file_qso = 'OVI/'+name_q(k)
  ;file_qso = FILE_SEARCH(dir_data+name_q(k)+'/*_conti.fits')

  
  if (grism_flg(k) eq 1) or (grism_flg(k) eq 10) then file_qso_old = FILE_SEARCH(dir_data+name_q(k)+'/*_coadd_G130M_final_lpALL.fits')
  if (grism_flg(k) eq 2) or (grism_flg(k) eq 20) then file_qso_old = FILE_SEARCH(dir_data+name_q(k)+'/*_coadd_G160M_final_lpALL.fits')
  if (grism_flg(k) eq 3) or (grism_flg(k) eq 30) then file_qso_old = FILE_SEARCH(dir_data+name_q(k)+'/*_coadd_FUVM_final_lpALL.fits')
  
  file_qso = FILE_SEARCH(dir_data+name_q(k)+'/*_conti_spectres.fits')
  
  aa=mrdfits(file_qso,1,h,/silent)
  wav=aa.wave & flux = aa.flux & sig = aa.error & conti =  aa.CONTI_SPLINE

  masked = mask_bad_reg(wav,flux,sig,conti, file_qso_old)
  wav = masked.w & flux = masked.f & sig = masked.e & conti =  masked.c
  
  ;;;;===================== To mask emission lines ==============
  ;masked_em  =  mask_emission(wav,flux,sig,conti,zem_q(k))
  ;wav = masked_em.w & flux = masked_em.f & sig = masked_em.e & conti =  masked_em.c
  ;;==================================================
  ;;1150,1449, 1406,1775
  
  wav_r = wav/(1.0 + zemi_cl(k)) &  flux_n = flux/conti &  sig_n = sig/conti
  
      if (grism_flg(k) eq 3) or (grism_flg(k) eq 30) then begin
        wmin_gr = max([1135, ll_wav(k)*(1.0 + zem_q(k))])
        wmax_gr = 1790
      endif else begin
         if (grism_flg(k) eq 1) or (grism_flg(k) eq 10) then begin
           wmin_gr = max([1135, ll_wav(k)*(1.0 + zem_q(k))])
           wmax_gr = 1450
        endif else begin
           if (grism_flg(k) eq 2) or (grism_flg(k) eq 20) then begin
              wmin_gr = max([1400, ll_wav(k)*(1.0 + zem_q(k))]) 
              wmax_gr = 1790
           endif
        endelse
     endelse

  wlim_l = max([ll_wav(k),wmin_gr,wmin*(1.0 + zemi_cl(k))])        
  wlim_u = min([wmax_gr,wmax*(1.0 + zemi_cl(k))])

  w_lya_loc_low = ion_wav_c - ion_wav_c*1000.0/299792.46  & w_lya_loc_up =  ion_wav_c + ion_wav_c*1000.0/299792.46
  ;w_lya_loc_low = ion_wav_c - ion_wav_c*500.0/299792.46  & w_lya_loc_up =  ion_wav_c + ion_wav_c*500.0/299792.46

  
  id_w = where( (wav ge wlim_l) and (wav le wlim_u) , n_wav)  ;;;; main check over full wave

  if n_wav gt 5 then begin

     id_lya_loc = where( (wav_r(id_w) ge w_lya_loc_low) and (wav_r(id_w) le w_lya_loc_up), n_lya_loc)    ;;;;; on selected region

     if ( median(flux(id_w)/sig(id_w)) gt SNR_req)  and n_lya_loc gt 5 then begin


        ;w1_mn = ion_wav_c - ion_wav_c*500.0/299792.46  & w2_mn =  ion_wav_c + ion_wav_c*500.0/299792.46
        ;id_mn = where( wav_r(id_w) gt w1_mn and wav_r(id_w) lt w2_mn, n_mn)

        n_mn = 2
        if n_mn gt 1 then begin

           
           mean_flx = median(flux_n(id_w))  ;;(id_mn)
           
           if mean_flx gt 0.9 then begin

              print,k,name_q(k),tt
              ;if tt eq 0 then wplot,wav_r(id_w),flux_n(id_w) else woplot,wav_r(id_w),flux_n(id_w),color=k
              
              
              wav_r_all = [wav_r_all, wav_r(id_w)]
              flux_all  = [flux_all , flux_n(id_w) ]
              id_all    = [id_all   , intarr(n_wav)+pair_id(k)]
              sig_all   = [sig_all,  median( flux(id_w)/ sig(id_w)) + flux_n(id_w)*0.0]   ;;;; sig_n(id_w) ]

              tt = k
              ;SNR_lya_tmp(k) = median( flux(id_w)/ sig(id_w) )
              ;lya_flg(k) = 'Lya_In'
           endif else continue
           
        endif else continue
        
        
     endif else continue
     
  endif else continue


  endfor  ;;;;; on k  

wav_r_all   = wav_r_all[1L:*]
flux_all    = flux_all[1L:*]
id_all      = id_all[1L:*]
sig_all     = sig_all[1L:*]

id_all_u = id_all[UNIQ(id_all, SORT(id_all))]


master_arr = {id_all: id_all, WAVE_r_all: wav_r_all, flux_all: flux_all, sig_all: sig_all, id_q_u: id_all_u}

;forprint,name_q,zem_q,zemi_cl,pair_id,lya_flg,SNR_lya,f='(A,1x,f,f,1x,I,1x,A,1x,f17.5)',text='lya_flg.txt'

;stop
;forprint,n1,zq1,zcl1,f_gt1,f='(A,1x,f17.5,1x,f17.5,1x,f17.5)',text='high_z_qcl_pairs_bad.txt'

;; id_nf = where(name_in ne 'NF')
;; forprint,name_in(id_nf),outfile_in(id_nf),wav_in_s(id_nf),wav_in_e(id_nf),zem_in_q(id_nf),zem_in_cl(id_nf),gr_in(id_nf),wav_in_s_r(id_nf),wav_in_e_r(id_nf),$
;;         f='(A,1x,A,1x,f17.5,1x,f17.5,1x,f17.5,1x,f17.5,1x,I,1x,f17.5,1x,f17.5)',text='contributed_file_ovi.txt',/nocomment

return, master_arr
end

pro get_stack_main,outfile

  common CS, dir_data,dir_pwd,dir_out,ra_q,dec_q,zem_q,SNR_q,NUM_q,name_q,grism_flg,ra_cl,dec_cl,zemi_cl,M500,R500,q_cl_vel,r_phy,r_phy_R500,cat,pair_id,ll_wav,ll_flg,zemi_cl_org,zem_org,lya_flg,lyb_flg,ovia_flg,ovib_flg,civa_flg,R200,RNORM_200,SNR_lya,snr_lfr,ex500,CF_abs
  
  Common CS1, ion_name,ion_wav_c,wmin,wmax, reso,SNR_req,vel_tol,del_v

   master_arr = get_master_wave_flux_arr()
   str_m = do_binning(master_arr.WAVE_r_all, master_arr.flux_all, master_arr.sig_all,master_arr.id_all,siglevel=50.0)

   flg_rnd = 0 & flg_bs = 0

   ;;====================================

   if flg_rnd eq 0 and flg_bs eq 0 then begin

   str_final = {WAVE:str_m.WAVE, med_flux: str_m.med_flux, mean_flux: str_m.mean_flux, N_q: str_m.N_q,Percentile68: str_m.Percentile68, $
                   std_flux:str_m.std_flux, N_pix:str_m.N_pix,err_mean:str_m.err_mean, gmean_flux: str_m.gmean_flux, med_tau: str_m.med_tau, $
                   mean_tau: str_m.mean_tau, tau_gmean: str_m.tau_gmean, id_q_u: master_arr.id_q_u, wmean_flux: str_m.wmean_flux, cmean_flux: str_m.cmean_flux}

   endif 
   
   ;;;;;; ================== for random regions =====================
   if flg_rnd eq 1 then begin
      randomNumbers = RANDOMU(10001, n_elements(zemi_cl))
      sortedRandomNumbers = SORT(randomNumbers)     
      rnd_ids = sortedRandomNumbers[0:n_elements(name_q)-1]
      zemi_cl = zemi_cl(rnd_ids)
      master_arr_rand = get_master_wave_flux_arr()
      str_m_rnd = do_binning(master_arr_rand.WAVE_r_all, master_arr_rand.flux_all, master_arr_rand.sig_all, master_arr_rand.id_all,siglevel=50.0)
      
      str_final = {WAVE:str_m.WAVE, med_flux: str_m.med_flux, mean_flux: str_m.mean_flux, N_q: str_m.N_q,Percentile68: str_m.Percentile68, $
                   std_flux:str_m.std_flux, N_pix:str_m.N_pix,err_mean:str_m.err_mean, gmean_flux: str_m.gmean_flux, med_tau: str_m.med_tau, $
                   mean_tau: str_m.mean_tau, tau_gmean: str_m.tau_gmean, id_q_u: master_arr.id_q_u, wmean_flux: str_m.wmean_flux, cmean_flux: str_m.cmean_flux, $
                   WAVE_rnd:str_m_rnd.WAVE, med_flux_rnd: str_m_rnd.med_flux, mean_flux_rnd: str_m_rnd.mean_flux,N_q_rnd: str_m_rnd.N_q, std_flux_rnd:str_m_rnd.std_flux, $
                   gmean_flux_rnd: str_m_rnd.gmean_flux, med_tau_rnd: str_m_rnd.med_tau, mean_tau_rnd: str_m_rnd.mean_tau, tau_gmean_rnd: str_m_rnd.tau_gmean }
    endif
   ;;====================================================
   zemi_cl = zemi_cl_org  ;;;; revert back to zemi_cl to its original values for other ion stacks
   zem_q   = zem_org
   ;;;====================== For bootstrap error computation ================
   if flg_bs eq 1 and flg_rnd eq 1 then begin
      str_bs = stack_bootstrap(master_arr.id_all, master_arr.WAVE_r_all, master_arr.flux_all,master_arr.sig_all,siglevel=20.0)
      str_final = {WAVE:str_m.WAVE, med_flux: str_m.med_flux, mean_flux: str_m.mean_flux, N_q: str_m.N_q,Percentile68: str_m.Percentile68, $
                   std_flux:str_m.std_flux, N_pix:str_m.N_pix,err_mean:str_m.err_mean, gmean_flux: str_m.gmean_flux, med_tau: str_m.med_tau, $
                   mean_tau: str_m.mean_tau, tau_gmean: str_m.tau_gmean, id_q_u: master_arr.id_q_u,wmean_flux: str_m.wmean_flux, cmean_flux: str_m.cmean_flux, $
                   WAVE_bs: str_bs.WAVE, med_flux_BS: str_bs.med_flux_BS, mean_flux_BS: str_bs.mean_flux_BS, gmean_flux_BS: str_bs.gmean_flux_BS, $
                   med_tau_BS: str_bs.med_tau_BS, mean_tau_BS: str_bs.mean_tau_BS, gmean_tau_BS: str_bs.gmean_tau_BS, wmean_flux_BS: str_bs.wmean_flux_BS, $
                   cmean_flux_BS: str_bs.cmean_flux_BS, WAVE_rnd:str_m_rnd.WAVE, med_flux_rnd: str_m_rnd.med_flux, mean_flux_rnd: str_m_rnd.mean_flux,$
                   N_q_rnd: str_m_rnd.N_q, std_flux_rnd:str_m_rnd.std_flux, gmean_flux_rnd: str_m_rnd.gmean_flux, med_tau_rnd: str_m_rnd.med_tau, $
                   mean_tau_rnd: str_m_rnd.mean_tau, tau_gmean_rnd: str_m_rnd.tau_gmean }
      
   endif
   ;;;;;====================================================
    
mwrfits,str_final,outfile,/create
return
end


pro get_sample_with_id, id_sample

  common CS, dir_data,dir_pwd,dir_out,ra_q,dec_q,zem_q,SNR_q,NUM_q,name_q,grism_flg,ra_cl,dec_cl,zemi_cl,M500,R500,q_cl_vel,r_phy,r_phy_R500,cat,pair_id,ll_wav,ll_flg,zemi_cl_org,zem_org,lya_flg,lyb_flg,ovia_flg,ovib_flg,civa_flg,R200,RNORM_200,SNR_lya,snr_lfr,ex500,CF_abs
   
  Common CS1, ion_name,ion_wav_c,wmin,wmax, reso,SNR_req,vel_tol,del_v
  
  ra_q         =    ra_q(id_sample)         
  dec_q        =    dec_q(id_sample)        
  zem_q        =    zem_q(id_sample)        
  SNR_q        =    SNR_q(id_sample)          
  NUM_q        =    NUM_q(id_sample)        
  name_q       =    name_q(id_sample)       
  grism_flg    =    grism_flg(id_sample)    
  ra_cl        =    ra_cl(id_sample)        
  dec_cl       =    dec_cl(id_sample)      
  zemi_cl      =    zemi_cl(id_sample)      
  M500         =    M500(id_sample)         
  R500         =    R500(id_sample)         
  q_cl_vel     =    q_cl_vel(id_sample)     
  r_phy        =    r_phy(id_sample)        
  r_phy_R500   =    r_phy_R500(id_sample)   
  cat          =    cat(id_sample)          
  pair_id      =    pair_id(id_sample)      
  
  ll_wav       =    ll_wav(id_sample)
  ll_flg       =    ll_flg(id_sample)
  zemi_cl_org  =    zemi_cl_org(id_sample)
  zem_org      =    zem_org(id_sample)
  lya_flg      =    lya_flg(id_sample)
  lyb_flg      =    lyb_flg(id_sample)
  ovia_flg     =    ovia_flg(id_sample)      
  ovib_flg     =    ovib_flg(id_sample)
  civa_flg     =    civa_flg(id_sample)
  R200         =    R200(id_sample)       
  RNORM_200    =    RNORM_200(id_sample)

  SNR_lya      =    SNR_lya(id_sample)
  snr_lfr      =    snr_lfr(id_sample) 
  ex500        =    ex500(id_sample)   
  CF_abs       =    CF_abs(id_sample)  
  
  
  return
end


pro stack_spec_main
  
  common CS, dir_data,dir_pwd,dir_out,ra_q,dec_q,zem_q,SNR_q,NUM_q,name_q,grism_flg,ra_cl,dec_cl,zemi_cl,M500,R500,q_cl_vel,r_phy,r_phy_R500,cat,pair_id,ll_wav,ll_flg,zemi_cl_org,zem_org,lya_flg,lyb_flg,ovia_flg,ovib_flg,civa_flg,R200,RNORM_200,SNR_lya,snr_lfr,ex500,CF_abs
  Common CS1, ion_name,ion_wav_c,wmin,wmax, reso,SNR_req,vel_tol,del_v

  dir_data = '/home/sapnamisra/IUCAA_work/New_Data/Fitted/'
  
  nskip = 0
  read_file = '5K_quasar-cl_pairs_with_1160-q-cl_info.txt' ;;;;;5K_quasar-cl_pairs_10R500_G130-160_Ex-bad.txt' 

  ;;;============================================= 
  readcol,read_file,ra_q,dec_q,zem_q,SNR_q,NUM_q,name_q,grism_flg,ra_cl,dec_cl,zemi_cl,M500,R500,q_cl_vel,r_phy,r_phy_R500,cat,pair_id,f=('f,f,f,f,I,A,I,f,f,f,f,f,f,f,f,A,I'),skipline= nskip
  readcol,read_file,ll_wav,ll_flg,lya_flg,lyb_flg,ovia_flg,ovib_flg,civa_flg,R200,RNORM_200,SNR_lya,snr_lfr,ex500,CF_abs,f=('x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,f,A,A,A,A,A,A,f,f,f,f,I,I'),skipline= nskip

  
  ;;;;====================
  zemi_cl_org = zemi_cl
  zem_org     =  zem_q

  id_pll = where(ll_flg eq 'pll', n_pll)
  if n_pll gt 0 then ll_wav(id_pll) = -9999.99900
  ;;;=====================================

  ;m_lim = cgPercentiles(M500, Percentiles=[0.68])
  ;id_ana = where( (lya_flg eq  'Lya_In') , n_ana)  ;;and (M500 lt m_lim(0))
  
  id_ana = where( (lya_flg eq  'Lya_In') and (zemi_cl gt 0.10653731521679899) and (ex500 eq -1) or (ex500 eq 2) or (ex500 eq 3), n_ana)
  ;id_ana = where( (lya_flg eq  'Lya_In') and (zemi_cl gt 0.10653731521679899) and (ex500 eq -1) or (ex500 eq 2) or (ex500 eq 3) and (CF_abs ne 1) and (CF_abs ne 2), n_ana)
  ;id_ana = where( (lya_flg eq  'Lya_In') and (zemi_cl gt 0.10653731521679899) and (CF_abs eq 1) or (CF_abs eq 2), n_ana)
  get_sample_with_id, id_ana 

  
  ;;===========================================================
  readcol,'stack_ions_lst.txt',ion_wav_full,ion_name_full,f_ion_full,f='(f,A,x,f)',skipline=1
  
  ion_name_input = ['lya']   ;;;;;;CIVA ;lya'] ;, 'CIVA', 'CIVB']   SiIIA, SiIIB,SiIIC, SiIII , SiIVA
  
  SNR_req = 1
  vel_tol = 3000.0
  del_v = 50.0
  
  binsample = 0
  
  if binsample eq 0 then begin

     print,'Hi I am in no bin-sample zone'

     ;for ww =0, n_elements(ion_name_arr)-1 do begin
     for ww = 0,0 do begin

        dir_out = 'Out/'
        id_ion_lst = where( ion_name_input(ww) eq ion_name_full, n_ion_lst)
        ion_name = ion_name_full(id_ion_lst(0))  & ion_wav_c =  ion_wav_full(id_ion_lst(0)) 
        wmin = ion_wav_c - ion_wav_c*vel_tol/299792.46  & wmax =  ion_wav_c + ion_wav_c*vel_tol/299792.46
        reso = (del_v*ion_wav_c)/299792.46
        
        ;;==============

        outfile_fits  = strcompress(dir_out+'composite_lya_942_mean_flux-gt0.9.fits',/remove_all)


        
        ;outfile_fits  = strcompress(dir_out+'composite_full-'+str(fix(del_v))+'kmps_SNgt'+str(SNR_req)+'_'+ion_name+'_rnorm-lt-10_const_SNR.fits',/remove_all)
        ;outfile_fits  = strcompress(dir_out+'composite_full-'+str(fix(del_v))+'kmps_SNgt'+str(SNR_req)+'_'+ion_name+'_rnorm-lt-10_241_abs_centeric.fits',/remove_all)
        ;outfile_fits  = strcompress(dir_out+'composite_full-'+str(fix(del_v))+'kmps_SNgt'+str(SNR_req)+'_'+ion_name+'_rnorm-lt-10_comm_lya-ovia.fits',/remove_all)
        ;outfile_fits  = strcompress(dir_out+'composite_full-'+str(fix(del_v))+'kmps_SNgt'+str(SNR_req)+'_'+ion_name+'_rnorm-lt-10_CF_sample_1553.fits',/remove_all)
        ;outfile_fits  = strcompress(dir_out+'composite_full-'+str(fix(del_v))+'kmps_SNgt'+str(SNR_req)+'_'+ion_name+'_rnorm-lt-10_CF_sample_649.fits',/remove_all)
        ;outfile_fits  = strcompress(dir_out+'composite_full-'+str(fix(del_v))+'kmps_SNgt'+str(SNR_req)+'_'+ion_name+'_rnorm-lt-10_CF_sample_noAbs.fits',/remove_all)
        ;outfile_fits  = strcompress(dir_out+'composite_full-'+str(fix(del_v))+'kmps_SNgt'+str(SNR_req)+'_'+ion_name+'_rnorm-lt-10_CF_sample_only_Abs.fits',/remove_all)


        ;outfile_fits  = strcompress(dir_out+'composite_full-'+str(fix(del_v))+'kmps_SNgt'+str(SNR_req)+'_'+ion_name+'_rnorm-lt-10_942_non-detection.fits',/remove_all)
        ;outfile_fits  = strcompress(dir_out+'composite_1160_all.fits',/remove_all)
        ;outfile_fits  = strcompress(dir_out+'composite_218_detection.fits',/remove_all)
        ;outfile_fits  = strcompress(dir_out+'composite_full_ex68_M500.fits',/remove_all)

        
        get_stack_main,outfile_fits
          
     endfor
     
  endif else begin

  ;;;;;============================ for binning samples ==================
  print,'Hi I am in bin-sample zone'

  dir_out = 'Out/SNR_ana/Const-SNR/' ;;;;;Out/Full/Inner-200/'   ;;;;;;'Out/ExCGM/Bins-5/'
  ;;;============ change for lya =======================
  ww = 0  ;;; 0:lya, 1: ovi
  
  ;;;;;;;===========================================
  if ww eq 0 then begin
     ;id_lya_in = where( lya_flg eq 'Lya_In' and zemi_cl gt 0.10653731521679899 and ( (ex_flg eq -1) or (ex_flg eq 2) or (ex_flg eq 3)), n_lya_in)
     id_lya_in = where( lya_flg eq 'Lya_In' , n_lya_in)
     get_sample_with_id, id_lya_in

  endif
  ;;;;===============================

  percent_var = SNR_lya
  percent_var_tag = 'SNR'
  bins_arr =  cgPercentiles(percent_var, Percentiles=[0.333333, 0.666666])

  ;bins_arr = [4.30035 ,     6.96786, 11.865835]
  ;bins_arr = [3,6,9,12,15,20]
  ;bins_arr = [5,10,15,20]  ;;[3,4,5,6,8,15,20]
  ;bins_arr =  cgPercentiles(percent_var, Percentiles=[0.25, 0.5,0.75])

   ;; percent_var = r_phy
   ;; percent_var_tag = 'rphy'
   ;; bins_arr =  [1.5,3.0,4.5,6.0]  ;;; for rphy for full

   ;; percent_var = r_phy/R500
   ;; percent_var_tag = 'rnorm'
   ;; bins_arr =  [1.5,3.5,5.5,8.0]  ;;; for rnorm for full


   ;percent_var = RNORM_200 
   ;percent_var_tag = 'RNORM200'

   
   ;bins_arr =  [1.5,2.5,4.0,6.5] ;;; for RNORM200 paper
   ;bins_arr =  [0.75,1.5,2.5,4.0,6.5] ;;; for RNORM200 paper inner bins
   
   bin_num = 2

   ;for bbb = 0 , n_elements(bins_arr) do begin
   for bbb = bin_num , bin_num do begin

     id_ion_lst = where( ion_name_input(ww) eq ion_name_full, n_ion_lst)
     ion_name = ion_name_full(id_ion_lst(0))  & ion_wav_c =  ion_wav_full(id_ion_lst(0))     
     wmin = ion_wav_c - ion_wav_c*vel_tol/299792.46  & wmax =  ion_wav_c + ion_wav_c*vel_tol/299792.46
     reso = (del_v*ion_wav_c)/299792.46

     CASE bbb OF
        
        0: begin
           id_bin = where( percent_var lt bins_arr[bbb], n_bins) 
           print,bbb,bins_arr[bbb],n_bins
        end

        n_elements(bins_arr): begin
           id_bin = where( percent_var ge bins_arr[bbb-1], n_bins)
           print,bbb,bins_arr[bbb-1],n_bins
        end

        ELSE: begin
           id_bin = where( percent_var ge bins_arr[bbb-1] and percent_var lt bins_arr[bbb], n_bins)
           print,bbb,bins_arr[bbb-1],bins_arr[bbb],n_bins  ;;;bbb-1,bbb
        end

     ENDCASE

     get_sample_with_id, id_bin
     outfile_fits  = strcompress(dir_out+'Bin'+str(bbb+1)+'-'+str(fix(del_v))+'kmps_'+ion_name+'_'+percent_var_tag+'.fits',/remove_all)
     print,outfile_fits,n_elements(id_bin), n_elements(UNIQ(name_q, SORT(name_q)))
     get_stack_main,outfile_fits

     
   end ;;;; loop on bin_arr

  
  endelse


return

end
