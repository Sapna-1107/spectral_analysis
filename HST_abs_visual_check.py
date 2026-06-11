"""
Python translation of IDL HST absorption visual check code.
HST_abs_visual_check.pro → Python / matplotlib / tkinter
Key libraries:
  astropy.io.fits   — replaces mrdfits
  numpy             — array ops
  matplotlib/TkAgg  — plotting (replaces IDL direct graphics)
  tkinter           — GUI (replaces IDL widgets)
  scipy             — Gaussian smoothing for gaussfold
Translation notes:
  IDL COMMON blocks      → AppState class (single global instance S)
  IDL 'where'            → np.where(...)[0]
  IDL 'remove'           → boolean-mask deletion
  IDL shift(arr,1)       → np.roll(arr, 1)
  IDL FILE_SEARCH        → glob.glob
  IDL mrdfits            → astropy.io.fits.getdata
  IDL readcol            → custom line-by-line parser
  IDL cursor/ginput      → tkinter Toplevel + mpl_connect click handler
  IDL WIDGET_* / XMANAGER → tkinter root.mainloop()
"""
import glob
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import messagebox
# ============================================================
#  Shared application state  (replaces IDL COMMON blocks)
# ============================================================
class AppState:
    # HAVC_1
    dir_name: str = ""
    name_q       = None
    zem_q        = None
    grism_flg    = None
    zemi_cl      = None
    ll_wav       = None
    pair_id      = None
    lya_flg      = None
    lyb_flg      = None
    ovia_flg     = None
    ovib_flg     = None
    N_Spectra_files: int = 0
    file_index:     int = 0
    rwave        = None
    rwave_name   = None
    f_ion_a      = None
    # HAVC_2 — matplotlib figure/canvas
    fig_left     = None
    canvas_left  = None
    # HAVC_3
    lya_T:    int   = -1
    lyb_T:    int   = -1
    lyg_T:    int   = -1
    lyd_T:    int   = -1
    lye_T:    int   = -1
    ovi_T:    int   = -1
    cii_T:    int   = -1
    siII_T:   int   = -1
    siIII_T:  int   = -1
    Line_CF:  int   = -1
    lim_v1_lya: float = -9999.0
    lim_v2_lya: float = -9999.0
    ew_lya:     float = -9999.0
    sigew_lya:  float = -9999.0
    vcent_lya:  float = -9999.0
    lim_v_lf:   float = -9999.0
    ew_lf:      float = -9999.0
    ex_T:     int   = -1
    # HAVC_4
    ll_ion: int = 6   # 6 = Lya (0-based)
    # GS — global spectrum
    wav   = None
    flux  = None
    sig   = None
    conti = None
    # Output file handle
    out_fh = None
S = AppState()
# ============================================================
#  Plot colour constants  (black-background IDL style)
# ============================================================
BG      = "black"
FG      = "white"
SPEC_C  = "#00CFFF"   # spectrum trace — bright cyan
CONT_C  = "#FFFF00"   # continuum hline — yellow
V0_C    = "#FF4444"   # zero-velocity vline — red
V200_C  = "#44FF44"   # ±200 km/s — green
V400_C  = "#4488FF"   # ±400 km/s — blue
V500_C  = "#AAAAAA"   # ±500 km/s — grey
V600_C  = "#00FFFF"   # ±600 km/s — cyan
_AOD_COLORS = {
    0: "#FF8888",   # OVIB
    1: "#FFAA44",   # OVIA
    4: "#AAFFAA",   # lyg
    5: "#88CCFF",   # lyb
    6: "white",     # lya
}
# ============================================================
#  GUI colour constants  — gray background, bold blue text
# ============================================================
GUI_BG   = "#111111"   # window / frame background
SEP_COL  = "#444444"   # thin separator frames
# All button backgrounds: shades of gray
NAV_BG   = "#888888"   # Navigation: Back / Next
NAV_ACT  = "#aaaaaa"
ACT_BG   = "#777777"   # Measurement: Get Lyα / Get LF
ACT_ACT  = "#999999"
SAVE_BG  = "#666666"   # Save / Done / Exclude
SAVE_ACT = "#888888"
FLAG_BG  = "#777777"   # Line-classification toggles
FLAG_ACT = "#999999"
FLAG_LIT = "#cccccc"   # highlighted when a flag is active
CF_BG    = "#777777"   # Covering-fraction buttons
CF_ACT   = "#999999"
# Universal button text — bold blue
BTN_FG   = "#0055DD"
BTN_FONT = ("Helvetica", 12, "bold")
fnt_sz = 10
# ============================================================
#  mask_bad_reg
# ============================================================
def mask_bad_reg(wav_t, flux_t, sig_t, conti_t, file_qso_old):
    wav_t   = wav_t.copy()
    flux_t  = flux_t.copy()
    sig_t   = sig_t.copy()
    conti_t = conti_t.copy()
    bb   = fits.getdata(file_qso_old, 1)
    w_ot = bb["wave"].astype(float)
    e_ot = bb["error"].astype(float)
    id_e0 = np.where(e_ot == 0)[0]
    n_e0  = len(id_e0)
    l_wave_jump = np.array([], dtype=float)
    u_wave_jump = np.array([], dtype=float)
    if n_e0 > 3:
        dim1     = n_e0 - 1
        dim2     = n_e0 - 2
        idd_jump = id_e0[1:dim1+1] - id_e0[0:dim2+1]
        id_gap   = np.where(idd_jump > 1)[0]
        n_gap    = len(id_gap)
        if n_gap >= 1:
            l_wave_jump = np.concatenate([[id_e0[0]], id_e0[id_gap + 1]])
            u_wave_jump = np.concatenate([id_e0[id_gap], [id_e0[dim1]]])
        else:
            l_wave_jump = np.array([id_e0[0]])
            u_wave_jump = np.array([id_e0[dim1]])
    fixed_low = np.array([1198.0, 1208.0, 1300.0])
    fixed_up  = np.array([1202.0, 1224.0, 1308.0])
    if len(l_wave_jump) > 0:
        mask_reg_low = np.concatenate([fixed_low, w_ot[l_wave_jump.astype(int)]])
        mask_reg_up  = np.concatenate([fixed_up,  w_ot[u_wave_jump.astype(int)]])
    else:
        mask_reg_low = fixed_low
        mask_reg_up  = fixed_up
    id_bad_reg = np.array([], dtype=int)
    for jj in range(len(mask_reg_low)):
        id_bad = np.where(
            (wav_t >= mask_reg_low[jj]) & (wav_t <= mask_reg_up[jj])
        )[0]
        if len(id_bad) > 0:
            id_bad_reg = np.concatenate([id_bad_reg, id_bad])
    if len(id_bad_reg) > 0:
        id_bad_reg = np.unique(id_bad_reg)
        good = np.ones(len(wav_t), dtype=bool)
        good[id_bad_reg] = False
        wav_t   = wav_t[good]
        flux_t  = flux_t[good]
        sig_t   = sig_t[good]
        conti_t = conti_t[good]
    return {"w": wav_t, "f": flux_t, "e": sig_t, "c": conti_t}
# ============================================================
#  get_spec
# ============================================================
def get_spec(num):
    gf = S.grism_flg[num]
    if gf in (1, 10):
        pattern_old = S.dir_name + S.name_q[num] + "/*_coadd_G130M_final_lpALL.fits"
    elif gf in (2, 20):
        pattern_old = S.dir_name + S.name_q[num] + "/*_coadd_G160M_final_lpALL.fits"
    elif gf in (3, 30):
        pattern_old = S.dir_name + S.name_q[num] + "/*_coadd_FUVM_final_lpALL.fits"
    else:
        raise ValueError(f"Unknown grism_flg: {gf}")
    matches = glob.glob(pattern_old)
    if not matches:
        raise FileNotFoundError(f"No file: {pattern_old}")
    file_qso_old = matches[0]
    conti_pattern = S.dir_name + S.name_q[num] + "/*_conti_spectres.fits"
    cm = glob.glob(conti_pattern)
    if not cm:
        raise FileNotFoundError(f"No conti file: {conti_pattern}")
    aa    = fits.getdata(cm[0], 1)
    wav   = aa["wave"].astype(float)
    flux  = aa["flux"].astype(float)
    sig   = aa["error"].astype(float)
    conti = aa["CONTI_SPLINE"].astype(float)
    masked = mask_bad_reg(wav, flux, sig, conti, file_qso_old)
    wav   = masked["w"]
    flux  = masked["f"]
    sig   = masked["e"]
    conti = masked["c"]
    if gf in (3, 30):
        wmin_gr = max(1135.0, S.ll_wav[num] * (1.0 + S.zem_q[num]))
        wmax_gr = 1790.0
    elif gf in (1, 10):
        wmin_gr = max(1135.0, S.ll_wav[num] * (1.0 + S.zem_q[num]))
        wmax_gr = 1450.0
    else:
        wmin_gr = max(1400.0, S.ll_wav[num] * (1.0 + S.zem_q[num]))
        wmax_gr = 1790.0
    id_spec = np.where((wav >= wmin_gr) & (wav <= wmax_gr))[0]
    S.wav   = wav[id_spec] / (1.0 + S.zemi_cl[num])
    S.flux  = flux[id_spec]
    S.sig   = sig[id_spec]
    S.conti = conti[id_spec]
# ============================================================
#  gaussfold / get_abs_prop
# ============================================================
def gaussfold(x, y, sigma_pix):
    return gaussian_filter1d(y, sigma_pix / 2.3548)
def get_abs_prop(v1, v2, w_rest, f_ion):
    vel        = 299792.46 * (S.wav / w_rest - 1.0)
    Const_fact = 2.654e-15 * f_ion * w_rest
    tau        = np.log(S.conti / S.flux)
    N_app      = tau / Const_fact
    dwv        = S.wav - np.roll(S.wav, 1)
    ilhs = np.argmin(np.abs(vel - v1))
    irhs = np.argmin(np.abs(vel - v2))
    if ilhs > irhs:
        ilhs, irhs = irhs, ilhs
    sl = slice(ilhs, irhs + 1)
    ew1    = np.sum((1.0 - S.flux[sl] / S.conti[sl]) * dwv[sl])
    sigew1 = np.sqrt(np.sum((S.sig[sl] / S.conti[sl] * dwv[sl]) ** 2))
    Nfact  = 1.13e20 / (w_rest ** 2 * f_ion)
    N_lya  = np.log10(max(ew1 * Nfact, 1e-30))
    N_err  = 1.0 / 2.303
    id_cent = np.where((vel >= v1) & (vel <= v2))[0]
    if len(id_cent) > 3:
        sm      = gaussfold(vel[id_cent], S.flux[id_cent] / S.conti[id_cent], 2.0)
        midx    = np.where(sm == sm.min())[0]
        vcent_loc = vel[id_cent[int(np.median(midx))]]
    else:
        vcent_loc = -9999.0
    return {"ew": ew1, "ew_err": sigew1, "N": N_lya, "N_err": N_err,
            "vcent": vcent_loc, "vaod": vel, "Naod": N_app}
# ============================================================
#  Axis helpers
# ============================================================
def vline(ax, positions, color="w", linestyle="--", linewidth=1.5):
    if np.isscalar(positions):
        positions = [positions]
    for xv in positions:
        ax.axvline(xv, color=color, linestyle=linestyle, linewidth=linewidth)
def hline(ax, position, color="w", linestyle="--", linewidth=1.5):
    ax.axhline(position, color=color, linestyle=linestyle, linewidth=linewidth)
def _style_ax(ax, show_xlabel=False):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(FG)
        sp.set_linewidth(0.7)
    # y-tick marks kept but ALL y-tick labels suppressed;
    # a single common "Normalized Flux" label is placed on the figure instead.
    ax.tick_params(axis="both", which="both",
                   bottom=True, top=True, left=True, right=True,
                   labelbottom=show_xlabel, labeltop=False,
                   labelleft=False, labelright=False,
                   colors=FG, direction="in", length=3, width=0.7,
                   labelsize=5.5)
    if show_xlabel:
        ax.set_xlabel("Velocity (km s$^{-1}$)", color=FG, fontsize= fnt_sz, fontweight="bold")
# ============================================================
#  main_plot_velocity
# ============================================================
def main_plot_velocity():
    fig = S.fig_left
    fig.clf()
    fig.patch.set_facecolor(BG)
    vel_lim = 600.0
    # Column geometry
    col = {0: (0.030, 0.318), 7: (0.348, 0.638)}
    aod_x0, aod_x1 = 0.668, 0.995
    # y_bot=0.06, AOD column tops out at 0.06+0.37=0.43 (Lyman) and
    # 0.50+0.37=0.87 (OVI).  7 panels × 0.116 + 0.06 = 0.872 ≈ matches OVI top.
    y_bot = 0.06
    y_ht  = 0.116
    y_gap = 0.0
    title_str = (
        f"{S.file_index}: {S.name_q[S.file_index]}   "
        f"zcl={S.zemi_cl[S.file_index]:.5f}   "
        f"Pairid:{S.pair_id[S.file_index]}"
    )
    ax_aod1 = None   # Lyman AOD  (lower)
    ax_aod2 = None   # OVI AOD    (upper)
    x_pos_l = col[0][0]
    x_pos_u = col[0][1]
    y_pos_l = y_bot
    y_pos_u = y_bot + y_ht
    for ii in range(14):
        if ii == 0:
            x_pos_l, x_pos_u = col[0]
            y_pos_l = y_bot
            y_pos_u = y_bot + y_ht
        if ii == 7:
            x_pos_l, x_pos_u = col[7]
            y_pos_l = y_bot
            y_pos_u = y_bot + y_ht
        vel       = 299792.46 * (S.wav / S.rwave[ii] - 1.0)
        id_plt    = np.where((vel >= -vel_lim) & (vel <= vel_lim))[0]
        if len(id_plt) > 3:
            ax = fig.add_axes([x_pos_l, y_pos_l,
                               x_pos_u - x_pos_l, y_pos_u - y_pos_l])
            is_bottom = ii in (0, 7)
            # CHANGE 1: smooth spectrum by 3 pixels before plotting
            flux_norm = S.flux[id_plt] / S.conti[id_plt]
            flux_sm   = gaussfold(vel[id_plt], flux_norm, 2.0)
            ax.step(vel[id_plt], flux_sm,
                    where="mid", linewidth=0.9, color=SPEC_C)
            ax.set_xlim(-vel_lim, vel_lim)
            # CHANGE 2: uniform ylim (0, 1.1) for all spectral panels
            ax.set_ylim(0.0, 1.1)
            _style_ax(ax, show_xlabel=is_bottom)
            # Ion name — gold, top-right
            ax.text(vel_lim - 12, 1.1 * 0.6,
                    S.rwave_name[ii],
                    color="#FFD700", fontsize=fnt_sz,
                    ha="right", va="top", fontweight="bold")
            vline(ax, 0.0,         color=V0_C,   linestyle="--", linewidth=1.3)
            vline(ax, [-200, 200], color=V200_C, linestyle=":",  linewidth=0.9)
            vline(ax, [-400, 400], color=V400_C, linestyle=":",  linewidth=0.9)
            vline(ax, [-500, 500], color=V500_C, linestyle=":",  linewidth=0.7)
            vline(ax, [-600, 600], color=V600_C, linestyle=":",  linewidth=0.9)
            hline(ax, 1.0,         color=CONT_C, linestyle="--", linewidth=1.0)
            # ---- Lyman AOD panel (ii = 4,5,6) ----
            if 4 <= ii <= 6:
                abs_aod = get_abs_prop(-100.0, 100.0, S.rwave[ii], S.f_ion_a[ii])
                if ax_aod1 is None:
                    ax_aod1 = fig.add_axes([aod_x0, 0.06,
                                            aod_x1 - aod_x0, 0.37])
                    ax_aod1.set_facecolor(BG)
                    for sp in ax_aod1.spines.values():
                        sp.set_color(FG); sp.set_linewidth(0.7)
                naod_sm = gaussfold(abs_aod["vaod"], abs_aod["Naod"] / 1e12, 2.0)
                ax_aod1.step(abs_aod["vaod"], naod_sm,
                             where="mid", color=_AOD_COLORS.get(ii, FG),
                             linewidth=1.0, label=S.rwave_name[ii])
            # ---- OVI AOD panel (ii = 0,1) ----
            if ii in (0, 1):
                abs_aod = get_abs_prop(-100.0, 100.0, S.rwave[ii], S.f_ion_a[ii])
                if ax_aod2 is None:
                    ax_aod2 = fig.add_axes([aod_x0, 0.50,
                                            aod_x1 - aod_x0, 0.37])
                    ax_aod2.set_facecolor(BG)
                    for sp in ax_aod2.spines.values():
                        sp.set_color(FG); sp.set_linewidth(0.7)
                naod_sm = gaussfold(abs_aod["vaod"], abs_aod["Naod"] / 1e12, 2.0)
                ax_aod2.step(abs_aod["vaod"], naod_sm,
                             where="mid", color=_AOD_COLORS.get(ii, FG),
                             linewidth=1.0, label=S.rwave_name[ii])
        y_pos_l = y_pos_u + y_gap
        y_pos_u = y_pos_l + y_ht
    # ---- Finalise AOD panels ----
    for ax_aod, series_label in [(ax_aod1, "Lyman series"), (ax_aod2, "OVI series")]:
        if ax_aod is None:
            continue
        ax_aod.set_xlim(-vel_lim, vel_lim)
        ax_aod.set_ylim(-0.1, 5.0)
        vline(ax_aod, 0.0,         color=V0_C,   linestyle="--", linewidth=1.3)
        vline(ax_aod, [-200, 200], color=V200_C, linestyle=":",  linewidth=0.8)
        vline(ax_aod, [-400, 400], color=V400_C, linestyle=":",  linewidth=0.8)
        hline(ax_aod, 0.0,         color=FG,     linestyle="-",  linewidth=0.5)
        ax_aod.tick_params(axis="both", which="both",
                           bottom=True, top=True, left=True, right=True,
                           labelbottom=True, labelright=True,
                           labelleft=False, labeltop=False,
                           colors=FG, direction="in", length=3, width=0.7,
                           labelsize=5.5)
        ax_aod.yaxis.set_label_position("right")
        ax_aod.set_ylabel(r"$N_\mathrm{app}$  ($10^{12}$ cm$^{-2}$)",
                          color=FG, fontsize=fnt_sz, labelpad=3)
        ax_aod.set_xlabel("Velocity (km s$^{-1}$)", color=FG, fontsize=fnt_sz, fontweight="bold")
        ax_aod.text(0.03, 0.97, series_label,
                    transform=ax_aod.transAxes,
                    color=CONT_C, fontsize=fnt_sz, fontweight="bold",
                    va="top", ha="left")
        ax_aod.legend(fontsize=fnt_sz, loc="upper right",
                      facecolor="#1a1a1a", edgecolor=FG,
                      labelcolor=FG, framealpha=0.85)
    # Single common y-label for the left + middle spectral panels
    fig.text(0.02, 0.5, "Normalized Flux",
             color=FG, fontsize=fnt_sz, fontweight="bold",
             ha="center", va="center", rotation="vertical")
    # Section headers
    mid0 = (col[0][0] + col[0][1]) / 2
    mid7 = (col[7][0] + col[7][1]) / 2
    midA = (aod_x0 + aod_x1) / 2
    for x, txt in [(mid0, "Normalised Spectra  (F / F$_{cont}$)"),
                   (mid7, "Normalised Spectra  (F / F$_{cont}$)"),
                   (midA, "Apparent Optical Depth  ($N_{app}$)")]:
        fig.text(x, 0.920, txt, color=FG, fontsize=fnt_sz,
                 ha="center", va="top", fontweight="bold")
    # CHANGE 3: larger, bold title
    fig.suptitle(title_str, color=FG, fontsize=fnt_sz+2, fontweight="bold", y=0.965)
    S.canvas_left.draw()
# ============================================================
#  reset_line_flags
# ============================================================
def reset_line_flags():
    S.lya_T = S.lyb_T = S.lyg_T = S.lyd_T = S.lye_T = -1
    S.ovi_T = S.cii_T = S.siII_T = S.siIII_T = S.Line_CF = -1
    S.lim_v1_lya = S.lim_v2_lya = -9999.0
    S.ew_lya = S.sigew_lya = S.vcent_lya = -9999.0
    S.lim_v_lf = S.ew_lf = -9999.0
    S.ex_T = -1
# ============================================================
#  Click-collection window
#  Uses tkinter Toplevel + mpl_connect — no plt.show/ginput
#  so it does NOT fight the running mainloop.
# ============================================================
def _click_window(n_clicks, instruction, title_suffix=""):
    """
    Open a Toplevel with a black-background spectrum plot.
    Collect exactly *n_clicks* left-button clicks (ignoring toolbar
    zoom/pan mode) and return them as [(x, y), ...].
    A red dashed marker is drawn at each click so the user can see it.
    The window closes automatically after the last click.
    Fixes vs. original:
      - BooleanVar explicitly tied to root so it survives Toplevel destruction
      - done_v.set(True) fires BEFORE top.destroy() — avoids silent failure
      - root.wait_variable() used (safe after Toplevel is gone)
      - top.lift() / focus_force() so the new window receives mouse events
      - event.xdata is None replaces event.inaxes is None (more robust)
      - str(toolbar.mode) cast handles the _Mode enum in matplotlib >= 3.6
      - cid stored in a list so _finish() can safely disconnect
    """
    root    = S.canvas_left.get_tk_widget().winfo_toplevel()
    clicks  = []
    done_v  = tk.BooleanVar(master=root, value=False)
    # ---- Toplevel ----
    top = tk.Toplevel(root)
    top.title(f"Click selector — {title_suffix}")
    top.configure(bg=GUI_BG)
    top.protocol("WM_DELETE_WINDOW", lambda: None)
    top.lift()
    top.focus_force()
    # ---- Banner ----
    tk.Label(top, text=instruction,
             bg="#1a1a3a", fg="#FFD700",
             font=("Helvetica", fnt_sz+5, "bold"),
             pady=8, padx=12, anchor="w").pack(fill=tk.X)
    # ---- Counter ----
    counter_var = tk.StringVar(value=f"  Clicks remaining: {n_clicks}")
    tk.Label(top, textvariable=counter_var,
             bg=GUI_BG, fg="#00CFFF",
             font=("Courier", fnt_sz+5, "bold"),
             anchor="w", padx=8).pack(fill=tk.X)
    # ---- Spectrum figure ----
    fig = plt.Figure(figsize=(11, 5), dpi=100)
    fig.patch.set_facecolor(BG)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(FG)
    ax.tick_params(colors=FG, direction="in", which="both",
                   top=True, right=True, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    vel    = 299792.46 * (S.wav / S.rwave[S.ll_ion] - 1.0)
    id_plt = np.where((vel >= -600.0) & (vel <= 600.0))[0]
    # also smooth in the click window for consistency
    flux_sm = gaussfold(vel[id_plt], S.flux[id_plt] / S.conti[id_plt], 3.0)
    ax.step(vel[id_plt], flux_sm,
            where="mid", color=SPEC_C, linewidth=1.3)
    ax.set_xlim(-600, 600)
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("Velocity (km s$^{-1}$)", color=FG, fontsize=fnt_sz)
    ax.set_ylabel("Normalized Flux",        color=FG, fontsize=fnt_sz)
    ax.set_title(
        f"{S.file_index}: {S.name_q[S.file_index]}   "
        f"zcl={S.zemi_cl[S.file_index]:.5f}   "
        f"Pairid:{S.pair_id[S.file_index]}",
        color=FG, fontsize=fnt_sz, fontweight="bold", pad=5,
    )
    ax.text(0.98, 0.97, S.rwave_name[S.ll_ion],
            transform=ax.transAxes,
            color="#FFD700", fontsize=fnt_sz, ha="right", va="top", fontweight="bold")
    hline(ax, 1.0, color=CONT_C, linestyle="--", linewidth=1.2)
    vline(ax, 0.0,         color=V0_C,   linestyle="--", linewidth=1.5)
    vline(ax, [-200, 200], color=V200_C, linestyle=":",  linewidth=1.0)
    vline(ax, [-400, 400], color=V400_C, linestyle=":",  linewidth=1.0)
    fig.tight_layout()
    canvas  = FigureCanvasTkAgg(fig, master=top)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.get_tk_widget().focus_set()
    tb_frame = tk.Frame(top, bg="black")
    tb_frame.pack(fill=tk.X)
    toolbar  = NavigationToolbar2Tk(canvas, tb_frame)
    toolbar.config(bg="black")
    toolbar.update()
    cid = [None]
    def _finish():
        if cid[0] is not None:
            fig.canvas.mpl_disconnect(cid[0])
            cid[0] = None
        top.protocol("WM_DELETE_WINDOW", top.destroy)
        done_v.set(True)
        root.after(50, top.destroy)
    def on_click(event):
        if event.xdata is None or event.button != 1:
            return
        if str(toolbar.mode) != "":
            return
        clicks.append((event.xdata, event.ydata))
        ax.axvline(event.xdata, color="#FF4444", linestyle="--",
                   linewidth=1.5, alpha=0.9)
        canvas.draw_idle()
        remaining = n_clicks - len(clicks)
        if remaining > 0:
            counter_var.set(f"  Clicks remaining: {remaining}")
        else:
            counter_var.set("  Done — closing…")
            top.after(500, _finish)
    cid[0] = fig.canvas.mpl_connect("button_press_event", on_click)
    root.wait_variable(done_v)
    plt.close(fig)
    return clicks
# ============================================================
#  Event callbacks
# ============================================================
def on_next():
    if S.file_index <= S.N_Spectra_files - 2:
        S.file_index += 1
    else:
        messagebox.showinfo("Info", "No More Spectra Files…")
        return
    reset_line_flags()
    get_spec(S.file_index)
    main_plot_velocity()
def on_back():
    if S.file_index >= 1:
        S.file_index -= 1
    else:
        messagebox.showinfo("Info", "At beginning of Spectra Files…")
        return
    reset_line_flags()
    get_spec(S.file_index)
    main_plot_velocity()
def on_get_lya():
    reset_line_flags()
    pts = _click_window(
        n_clicks=3,
        instruction="Click 3 points:  (1) v1_lya   (2) v2_lya   (3) line-free v_lf",
        title_suffix="Get Lyα limits",
    )
    if len(pts) < 3:
        return
    vel = 299792.46 * (S.wav / S.rwave[S.ll_ion] - 1.0)
    S.lim_v1_lya = pts[0][0]
    S.lim_v2_lya = pts[1][0]
    S.lim_v_lf   = pts[2][0]
    ap = get_abs_prop(S.lim_v1_lya, S.lim_v2_lya,
                      S.rwave[S.ll_ion], S.f_ion_a[S.ll_ion])
    S.ew_lya    = ap["ew"]
    S.sigew_lya = ap["ew_err"]
    S.vcent_lya = ap["vcent"]
    mn_lf = np.argmin(np.abs(vel - S.lim_v_lf))
    lo = max(0, mn_lf - 10)
    hi = min(len(S.flux) - 1, mn_lf + 10)
    S.ew_lf = float(np.median(S.flux[lo:hi] / S.sig[lo:hi]))
    main_plot_velocity()
def on_get_lf():
    pts = _click_window(
        n_clicks=1,
        instruction="Click 1 point:  line-free velocity (v_lf) for SNR",
        title_suffix="Get LF",
    )
    if not pts:
        return
    vel = 299792.46 * (S.wav / S.rwave[S.ll_ion] - 1.0)
    S.lim_v_lf = pts[0][0]
    mn_lf = np.argmin(np.abs(vel - S.lim_v_lf))
    lo = max(0, mn_lf - 10)
    hi = min(len(S.flux) - 1, mn_lf + 10)
    S.ew_lf = float(np.median(S.flux[lo:hi] / S.sig[lo:hi]))
    main_plot_velocity()
def on_save():
    line = (
        f"{S.name_q[S.file_index]}  "
        f"{S.pair_id[S.file_index]:d}  "
        f"{S.zem_q[S.file_index]:17.5f} "
        f"{S.zemi_cl[S.file_index]:17.5f}  "
        f"{S.lya_T:d} {S.lyb_T:d} {S.lyg_T:d} {S.lyd_T:d} {S.lye_T:d} "
        f"{S.ovi_T:d} {S.cii_T:d} {S.siII_T:d} {S.siIII_T:d} {S.Line_CF:d} "
        f"{S.lim_v1_lya:17.5f} {S.lim_v2_lya:17.5f} "
        f"{S.ew_lya:17.5f} {S.sigew_lya:17.5f} {S.vcent_lya:17.5f} "
        f"{S.lim_v_lf:17.5f} {S.ew_lf:17.5f} {S.ex_T:d}\n"
    )
    S.out_fh.write(line)
    S.out_fh.flush()
    print(f"Saved: {S.name_q[S.file_index]}  pair={S.pair_id[S.file_index]}")
    reset_line_flags()
def on_done(root):
    if messagebox.askyesno("Exit", "EXIT PROGRAM — are you sure?"):
        print(f"Finished at source index {S.file_index}")
        if S.out_fh:
            S.out_fh.close()
        root.destroy()
# ============================================================
#  Catalogue reader
# ============================================================
def read_catalog(read_file, nskip=0):
    rows = []
    with open(read_file, "r", encoding="utf-8") as fh:
        for _ in range(nskip):
            fh.readline()
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    if not rows:
        raise ValueError(f"No data in {read_file} after {nskip} skipped lines.")
    min_cols = 30
    for i, r in enumerate(rows):
        if len(r) < min_cols:
            raise ValueError(
                f"Row {i+nskip+1} has {len(r)} columns (need {min_cols}).\n"
                f"Row: {' '.join(r)}"
            )
    n = len(rows)
    ra_q       = np.array([r[0]  for r in rows], dtype=float)
    dec_q      = np.array([r[1]  for r in rows], dtype=float)
    zem_q      = np.array([r[2]  for r in rows], dtype=float)
    SNR_q      = np.array([r[3]  for r in rows], dtype=float)
    NUM_q      = np.array([r[4]  for r in rows], dtype=int)
    name_q     = np.array([r[5]  for r in rows], dtype=str)
    grism_flg  = np.array([r[6]  for r in rows], dtype=int)
    ra_cl      = np.array([r[7]  for r in rows], dtype=float)
    dec_cl     = np.array([r[8]  for r in rows], dtype=float)
    zemi_cl    = np.array([r[9]  for r in rows], dtype=float)
    M500       = np.array([r[10] for r in rows], dtype=float)
    R500       = np.array([r[11] for r in rows], dtype=float)
    q_cl_vel   = np.array([r[12] for r in rows], dtype=float)
    r_phy      = np.array([r[13] for r in rows], dtype=float)
    r_phy_R500 = np.array([r[14] for r in rows], dtype=float)
    cat        = np.array([r[15] for r in rows], dtype=str)
    pair_id    = np.array([r[16] for r in rows], dtype=int)
    ll_wav     = np.array([r[17] for r in rows], dtype=float)
    ll_flg     = np.array([r[18] for r in rows], dtype=str)
    lya_flg    = np.array([r[19] for r in rows], dtype=str)
    lyb_flg    = np.array([r[20] for r in rows], dtype=str)
    ovia_flg   = np.array([r[21] for r in rows], dtype=str)
    ovib_flg   = np.array([r[22] for r in rows], dtype=str)
    ex500      = np.array([r[28] for r in rows], dtype=int)
    CF_abs     = np.array([r[29] for r in rows], dtype=int)
    dt1 = np.dtype([
        ("ra_q", float), ("dec_q", float), ("zem_q", float),
        ("SNR_q", float), ("NUM_q", int), ("name_q", "U50"),
        ("grism_flg", int), ("ra_cl", float), ("dec_cl", float),
        ("zemi_cl", float), ("M500", float), ("R500", float),
        ("q_cl_vel", float), ("r_phy", float), ("r_phy_R500", float),
        ("cat", "U20"), ("pair_id", int),
    ])
    data1 = np.empty(n, dtype=dt1)
    for k, v in [("ra_q", ra_q), ("dec_q", dec_q), ("zem_q", zem_q),
                 ("SNR_q", SNR_q), ("NUM_q", NUM_q), ("name_q", name_q),
                 ("grism_flg", grism_flg), ("ra_cl", ra_cl), ("dec_cl", dec_cl),
                 ("zemi_cl", zemi_cl), ("M500", M500), ("R500", R500),
                 ("q_cl_vel", q_cl_vel), ("r_phy", r_phy),
                 ("r_phy_R500", r_phy_R500), ("cat", cat), ("pair_id", pair_id)]:
        data1[k] = v
    return data1, ll_wav, ll_flg, lya_flg, lyb_flg, ovia_flg, ovib_flg, ex500, CF_abs
# ============================================================
#  Main entry point
# ============================================================
def HST_abs_visual_check():
    # Ion tables (identical values to IDL)
    S.rwave = np.array([
        1037.616, 1031.927, 937.8035, 949.7431, 972.5368, 1025.7223,
        1215.6701, 977.020, 1206.500, 1334.5323, 1036.3367,
        1260.4221, 1193.2897, 1190.4158,
    ])
    S.rwave_name = np.array([
        "OVIB", "OVIA", "lye", "lyd", "lyg", "lyb", "lya",
        "CIII", "SiIII", "CII1334", "CII1036", "SiII1260", "SiII1193", "SiII1190",
    ])
    S.f_ion_a = np.array([
        0.06609, 0.1329, 0.007799, 0.01394, 0.029, 0.07912, 0.4165,
        0.7620, 1.669, 0.1278, 0.1231, 1.007, 0.4991, 0.2502,
    ])
    S.ll_ion   = 6
    S.dir_name = "/Users/smishra/Dropbox/HSLADR1/Fitted/"
    read_file = "5K_quasar-cl_pairs_with_1160-q-cl_info.txt"
    nskip     = 2 #1200
    (data1, ll_wav, ll_flg, lya_flg, lyb_flg,
     ovia_flg, ovib_flg, ex500, CF_abs) = read_catalog(read_file, nskip)
    S.name_q    = data1["name_q"]
    S.zem_q     = data1["zem_q"]
    S.grism_flg = data1["grism_flg"]
    S.zemi_cl   = data1["zemi_cl"]
    S.pair_id   = data1["pair_id"]
    S.ll_wav    = ll_wav
    S.lya_flg   = lya_flg
    S.lyb_flg   = lyb_flg
    S.ovia_flg  = ovia_flg
    S.ovib_flg  = ovib_flg
    id_pll = np.where(ll_flg == "pll")[0]
    if len(id_pll) > 0:
        S.ll_wav[id_pll] = -9999.999
    id_ana = np.where(
        ((lya_flg == "Lya_In") & (data1["zemi_cl"] > 0.10653731521679899) & (ex500 == -1))
        | (ex500 == 2)
        | ((ex500 == 3) & (CF_abs != 1) & (CF_abs != 2))
    )[0]
    for attr, arr in [("name_q", S.name_q), ("zem_q", S.zem_q),
                      ("grism_flg", S.grism_flg), ("zemi_cl", S.zemi_cl),
                      ("ll_wav", S.ll_wav), ("pair_id", S.pair_id),
                      ("lya_flg", S.lya_flg), ("lyb_flg", S.lyb_flg),
                      ("ovia_flg", S.ovia_flg), ("ovib_flg", S.ovib_flg)]:
        setattr(S, attr, arr[id_ana])
    S.file_index      = 0
    S.N_Spectra_files = len(S.name_q)
    S.out_fh = open("tmp_rnd", "a")
    S.out_fh.write(
        "#qso_name pairid z_q z_cl lya lyb lyg lyd lye ovi cii siII siIII CF "
        "v1_lya v2_lya ew_lya sigew_lya vcent_lya v_lf snr_lf ex\n"
    )
    reset_line_flags()
    # ===========================================================
    #  Build GUI
    # ===========================================================
    root = tk.Tk()
    root.title("HST/COS Absorption Line Visual Inspection")
    root.configure(bg=GUI_BG)
    # ---- Button factory ----
    def mk_btn(parent, text, cmd, bg, abg, w=15):
        b = tk.Button(
            parent,
            text=text,
            command=cmd,
            bg=bg,
            activebackground=abg,
            fg=BTN_FG,
            activeforeground=BTN_FG,
            font=BTN_FONT,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=10,
            width=w,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground="#666666",
        )
        return b
    # ---- Main layout ----
    main_frame = tk.Frame(root, bg=GUI_BG)
    main_frame.pack(fill=tk.BOTH, expand=True)
    left_col = tk.Frame(main_frame, bg=GUI_BG)
    left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    # ---- Embedded plot ----
    S.fig_left = plt.Figure(figsize=(16, 7), dpi=100)
    S.fig_left.patch.set_facecolor(BG)
    S.canvas_left = FigureCanvasTkAgg(S.fig_left, master=left_col)
    S.canvas_left.get_tk_widget().configure(bg=GUI_BG, highlightthickness=0)
    S.canvas_left.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk.Frame(left_col, bg=SEP_COL, height=2).pack(fill=tk.X, pady=2)
    # ---- Row 1: navigation + measurement + save ----
    row1 = tk.Frame(left_col, bg=GUI_BG)
    row1.pack(fill=tk.X, padx=6, pady=8)
    status_var = tk.StringVar(value="")
    def nav(fn):
        def _cb():
            fn()
            status_var.set(
                f"  Source {S.file_index+1} / {S.N_Spectra_files}   |   "
                f"{S.name_q[S.file_index]}   |   "
                f"zcl = {S.zemi_cl[S.file_index]:.5f}"
            )
        return _cb
    mk_btn(row1, "◀  Back",   nav(on_back),    NAV_BG,  NAV_ACT,  w=10).pack(side=tk.LEFT, padx=3)
    mk_btn(row1, "Next  ▶",   nav(on_next),    NAV_BG,  NAV_ACT,  w=10).pack(side=tk.LEFT, padx=3)
    tk.Frame(row1, bg=SEP_COL, width=2).pack(side=tk.LEFT, fill=tk.Y, padx=5)
    mk_btn(row1, "Get  EW",  on_get_lya, ACT_BG,  ACT_ACT,  w=10).pack(side=tk.LEFT, padx=3)
    mk_btn(row1, "Get  LF",   on_get_lf,  ACT_BG,  ACT_ACT,  w=10).pack(side=tk.LEFT, padx=3)
    tk.Frame(row1, bg=SEP_COL, width=2).pack(side=tk.LEFT, fill=tk.Y, padx=5)
    mk_btn(row1, "💾  Save",  on_save,               SAVE_BG, SAVE_ACT, w=10).pack(side=tk.LEFT, padx=3)
    mk_btn(row1, "✕  Done",  lambda: on_done(root),  SAVE_BG, SAVE_ACT, w=10).pack(side=tk.LEFT, padx=3)
    tk.Frame(left_col, bg=SEP_COL, height=1).pack(fill=tk.X, pady=1)
    # ---- Row 2: line classification ----
    row2 = tk.Frame(left_col, bg=GUI_BG)
    row2.pack(fill=tk.X, padx=6, pady=1)
    tk.Label(row2, text="Classify line:",
             bg=GUI_BG, fg="#dddddd",
             font=("Helvetica", fnt_sz+5, "bold")).pack(side=tk.LEFT, padx=(2, 8))
    flag_btns: dict = {}
    def make_flag_btn(parent, label, attr):
        def cb():
            setattr(S, attr, 1)
            for b in flag_btns.values():
                b.configure(bg=FLAG_BG)
            flag_btns[attr].configure(bg=FLAG_LIT)
        b = mk_btn(parent, label, cb, FLAG_BG, FLAG_ACT, w=7)
        b.pack(side=tk.LEFT, padx=2)
        flag_btns[attr] = b
    for lbl, attr in [("Lya", "lya_T"), ("Lyb", "Lyb_T"), ("Lyg", "lyg_T"),
                      ("Lyd", "lyd_T"), ("Lye", "lye_T"), ("OVI", "ovi_T"),
                      ("CII", "cii_T"), ("SiII", "siII_T"), ("SiIII", "siIII_T")]:
        make_flag_btn(row2, lbl, attr)
    tk.Frame(row2, bg=SEP_COL, width=2).pack(side=tk.LEFT, fill=tk.Y, padx=5)
    def set_cf(val):
        S.Line_CF = val
        
    # for lbl, val in [("CF-1", 1), ("CF-2", 2), ("CF-3", 3)]:
    #     mk_btn(row2, lbl, lambda v=val: set_cf(v), CF_BG, CF_ACT, w=6).pack(side=tk.LEFT, padx=2)
        
    tk.Frame(row2, bg=SEP_COL, width=2).pack(side=tk.LEFT, fill=tk.Y, padx=5)
    def on_exclude():
        S.ex_T = 1
    mk_btn(row2, "Exclude", on_exclude, SAVE_BG, SAVE_ACT, w=8).pack(side=tk.LEFT, padx=2)
    # ---- Status bar ----
    tk.Frame(left_col, bg=SEP_COL, height=1).pack(fill=tk.X)
    tk.Label(left_col, textvariable=status_var,
             bg=GUI_BG, fg="#88ccff",
             font=("Courier", fnt_sz, "bold"),
             anchor="w", padx=8, pady=2).pack(fill=tk.X)
    # ---- Initial load ----
    get_spec(S.file_index)
    status_var.set(
        f"  Source {S.file_index+1} / {S.N_Spectra_files}   |   "
        f"{S.name_q[S.file_index]}   |   "
        f"zcl = {S.zemi_cl[S.file_index]:.5f}"
    )
    main_plot_velocity()
    root.mainloop()
if __name__ == "__main__":
    HST_abs_visual_check()
