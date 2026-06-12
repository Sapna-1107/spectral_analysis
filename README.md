# spectral_analysis

Python and IDL codes for UV absorption-line spectral analysis of HST/COS spectra, used in **Mishra et al. (2024a)** and **Mishra et al. (2024b)**.

---

## Repository Contents

### `HST_abs_visual_check.py`
Interactive Python/tkinter GUI tool (translated from IDL) for visually inspecting HST/COS UV absorption spectra of quasar–cluster sightlines. Loads co-added spectra and continuum fits from FITS files, displays normalized velocity profiles for 14 ions (Lyα through Lyε, OVI doublet, CII, SiII, SiIII) simultaneously, and computes apparent optical depth (N_app) profiles for the Lyman series and OVI. The user can navigate between sightlines, interactively click to set absorption-line velocity limits, measure equivalent widths and centroid velocities, classify detected lines, and save results to a catalogue file.

---

### `check_contamination.py`
Plots normalized velocity profiles of multiple UV absorption lines (low ions, Si II, C II, Si IV, C IV, etc.) for individual LMC sightlines, overplotting Voigt-profile fits read from VPFIT output files. Computes apparent optical depth (AOD) profiles and applies LSR velocity corrections. Used to check whether absorption features at the cluster redshift are genuinely associated or are contaminants from unrelated intervening systems.

---

### `cos_coadd_correction.py`
Determines the optimal wavelength-division point when co-adding HST/COS spectra from multiple gratings and FUSE channels (G130M, G160M, LIF1, LIF2, SIC2A). Finds the crossover wavelength in overlapping regions by locating where the signal-to-noise ratio curves of adjacent segments intersect, then splices the segments together at that point to produce a single, SNR-optimised co-added spectrum.

---

### `master_stack_13-07-2023.pro`
IDL procedure for stacking (co-adding) continuum-normalised HST/COS spectra in velocity space. Reads a catalogue of quasar–cluster sightlines, shifts each spectrum to the cluster rest frame, and combines them using multiple stacking estimators (median, mean, SNR-weighted mean, sigma-clipped mean) to produce a composite absorption profile.

---

### `match_component_rdgen.py`
Plots multi-ion velocity profiles for individual sightlines, overplotting Voigt-profile components from VPFIT fits. Computes AOD profiles for the Lyman series. Also includes a diagnostic display (`plot_expected_lines_ver1`) that shows all Lyman and OVI transitions simultaneously with absorption velocity limits and centroid marked, used to verify line identifications and component matching across ions.

---

### `simulate_spectra_lya_23-06-23.py`
Monte Carlo simulation framework for assessing systematic effects in Lyα equivalent-width measurements from spectral stacks. Generates large numbers of mock HST/COS spectra with realistic noise (drawing line parameters from the Danforth et al. Lyα catalogue), injects Lyα absorbers at a specified covering fraction and velocity dispersion, stacks the mock spectra using median, mean, SNR-weighted mean, and sigma-clipped mean estimators, and compares the recovered equivalent width against the input value using bootstrap uncertainties. Used to characterise the SNR-dependent bias and scatter of each stacking method.

---

### `simulation_effect.pdf`
Output figure from `simulate_spectra_lya_23-06-23.py` showing the simulated stacked Lyα profiles and the comparison between recovered and expected equivalent widths across different SNR bins.

---

## References

- Mishra et al. (2024a)
- Mishra et al. (2024b)
