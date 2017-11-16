# GPSpec
Interpolating spectra with GPs

## Contents:

- **soft**: Python code

- **data**: data used in the tests

- **plots**: plots produced by the code on the test data

## More details on the code:

  - `GPS_utils.py`: where all the most up-to-date code currently
    lives. There are two top-level routines: `GPSpec_1Comp` and
    `GPSpec_2Comp`, for modelling single- and double-lined spectra,
    respectively. The single-lined version uses `celerite` out of the
    box, and fits for the hyper parameters of the GP and the relative
    velocity shifts at the same time. The double-lined version uses
    `celerite` with Dan F-M's conjugate gradient trick, and first
    first for the GP HPs on individual spectra, then fixes them and
    fits a set of relative velocity shifts for each component. Both
    routines take 2-D arrays of wavelength, fluxes and flux
    uncertainties as input, where the first index represents epoch and
    the second pixel number, plus some optional keyword arguments that
    control things like the length of the MCMC chains, the format of
    the output plots and the names of the files they are saved in. As
    well as graphical output, the code prints the best-fit velocities,
    and associated errors (from the MCMC) to the screen. The little
    test routines `test1` and `test2` illustrate how to use them on
    single- and double-lined synthetic datasets produced by the two
    routines below. The code for the double-lined case is not quite
    stable yet, so there are no plots saved, but the single-lined case
    is ok, and the corresponding plots are in `../plots/synth2*.png`

  - `simulate_dataset.py`: generates a time-series of single-lined
    spectra with random velocity shifts. Produced files
    `synth_dataset_001.mat` and `synth_dataset_002.mat` in `data`
    directory.

  - `simulate_dataset_2stars.py`: same as `synth_dataset.py`, but for
    double-lined spectra. Produced file `synth_dataset_003.mat` in
    data directory.

The rest of the files in this directory, listed below for
completeness, represents earlier stages of the code development,
before I was using `celerite`. It may become useful at a later stage
to re-visit some of the things I was trying out then, including
splitting the wavelength range into segments and evaluating the
likelihood separately for each segment, particularly if we encounter
situations where `celerite` can't be used and we have to revert to
`george`.

  - `test_simulated.py`: the very first go I had at doing this,
    including simulating spectra, degrading their resolution, and then
    trying to model them.
	
  - `test2_HD127334.py`: example code modelling HARPS-N spectra of
     HD127334. Works by modelling all the spectra as a single
     realisation of a Matern32 GP with free log wavelength shifts
     between epochs. Uses `george`, but to make it run reasonably fast
     I model only the data in the absorption lines (where the flux is
     <0.9 of the continuum. Works with subset of spectra in
     `data/HD127334_HARPSN` directory. Produces plots
     `rollsRoyce_spec.png`, `rollsRoyce_chain.png`,
     `rollsRoyce_triangle.png` in plots directory, and prints output
     to screen.

  - `test3_synth.py`: like `test2_...`, but for the synthetic
    single-lined dataset. Doesn't use the "in absorption line" trick,
    but models only 5 spectra. Produces plots
    `rollsRoyce_synth_spec.png`, `rollsRoyce_synth_chain.png`,
    `rollsRoyce_synth_triangle.png` in plots directory, and prints
    output to screen.

  - `test4_synth.py`: like `test3_...`, but breaks down the spectra
    into small segments and combines the results afterwards. Intended
    only as a computational speed test so no graphical output.

  - `test5_synth.py`: like `test4_...`, but breaks down the spectra
    into small segments but the individual likelihoods from the
    segments are combined into a global likelihood (rather than
    averaging the best-fit velocity shifts). Intended
    only as a computational speed test so no graphical output.

  - `test6_synth.py`: like `test5_...`, but included MCMC. Also ran on
    `synth_dataset_002` rather than `001`.

