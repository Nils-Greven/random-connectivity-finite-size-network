# random-connectivity-finite-size-network
Repository for Greven, N. E., Ranft, J., & Schwalger, T. (2026). _How random connectivity shapes the fluctuating dynamics of finite-size neural populations_. PRX Life, 4(1), 013007. https://doi.org/10.1103/shvm-x4x6

The code provided will help recreate the data for the figures of the paper. In the following will be a short instruction which code needs to be run.
I might add more instructions sometime later.
I might clean up the code sometime later as for now it will contain all the code I have written for the project, regardless of whether it was used in the paper,
some of the code might be redundant. Variable names have changed during the project, some might not be consistent with the parameter names in the paper. 

In general the following functions have been used for the simulation of the different models:
- quenched network simulations: `micro_simulation`
- annealed microsopic network: `Micro_Simulation_Annealed_fixedindegree`
- MF1 and MF2: `mesoscopic_model_correction_colored_noise_time_dep_stimulus`
- MFSparselim has been taken with the same function as MF1/2 but for large $N$ and a correspondingly small $p$

You can explore my implementation of these models via `Demo_of_different_models_for_paper.jl`

## Fig 1.

## Fig 2.
schematic and purely illustrative

## Fig 3.
a) Can be calculated via `F_sigm_erf` for instance in `Plot_Fh_of_mu.jl`

b) Calculated via `onset_of_oscillation_sig_erf_vary_any_two_parameter.jl`

c) Trajectories can be directly calculated with the functions mentioned above

d-f) Can be calculated via `Susceptibility_with_WN_modulation.jl`

## Fig 4.

## Fig 5.
- `var_annealed_several_p.jl`:
    parameters inside the file need to be adapted to the values of the figure
    - `micro_yn = true` for the quenched network
    - `micro_annealed_yn = true` for the annealed network
    - `meso_yn     = true` for MF2
    - `naive_yn    = true` for MF1

## Fig 6.

## Fig 7.

## Fig 8.
`wlim_color_correction.jl` has been used
