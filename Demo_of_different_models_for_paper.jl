#= 
Demo of the different models used in the paper
=#

using PyPlot, Printf, Statistics, Random, FFTW, StatsBase,LinearAlgebra, SpecialFunctions, Roots, Distributions
using PyCall, NPZ, UUIDs
py"""
       import sys
       sys.path.insert(0, "./src")
       """
scipy_special = pyimport("scipy_special")
include("src/sim.jl")
include("src/sim_2.jl")
include("src/demo_functions.jl")

Ne          = 0             # number excitatory cells, in the paper always zero
global Ni   = 1000          # number inhibitory cells
Ncells      = Ne + Ni
Nrecord     = 1#Ncells      # number neurons to be recorded

T           = 10.0          # simulation time [s]
dt          = 1e-5          # [s] simulation timestep
dt_record   = dt            # [s] record time step 
dt_meso     = dt_record/10
dt_naive    = dt_meso
alpha       = 2.            # exponent in transfer function, obsolete
beta        = 5             # [mV^-1]steepness of the sigmoidal-errf transfer function
theta       = 0.            # [mV] voltage of maximal increase in sigm-errf transfer function
Phi0        = 0.65          # [s^-1] prefactor of transfer function theta^(-alpha) = (1 mV)^(-alpha), obsolete 
Phimax      = 100           # [s^-1] maximal firing rate of neurons
con_ii      = -1            # [mVs] connection strength from i to i, negative for inhibitory connection!
con_ie      = 0.            # [mVs] connection strength from e to i, in paper always zero
con_ei      = 0.            # [mVs] connection strength from i to e, in paper always zero
con_ee      = 0.            # [mVs] connection strength from e to e, in paper always zero
p_e         = 0.000         # probability of connection from excitatory neuron, in paper always zero
p_i         = 0.1           # probability of connection from inhibitory neuron
taue        = 0.0           # membrane time constant for exc. neurons [s], in paper always zero
taui        = 0.02          # membrane time constant for inh. neurons [s]
taus_e      = 0.00          # synaptic time constant, excitatory [s], in paper always zero
taus_i      = 0.00          # synaptic time constant, inhibitory [s], in paper always zero
mu0         = 10            # input current [mV]
delay_micro = 0.00001       # delay [s], cannot be set to zero (for now)
delay       = 0             # delay [s] for MF1 and MF2

seed        = 5             # seed for initial condition, obsolete
#Random.seed!(seed)         # set global seed -> same initial conditions each run,
                            # comment out this line if initial conditions v(0) should be different for each run
seed_quenched = 0           # seed for quenched random connectivity, set 0 to be different for each run
nr_trials   = 70            # number of trial runs
T_measure   = 0.24          # time length of each measurement [s]
T_relax     = 0.25          # relaxation time before PSD is measured [s]
n_measure   = trunc(Int, T_measure/dt_record)
micro_yn    = true          # boolean: simulate microscopic quenched network? Yes/No
micro_annealed_yn = false   # boolean: simulate microscopic annealed network? Yes/No
meso_yn     = false         #
naive_yn    = false
comparemode = true          # compare the elements of the Gamma matrix
C_const     = true        # ...
J_const     = false
just_theory = false
naive_rescaling_yn = false
linearized_theory_yn = true # toggle to calculate the variances in the linearized system
savedat_yn  = true
savefig_yn  = false
supress_noise_yn            = false
max_search_for_fixed_point = 10000000
fixed_in_degree             = false     # only for the microscopic model
sparse_limit_yn = false                  # use sparse limit for fixed point calculation in theory part
pmin                        = 0.01#0.001#0.0001#0.01
pmax                        = 1
dp                          = 0.01#0.001#0.05#0.0001#0.001
mu_threshold                = 0.25
mu_const                    = 0	
mu_a                        = 1
mu_om                       = 10
mu_phi                      = 0
mu_t_jump_micro             = dt
mu_dt_micro                 = dt
mu_t_jump                   = dt_meso
mu_dt                       = dt_meso
mu_func         = mu_func_0
fixed_in_degree_annaeled   = false         # for the annealed model, otherwise the code will be very slow, the microscopic model remains as is
record_skew_h_i             = false

n_meso      = round(Int, T_measure/dt_meso)
n_naive     = round(Int, T_measure/dt_naive)
n_micro     = round(Int, T_measure/dt)

n_relax_micro   = round(Int, T_relax/dt)
n_relax_naive   = round(Int, T_relax/dt_naive)
n_relax_meso    = round(Int, T_relax/dt_meso)


hmax        = (Phimax/Phi0)^(1.0/alpha)
round_yn    = false
ini_h_dis   = 0.1#-30#-0.2#0.2
ini_sig     = 0

para_string = "_tau_"*string(taui)*"_N_"*string(Ni)*"_Const_"*string(C_const)*"_J_const_"*string(J_const)
if(just_theory)
    para_string = para_string*"just_theory"
end

save_location               = "data/"
dat_file_name               = "Demo_trajectories"
source_file_path            = "Demo_of_different_models_for_paper.jl"

if(savedat_yn)
    filename_uuid = save_simulation_result(save_location, source_file_path, dat_file_name*para_string; get_also_current_time=true)
    total_save_data_name = save_location*filename_uuid*".npz"
end

C = round(Int, Ni * p_i)
if(C_const)
    C0  = C
    Ni0 = C
end
J = con_ii/C
if(J_const)
    J0 = J
    con_ii0 = con_ii
end


global param_Phi   = Dict("beta"=>beta, "theta"=>theta,"Phimax"=> Phimax)
global param_lnp   = Dict("con_ii"=> con_ii, "con_ei"=> con_ei, "con_ie"=> con_ie, "con_ee"=> con_ee, "p_e"=> p_e, "p_i" => p_i,
                    "taue"=> taue, "taui"=> taui, "taus_e"=> taus_e, "taus_i"=> taus_i, "mu0"=> mu0, "delay"=> delay_micro, "mu_dt" => mu_dt_micro, "mu_t_jump"=>mu_t_jump_micro, "mu_threshold"=> mu_threshold,
                        "mu_const"=>mu_const, "mu_a"=>mu_a, "mu_phi"=>mu_phi, "mu_om"=>mu_om, "dummy"=>[1,2,3], "sparse_limit_yn"=> sparse_limit_yn)

global param_meso_corrected = Dict("theta"=>theta, "beta"=> beta, "Phimax"=> Phimax, "tmax"=> T, "dt"=> dt_meso, "p"=>p_i,
                        "w"=>con_ii, "N"=> Ni, "C"=> Ni * p_arr[i], "tau"=>taui, "delta"=>delay, "mu"=> mu0, "round_yn"=>round_yn, "mu0"=>mu0, "max_search_for_fixed_point"=>max_search_for_fixed_point,
                        "F_smoothed"=> F_sigm_erf, "delFdels" => delFdels_sig_erf, "phi_prime" => del_sigm_erf_del_h, "mu_dt" => mu_dt, "mu_t_jump"=>mu_t_jump, "mu_threshold"=> mu_threshold,
                        "mu_const"=>mu_const, "mu_a"=>mu_a, "mu_phi"=>mu_phi, "mu_om"=>mu_om, "sparse_limit_yn"=>sparse_limit_yn)

if(C_const)
    global param_meso_naive = Dict("theta"=>theta, "beta"=> beta, "Phimax"=> Phimax, "tmax"=> T, "dt"=> dt_meso, "p"=>1,
                        #"w"=>con_ii, "N"=> Ni0, "C"=> Ni0, "tau"=>taui, "delta"=>delay, "mu"=> mu0, "round_yn"=>round_yn, "mu0"=>mu0, "max_search_for_fixed_point"=>max_search_for_fixed_point,
                        "w"=>con_ii, "N"=> Ni0/p_i, "C"=> Ni0, "tau"=>taui, "delta"=>delay, "mu"=> mu0, "round_yn"=>round_yn, "mu0"=>mu0, "max_search_for_fixed_point"=>max_search_for_fixed_point,
                        "F_smoothed"=> F_sigm_erf, "delFdels" => delFdels_sig_erf, "phi_prime" => del_sigm_erf_del_h, "mu_dt" => mu_dt, "mu_t_jump"=>mu_t_jump, "mu_threshold"=> mu_threshold,
                        "mu_const"=>mu_const, "mu_a"=>mu_a, "mu_phi"=>mu_phi, "mu_om"=>mu_om, "sparse_limit_yn"=> sparse_limit_yn)
else
    global param_meso_naive = Dict("theta"=>theta, "beta"=> beta, "Phimax"=> Phimax, "tmax"=> T, "dt"=> dt_meso, "p"=>1,
                        "w"=>con_ii, "N"=> Ni, "C"=> Ni, "tau"=>taui, "delta"=>delay, "mu"=> mu0, "round_yn"=>round_yn, "mu0"=>mu0, "max_search_for_fixed_point"=>max_search_for_fixed_point,
                        "F_smoothed"=> F_sigm_erf, "delFdels" => delFdels_sig_erf, "phi_prime" => del_sigm_erf_del_h, "mu_dt" => mu_dt, "mu_t_jump"=>mu_t_jump, "mu_threshold"=> mu_threshold,
                        "mu_const"=>mu_const, "mu_a"=>mu_a, "mu_phi"=>mu_phi, "mu_om"=>mu_om, "sparse_limit_yn"=>sparse_limit_yn)
end


global h0, s0, r0, fix_found = PSD_get_one_fix_point(param_meso_corrected, false)
global h0_sparse, s0_sparse, r0_sparse, fix_found_sparse = PSD_get_one_fix_point_MFsparselim_hypothetical(param_meso_corrected, false) # "Pseudo-values" !!!
global X0 = 0
global ini_h_dis = h0
global ini_sig = s0

if(micro_annealed_yn)
    global micro_annealed_res = Micro_Simulation_Annealed_fixedindegree(T, dt, Ne, Int(Ni), Nrecord, dt_record, Phi_sigm_erf, param_Phi, param_lnp, seed_quenched,fixed_in_degree=fixed_in_degree_annaeled, ini_v = ini_h_dis,
                                                     record_mean_h_i=true, ini_sig=ini_sig, record_var_h_i=true, mu_func = mu_func, record_skew_h_i=record_skew_h_i)
end

if(micro_yn)
    global micro_res = micro_simulation(T, dt, Ne, Int(Ni), Nrecord, dt_record, Phi_sigm_erf, param_Phi, param_lnp, seed_quenched,fixed_in_degree=fixed_in_degree, ini_v = ini_h_dis,
                                                     record_mean_h_i=true, ini_sig=ini_sig, record_var_h_i=true, mu_func = mu_func, record_skew_h_i=record_skew_h_i)
end

if(meso_yn)
    global meso_corrected_res = mesoscopic_model_correction_colored_noise_time_dep_stimulus(h0, s0, X0, F_sigm_erf, G_sigm_erf, mu_func, param_meso_corrected, model3d=1)
end

if(naive_yn)
    global h0n, s0n, r0n, fix_foundn = PSD_get_one_fix_point(param_meso_naive, true)
    global X0n=0
    global naive_corrected_res = mesoscopic_model_correction_colored_noise_time_dep_stimulus(h0n, 0.0, 0.0, F_sigm_erf, G_sigm_erf, mu_func, param_meso_naive, naive_yn=naive_yn)
end





if(savedat_yn)
    #NPZ.npzwrite(total_save_data_name, Dict())
end
