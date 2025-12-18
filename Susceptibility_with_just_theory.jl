using PyPlot, Printf, Statistics, Random, FFTW, StatsBase,LinearAlgebra, SpecialFunctions, Roots, Distributions
using PyCall, NPZ, UUIDs
py"""
       import sys
       sys.path.insert(0, "./src")
       """
scipy_special = pyimport("scipy_special")
WN_generator = pyimport("WN_generator")
include("src/sim.jl")
include("src/sim_2.jl")
include("src/demo_functions.jl")

Ne          = 0             # number excitatory cells
global Ni   = 1e9         # number inhibitory cells
Ncells      = Ne + Ni
Nrecord     = 1#Ncells      # number neurons to be recorded

T           = 0.9         # simulation time [s]
dt          = 1e-4          # [s] simulation timestep
dt_record   = dt            # [s] record time step 
dt_meso     = dt_record/10
dt_naive    = dt_meso
alpha       = 2.            # exponent in transfer function
beta        = 5.0           # [mV^-1]steepness of the sigmoidal-errf transfer function
theta       = 0.            # [mV] voltage of maximal increase in sigm-errf transfer function
Phi0        = 0.65          # [s^-1] prefactor of transfer function theta^(-alpha) = (1 mV)^(-alpha) 
Phimax      = 100           # [s^-1] maximal firing rate of neurons
con_ii      = -1           # [mVs] connection strength from i to i
con_ie      = 0.            # [mVs] connection strength from e to i
con_ei      = 0.            # [mVs] connection strength from i to e
con_ee      = 0.            # [mVs] connection strength from e to e
p_e         = 0.000         # probability of connection from excitatory neuron
p_i         = 950/Ni#0.9         # probability of connection from inhibitory neuron
taue        = 0.0 #0.1      # membrane time constant for exc. neurons [s]
taui        = 0.02          # membrane time constant for inh. neurons [s]
taus_e      = 0.00          # synaptic time constant, excitatory [s]
taus_i      = 0.00          # synaptic time constant, inhibitory [s]
mu0         = 10            # input current [mV]
delay_micro = 0.00001       # delay [s], cannot be set to zero (for now)
delay       = 0.0#0
shf_yn      = true          # sh(ow) f(igure) yes/no: no for time benchmark

seed        = 5             # seed for initial condition
#Random.seed!(seed)         # set global seed -> same initial conditions each run,
                            # comment out this line if initial conditions v(0) should be different for each run
seed_quenched = 0           # seed for quenched random connectivity, set 0 to be different for each run
nr_trials   = 2000          # number of trial runs
T_measure   = 2.0           # time length of each measurement [s]
T_relax     = 0.1           # relaxation time before PSD is measured [s]
micro_yn    = false
micro_annealed_yn = false
meso_yn     = true
naive_yn    = true
C_const     = false         # ...
naive_rescaling_yn = false
linearized_theory_yn = false # toggle to calculate the variances in the linearized system
linear_naive_sim_yn = false
savedat_yn  = true
savefig_yn  = false
supress_noise_yn            = false
max_search_for_fixed_point = 1000
fixed_in_degree             = false     # only for the microscopic model
save_location               = "data/variance/"
dat_file_mame               = "variance_data_vs_p"
pmin                        = 0.01
pmax                        = 1
dp                          = 0.01
mu_threshold                = 0.25
mu_const                    = 0	
mu_a                        = 0.1
mu_om                       = 10
mu_phi                      = 0
mu_t_jump_micro             = T_relax
mu_dt_micro                 = dt
mu_t_jump                   = T_relax
mu_dt                       = dt_meso
mu_func                     = mu_func_WN
mu_sigma                    = 1                 # [Hz^2]
mu_f_max                    = 10000
model3d                     = 1
record_skew_h_i             = false
annealed_inefficient        = false

om_min      = 0.1
om_max      = 100
d_om        = 0.1

############################

#n_meso      = round(Int, T_measure/dt_meso)
#n_naive     = round(Int, T_measure/dt_naive)
#n_micro     = round(Int, T_measure/dt)


hmax        = (Phimax/Phi0)^(1.0/alpha)
round_yn    = false
ini_h_dis   = 0.1#-30#-0.2#0.2
ini_sig     = 0

para_string = "_eps_"*string(mu_a)*"_"
if(meso_yn)
    para_string*= "meso_"
end
if(micro_annealed_yn)
    para_string*= "ann_"
end
if(micro_yn)
    para_string*="micro_"
end
if(naive_yn)
    para_string*="naive_"
end

save_location               = "data/susceptibility_with_WN_modulation/"
dat_file_name               = "susceptibility_with_just_theory"
source_file_path            = "Susceptibility_with_just_theory.jl"

if(savedat_yn)
    filename_uuid = save_simulation_result(save_location, source_file_path, dat_file_name*para_string; get_also_current_time=true)
    total_save_data_name = save_location*filename_uuid*".npz"
end

param_Phi   = Dict("beta"=>beta, "theta"=>theta,"Phimax"=> Phimax)
param_lnp   = Dict("con_ii"=> con_ii, "con_ei"=> con_ei, "con_ie"=> con_ie, "con_ee"=> con_ee, "p_e"=> p_e, "p_i" => p_i,
                    "taue"=> taue, "taui"=> taui, "taus_e"=> taus_e, "taus_i"=> taus_i, "mu0"=> mu0, "delay"=> delay_micro, "mu_dt" => mu_dt_micro, "mu_t_jump"=>mu_t_jump_micro, 
                        "mu_threshold"=> mu_threshold, "mu_const"=>mu_const, "mu_a"=>mu_a, "mu_phi"=>mu_phi, "mu_om"=>mu_om, "dt"=>dt, "mu_sigma"=>mu_sigma, "mu_f_max"=> mu_f_max, 
                        "dummy"=>[1,2,3])


param_meso_corrected   = Dict("theta"=>theta, "beta"=> beta, "Phimax"=> Phimax, "tmax"=> T, "dt"=> dt_meso, "p"=>p_i,
                        "w"=>con_ii, "N"=> Ni, "C"=> Ni * p_i, "tau"=>taui, "delta"=>delay, "mu"=> mu0, "round_yn"=>round_yn, "mu0"=>mu0, 
                        "max_search_for_fixed_point"=>max_search_for_fixed_point, "F_smoothed"=> F_sigm_erf, "delFdels" => delFdels_sig_erf, "phi_prime" => del_sigm_erf_del_h, 
                        "mu_dt" => mu_dt, "mu_t_jump"=>mu_t_jump, "mu_threshold"=> mu_threshold, "mu_const"=>mu_const, "mu_a"=>mu_a, "mu_phi"=>mu_phi, "mu_om"=>mu_om, "delay"=>delay,
                        "mu_sigma"=> mu_sigma, "mu_f_max"=>mu_f_max)

param_meso_naive      = Dict("theta"=>theta, "beta"=> beta, "Phimax"=> Phimax, "tmax"=> T, "dt"=> dt_meso, "p"=>1,
                        "w"=>con_ii, "N"=> Ni, "C"=> Ni, "tau"=>taui, "delta"=>delay, "mu"=> mu0, "round_yn"=>round_yn, "mu0"=>mu0, 
                        "max_search_for_fixed_point"=>max_search_for_fixed_point, "F_smoothed"=> F_sigm_erf, "delFdels" => delFdels_sig_erf, "phi_prime" => del_sigm_erf_del_h, 
                        "mu_dt" => mu_dt, "mu_t_jump"=>mu_t_jump, "mu_threshold"=> mu_threshold, "mu_const"=>mu_const, "mu_a"=>mu_a, "mu_phi"=>mu_phi, "mu_om"=>mu_om, "delay"=>delay,
                        "mu_sigma"=>mu_sigma, "mu_f_max"=>mu_f_max)

n_meso  = round(Int, (T-T_relax)/dt_meso)
n_micro = round(Int, (T-T_relax)/dt)

f = 1:1:50000

global h0, s0, r0, fix_found = PSD_get_one_fix_point(param_meso_corrected, false)
global X0 = 0
global ini_h_dis = h0
global ini_sig = s0
global h0n, s0n, r0n, fix_foundn = PSD_get_one_fix_point(param_meso_naive, true)
s0n=0
X0n =0
global X0 = 0

if(meso_yn)
    chi_r_meso_the = complex(zeros(length(f)))
    for i=1:length(f)
        chi_r_meso_the[i] = give_meso_susceptibility_r(2*pi*f[i], param_meso_corrected, naive_yn=false)
    end
end

if(naive_yn)
    chi_r_naive_the = complex(zeros(length(f)))
    for i=1:length(f)
        chi_r_naive_the[i] = give_meso_susceptibility_r(2*pi*f[i], param_meso_naive, naive_yn=true)
    end
end

if(micro_annealed_yn)
    chi_ann = 2*param_lnp["mu_f_max"] .* cpsd_ann ./ param_lnp["mu_sigma"]
end

if(micro_annealed_yn)
    chi_micro = 2*param_lnp["mu_f_max"] .* cpsd_micro ./ param_lnp["mu_sigma"]
end

figure(1)
plot(f, abs.(chi_r_meso_the), "--", color="blue")
plot(f, abs.(chi_r_naive_the), "--", color="orange")
xscale("log")
yscale("log")
xlabel(L"$\omega$ [Hz]")
ylabel(L"$\vert \tilde \chi_r \vert$")

figure(2)
plot(f, angle.(chi_r_meso_the), "--", color="blue")
plot(f, angle.(chi_r_naive_the), "--", color="orange")
xscale("log")
xlabel(L"$\omega$ [Hz]")
ylabel(L"$\phi_r$")

if(savedat_yn)
    npzwrite(total_save_data_name, Dict("f_meso"=>f, "chi_r_meso_the"=>chi_r_meso_the, "chi_r_naive_the"=>chi_r_naive_the))
end
