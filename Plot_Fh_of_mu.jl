#= 
Calculate The partial derivative of F(h, sigma(h)) at the fixed point h

=#


# load libraries
using PyPlot, Printf, Statistics, Random, FFTW, StatsBase,LinearAlgebra, SpecialFunctions, Roots, Distributions
using PyCall, NPZ, UUIDs
py"""
       import sys
       sys.path.insert(0, "./src")
       """
scipy_special = pyimport("scipy_special")       # needed for owen's T function
include("src/sim.jl")
include("src/sim_2.jl")
include("src/demo_functions.jl")


# Parameter set
Ne          = 0             # number excitatory cells
global Ni   = 50000#1000#50000     # number inhibitory cells
Ncells      = Ne + Ni
Nrecord     = 1#Ncells      # number neurons to be recorded

T           = 0.50          # simulation time [s]
dt          = 1e-4          # [s] simulation timestep
dt_record   = dt            # [s] record time step 
dt_meso     = dt_record/10
dt_naive    = dt_meso
alpha       = 2.            # exponent in transfer function
beta        = 5.0         # [mV^-1]steepness of the sigmoidal-errf transfer function
theta       = 0.            # [mV] voltage of maximal increase in sigm-errf transfer function
Phi0        = 0.65          # [s^-1] prefactor of transfer function theta^(-alpha) = (1 mV)^(-alpha) 
Phimax      = 100           # [s^-1] maximal firing rate of neurons
con_ii      = -1.0          # [mVs] connection strength from i to i
con_ie      = 0.            # [mVs] connection strength from e to i
con_ei      = 0.            # [mVs] connection strength from i to e
con_ee      = 0.            # [mVs] connection strength from e to e
p_e         = 0.00          # probability of connection from excitatory neuron
p_i         = 0.2#0.7         # probability of connection from inhibitory neuron
taue        = 0.0 #0.1      # membrane time constant for exc. neurons [s]
taui        = 0.02          # membrane time constant for inh. neurons [s]
taus_e      = 0.00          # synaptic time constant, excitatory [s]
taus_i      = 0.00          # synaptic time constant, inhibitory [s]
mu0         = 25            # input current [mV]
delay_micro = 0.00001       # delay [s], cannot be set to zero (for now)
delay       = 0
shf_yn      = true          # sh(ow) f(igure) yes/no: no for time benchmark

seed        = 5             # seed for initial condition
#Random.seed!(seed)         # set global seed -> same initial conditions each run,
                            # comment out this line if initial conditions v(0) should be different for each run
seed_quenched = 0           # seed for quenched random connectivity, set 0 to be different for each run
nr_trials   = 70            # number of trial runs
T_measure   = 0.24          # time length of each measurement [s]
T_relax     = 0.25          # relaxation time before PSD is measured [s]
n_measure   = trunc(Int, T_measure/dt_record)
micro_yn    = false
meso_yn     = true
naive_yn    = true
C_const     = false         # ...
naive_rescaling_yn = false
linearized_theory_yn = true # toggle to calculate the variances in the linearized system
savedat_yn  = false
savefig_yn  = false
supress_noise_yn            = false
max_search_for_fixed_point = 1000
fixed_in_degree             = false     # only for the microscopic model
save_location               = "data//"
dat_file_name               = ""
source_file_path            = ".jl"
f_max       = 10000
f_min       = 2
num_f       = 1000
mu_min      = -7.
d_mu        = 0.1
mu_max      = 250.0
mu_arr      = mu_min:d_mu:mu_max
Fh_arr_meso = zeros(length(mu_arr))
Fh_arr_naive= zeros(length(mu_arr))
r0_arr_meso = zeros(length(mu_arr))
Fh_arr_appr = zeros(length(mu_arr))
Fh_arr_appr_2 = zeros(length(mu_arr))
r0_arr      = zeros(length(mu_arr))
h0_arr      = zeros(length(mu_arr))
s0_arr      = zeros(length(mu_arr))
r0_arr_app  = zeros(length(mu_arr))
h0_arr_app  = zeros(length(mu_arr))
s0_arr_app  = zeros(length(mu_arr))

n_meso      = round(Int, T_measure/dt_meso)
n_naive     = round(Int, T_measure/dt_naive)
n_micro     = round(Int, T_measure/dt)


hmax        = (Phimax/Phi0)^(1.0/alpha)
round_yn    = false
ini_h_dis   = 0.1#-30#-0.2#0.2

#para_string = "_beta_"*string(beta)*"_theta_"*string(theta)*"_T_"*string(T)*"_dt_"*string(dt)*"dt_meso"*string(dt_meso)*"_dt_naive_"*string(dt_naive)
#para_string = para_string*"_N_"*string(Ncells)*"_phimax_"*string(Phimax)*"_pmin_pmax_dp_"*string(pmin)*"_"*string(pmax)*"_"*string(dp)*"_w_"*string(con_ii)
#para_string = para_string*"_taui_"*string(taui)*"_mu_"*string(mu0)*"_delay_"*string(delay)*"_nrtrials_"*string(nr_trials)
#para_string = para_string*"_Tmeasure_"*string(T_measure)*"_T_relax_"*string(T_relax)*"_supressnoiseyn__"*string(supress_noise_yn)*"_Cconst_"*string(C_const)

para_string = "_p_"*string(p_i)*"_N_"*string(Ni)

save_location               = "data/Plot_Fh_for_paper/"#"data/time_dependent_variance_with_WN/"
dat_file_name               = "Plot_Fh_of_mu"#"time_dependent_variance_with_WN"
source_file_path            = "Plot_Fh_of_mu.jl"

if(savedat_yn)
    filename_uuid = save_simulation_result(save_location, source_file_path, dat_file_name*para_string; get_also_current_time=true)
    total_save_data_name = save_location*filename_uuid*".npz"
end


param_Phi   = Dict("beta"=>beta, "theta"=>theta,"Phimax"=> Phimax)
param_lnp   = Dict("con_ii"=> con_ii, "con_ei"=> con_ei, "con_ie"=> con_ie, "con_ee"=> con_ee, "p_e"=> p_e, "p_i" => p_i,
                    "taue"=> taue, "taui"=> taui, "taus_e"=> taus_e, "taus_i"=> taus_i, "mu0"=> mu0, "delay"=> delay_micro)

param_naive   = Dict("theta"=>theta, "beta"=> beta, "Phimax"=> Phimax, "tmax"=> T, "dt"=> dt_meso, "p"=>1,
                        "w"=>con_ii, "N"=> Ni, "C"=> Ni * p_i, "tau"=>taui, "delta"=>delay, "mu"=> mu0, "round_yn"=>round_yn, "mu0"=>mu0, 
                        "max_search_for_fixed_point"=>max_search_for_fixed_point, "delay"=>delay)


param_meso   = Dict("theta"=>theta, "beta"=> beta, "Phimax"=> Phimax, "tmax"=> T, "dt"=> dt_meso, "p"=>p_i,
                        "w"=>con_ii, "N"=> Ni, "C"=> Ni * p_i, "tau"=>taui, "delta"=>delay, "mu"=> mu0, "round_yn"=>round_yn, "mu0"=>mu0, 
                        "max_search_for_fixed_point"=>max_search_for_fixed_point, "F_smoothed"=> F_sigm_erf, "delFdels" => delFdels_sig_erf, 
                        "phi_prime" => del_sigm_erf_del_h, "delay"=>delay)

h_arr = -20:0.1:20
sig_arr     = con_ii * (1 - p_i)/(2*taui*Ni*p_i) .* (h_arr .- mu0)
F_arr       = F_sigm_erf(h_arr, abs.(sig_arr), param_Phi)
lin_arr     = (h_arr .- mu0)./con_ii




mu_of_inflection_point = abs(con_ii)*Phimax/2

for i=1:length(mu_arr)
    param_meso["mu"]    = mu_arr[i]
    param_meso["mu0"]   = mu_arr[i]
    param_naive["mu0"]  = mu_arr[i]
    param_naive["mu"]   = mu_arr[i]
    global h0, s0, r0, fix_found = PSD_get_one_fix_point(param_meso, false)
    global X0 = 0
    global ini_h_dis = h0
    global ini_sig = s0
    global h0n, s0n, r0n, fix_foundn = PSD_get_one_fix_point(param_naive, true)
    s0n=0
    X0n =0
    global X0 = 0
    Fh_arr_meso[i]  = delFdelh_sig_erf(h0, s0, param_meso)
    r0_arr_meso[i]  = r0
    Fh_arr_naive[i] = delFdelh_sig_erf(h0n, 0, param_naive)
    Fh_arr_appr[i]  = Fh_approxed_FP(param_meso)
    r0_arr[i]       = r0
    h0_arr[i]       = h0
    s0_arr[i]       = s0
    h0_app, s0_app, r0_app = FP_approxed(param_meso)
    r0_arr_app[i]   = r0_app
    h0_arr_app[i]   = h0_app
    s0_arr_app[i]   = s0_app
    Fh_arr_appr_2[i]= delFdelh_sig_erf(h0_app, s0_app, param_meso)
end

# calculate mu_0_inflection_point
mu_infl = 0.5 * abs(con_ii) * Phimax
mu_max_ind = argmax(Fh_arr_meso[70:end])
Fh_meso_max =  Fh_arr_meso[mu_max_ind]
Fh_arr_meso_normed = Fh_arr_meso ./ Fh_meso_max

param_meso["mu"] = mu_infl
param_meso["mu0"]= mu_infl
h0, s0, r0, fix_found = PSD_get_one_fix_point(param_meso, false)
Fh_mu_infl = delFdelh_sig_erf(h0, s0, param_meso)

betahat = -con_ii*Phimax*beta
muhat  = mu_arr./(-con_ii*Phimax)
Fh_hat_arr = zeros(length(mu_arr))
for i=1:length(mu_arr)
    Fh_hat_arr[i] = Fh_hat_in_scaling(muhat[i], betahat, taui, Ni, p_i)
end

#=
figure(1)
#labelstr = L"$\beta$ ="*string(beta)*L" mV$^{-1}$"
labelstr = L"$p$ ="*string(p_i)
#plot(mu_arr, Fh_arr_meso_normed, label=labelstr)
plot(mu_arr, -con_ii*Fh_arr_meso/sqrt(Ni*p_i)   , label=labelstr)
plot(mu_arr, Q_exp_sig_erf_inv.(mu_arr ./ (-con_ii*Phimax)))
axvline(x=mu_infl, color="grey", linestyle="dashed")
xlabel(L"$\mu$ [mV]")
ylabel(L"$F_h$ [(mVs)$^{-1}$](normed)")
legend()

figure(2)
#labelstr2 = L"$\beta$ ="*string(beta)*L" mV$^{-1}$"
labelstr2 = L"$p$ ="*string(p_i)
plot(mu_arr, r0_arr .* Fh_arr_meso/sqrt(Ni*p_i), label=labelstr2)
axvline(x=mu_infl, color="grey", linestyle="dashed")
xlabel(L"$\mu$ [mV]")
ylabel(L"$F_h \cdot r_0$ [(mVs)$^{-1}$]")
legend()



figure(3)

#labelstr2 = L"$\beta$ ="*string(beta)*L" mV$^{-1}$"
labelstr2 = L"$p$ ="*string(p_i)
labelstr3 = L"$p$ ="*string(p_i)*" appr."
plot(muhat, Fh_hat_arr, label=labelstr2)
plot(muhat, sqrt(2*pi*con_ii^2/betahat^2)*Fh_arr_meso, label=labelstr3)
axvline(x=0.5, color="grey", linestyle="dashed")
xlabel(L"$\mu$ [mV]")
ylabel(L"$F_h \cdot$ [(mVs)$^{-1}$]")
legend()

figure(4)

#labelstr2 = L"$\beta$ ="*string(beta)*L" mV$^{-1}$"
labelstr2 = L"$p$ ="*string(p_i)
labelstr3 = L"$p$ ="*string(p_i)*" appr."
plot(muhat, Fh_hat_arr.*(muhat), label=labelstr2)
plot(muhat, sqrt(2*pi*con_ii^2/betahat^2)*Fh_arr_meso .*r0_arr ./ Phimax, label=labelstr3)
axvline(x=0.5, color="grey", linestyle="dashed")
xlabel(L"$\mu$ [mV]")
ylabel(L"$F_h \cdot$ [(mVs)$^{-1}$]")
legend()
=#
if(savedat_yn)
    NPZ.npzwrite(total_save_data_name, Dict("mu_arr"=>mu_arr, "muhat" => muhat, "Fh_arr_meso"=>Fh_arr_meso, "Fh_hat_arr"=> Fh_hat_arr, "r0_arr"=>r0_arr))
end
