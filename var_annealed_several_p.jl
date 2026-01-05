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

Ne          = 0             # number excitatory cells [obsolete and not used in the paper, kept in to avoid accidental errors in github]
global Ni   = 1000          # number inhibitory cells
Ncells      = Ne + Ni
Nrecord     = 1#Ncells      # number neurons to be recorded

T           = 10.0#1.0          # simulation time [s]
dt          = 1e-5          # [s] simulation timestep
dt_record   = dt            # [s] record time step 
dt_meso     = dt_record/10
dt_naive    = dt_meso
alpha       = 2.            # exponent in transfer function  [obsolete and not used in the paper, kept in to avoid accidental errors in github]
beta        = 5#5.0           # [mV^-1]steepness of the sigmoidal-errf transfer function
theta       = 0.            # [mV] voltage of maximal increase in sigm-errf transfer function
Phi0        = 0.65          # [s^-1] prefactor of transfer function theta^(-alpha) = (1 mV)^(-alpha)  [obsolete and not used in the paper, kept in to avoid accidental errors in github]
Phimax      = 100           # [s^-1] maximal firing rate of neurons
con_ii      = -1            # [mVs] connection strength from i to i
con_ie      = 0.            # [mVs] connection strength from e to i [obsolete and not used in the paper, kept in to avoid accidental errors in github]
con_ei      = 0.            # [mVs] connection strength from i to e [obsolete and not used in the paper, kept in to avoid accidental errors in github]
con_ee      = 0.            # [mVs] connection strength from e to e [obsolete and not used in the paper, kept in to avoid accidental errors in github]
p_e         = 0.000         # probability of connection from excitatory neuron [obsolete and not used in the paper, kept in to avoid accidental errors in github]
p_i         = 0.1#100/Ni#0.5#100/Ni#0.1#0.05         # probability of connection from inhibitory neuron
taue        = 0.0 #0.1      # membrane time constant for exc. neurons [s] [obsolete and not used in the paper, kept in to avoid accidental errors in github]
taui        = 0.02#0.02          # membrane time constant for inh. neurons [s]
taus_e      = 0.00          # synaptic time constant, excitatory [s] [obsolete and not used in the paper, kept in to avoid accidental errors in github]
taus_i      = 0.00          # synaptic time constant, inhibitory [s] [obsolete and not used in the paper, kept in to avoid accidental errors in github]
mu0         = 10            # input current [mV]
delay_micro = 0.00001       # delay [s], cannot be set to zero (for now)
delay       = 0#0.002#0
shf_yn      = true          # sh(ow) f(igure) yes/no: no for time benchmark [obsolete and not used in the paper, kept in to avoid accidental errors in github]

seed        = 5             # seed for initial condition [obsolete and not used in the paper, kept in to avoid accidental errors in github]
#Random.seed!(seed)         # set global seed -> same initial conditions each run,
                            # comment out this line if initial conditions v(0) should be different for each run
seed_quenched = 0           # seed for quenched random connectivity, set 0 to be different for each run
nr_trials   = 70            # number of trial runs
T_measure   = 0.24          # time length of each measurement [s]
T_relax     = 0.25#0.25          # relaxation time before PSD is measured [s]
n_measure   = trunc(Int, T_measure/dt_record)
micro_yn    = true
micro_annealed_yn = false
meso_yn     = false
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
#save_location               = "data/var_annealed/"
#dat_file_mame               = "var_annealed_several_p"
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


hmax        = (Phimax/Phi0)^(1.0/alpha)         # [obsolete and not used in the paper, kept in to avoid accidental errors in github]
round_yn    = false                             # [obsolete and not used in the paper, kept in to avoid accidental errors in github]
ini_h_dis   = 0.1#-30#-0.2#0.2                  # [obsolete and not used in the paper, kept in to avoid accidental errors in github]
ini_sig     = 0                                 # [obsolete and not used in the paper, kept in to avoid accidental errors in github]

para_string = "_tau_"*string(taui)*"_N_"*string(Ni)*"_Const_"*string(C_const)*"_J_const_"*string(J_const)
if(just_theory)
    para_string = para_string*"just_theory"
end

save_location               = "data/var_annealed/"
dat_file_name               = "var_annealed_several_p"
source_file_path            = "var_annealed_several_p.jl"

if(savedat_yn)
    filename_uuid = save_simulation_result(save_location, source_file_path, dat_file_name*para_string; get_also_current_time=true)
    total_save_data_name = save_location*filename_uuid*".npz"
end

p_arr = pmin:dp:pmax
#p_arr = [0.01, 0.15, 0.29, 0.43, 0.57, 0.71, 0.85, 0.99]
#p_arr = [0.43, 0.57, 0.71, 0.85, 0.99]
#p_arr = collect(p_arr)
#push!(p_arr, 1.0)
#pushfirst!(p_arr, 0.0)
#p_arr = [0]
#p_arr = [100/Ni]  # undo this line afterwards!
len_p           = length(p_arr)
var_rr_ann      = zeros(len_p)
mean_rr_ann     = zeros(len_p)
var_rr_micro    = zeros(len_p)
mean_rr_micro   = zeros(len_p)  
var_rr_meso     = zeros(len_p)
mean_rr_meso    = zeros(len_p)  
var_rr_naive    = zeros(len_p)
mean_rr_naive   = zeros(len_p)
var_hh_ann      = zeros(len_p)
mean_hh_ann     = zeros(len_p)
var_hh_micro    = zeros(len_p)
mean_hh_micro   = zeros(len_p)
var_hh_meso     = zeros(len_p)
mean_hh_meso    = zeros(len_p)
var_hh_naive    = zeros(len_p)
mean_hh_naive   = zeros(len_p)
var_ss_micro    = zeros(len_p)
mean_ss_micro   = zeros(len_p)
var_ss_meso     = zeros(len_p)
mean_ss_meso    = zeros(len_p)
var_ss_ann      = zeros(len_p)
mean_ss_ann     = zeros(len_p)
cov_hs_micro    = zeros(len_p)
cov_hs_meso     = zeros(len_p)
cov_hs_ann      = zeros(len_p)
mean_xixi_meso  = zeros(len_p)
var_xixi_meso   = zeros(len_p)
cov_hxi_meso    = zeros(len_p)
cov_sxi_meso    = zeros(len_p)

var_hh_meso_the             = zeros(len_p)
var_hh_meso_the_simple      = zeros(len_p)      # turn of xi variable, G0 = 0
var_ss_meso_the             = zeros(len_p)
var_ss_meso_the_simple      = zeros(len_p)      # turn of xi variable, G0 = 0
cov_hs_meso_the             = zeros(len_p)
cov_hs_meso_the_simple      = zeros(len_p)
cov_hxi_meso_the_simple     = zeros(len_p)
cov_sxi_meso_the_simple     = zeros(len_p)
var_xixi_meso_the_simple    = zeros(len_p)
cov_hxi_meso_the            = zeros(len_p)
cov_sxi_meso_the            = zeros(len_p)
var_xixi_meso_the           = zeros(len_p)
var_rr_meso_the             = zeros(len_p)
var_rr_meso_the_simple      = zeros(len_p)
mean_xixi_meso_the          = zeros(len_p)
var_hh_naive_the            = zeros(len_p)
var_rr_naive_the            = zeros(len_p)
mean_hh_meso_the            = zeros(len_p)
mean_ss_meso_the            = zeros(len_p)
mean_rr_meso_the            = zeros(len_p)
mean_hh_naive_the           = zeros(len_p)
mean_rr_naive_the           = zeros(len_p)

mean_rr_meso_analyt         = zeros(len_p)
mean_hh_meso_analyt         = zeros(len_p)
var_rr_meso_analyt          = zeros(len_p)
var_hh_meso_analyt          = zeros(len_p)


mean_hh_meso_sparse_the     = zeros(len_p)
mean_ss_meso_sparse_the     = zeros(len_p)
mean_rr_meso_sparse_the     = zeros(len_p)



if(comparemode)
    G11 = zeros(len_p)
    G12 = zeros(len_p)
    G13 = zeros(len_p)
    G21 = zeros(len_p)
    G22 = zeros(len_p)
    G23 = zeros(len_p)
    G31 = zeros(len_p)
    G32 = zeros(len_p)
    G33 = zeros(len_p)
    rv  = zeros(len_p)
    sv  = zeros(len_p)
    hv  = zeros(len_p)
    Fhv = zeros(len_p)
    Fsv = zeros(len_p)
    av  = zeros(len_p)
    bv  = zeros(len_p)
    gv  = zeros(len_p)
    Fh1 = zeros(len_p)
    Fh2 = zeros(len_p)
    mu0wphimax = zeros(len_p)
    phimu0wphimax = zeros(len_p)
    phimu0wphimaxsq = zeros(len_p)
    shh_naive_simple=zeros(len_p)
    Fh_naive = zeros(len_p)
    r0n_simp = zeros(len_p)
    r0m_simp = zeros(len_p)
    h0n_simp = zeros(len_p)
    s0m_simp = zeros(len_p)
    h0m_simp = zeros(len_p)
    Fhn_semi = zeros(len_p)
    Fhn_simp = zeros(len_p)
    Fhm_simp = zeros(len_p)
    Fh_param = zeros(len_p)
    Fs_param = zeros(len_p)
    h0_param = zeros(len_p)
    s0_param = zeros(len_p)
    r0_param = zeros(len_p)
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

for i=1:len_p
    println(i)
    if(C_const)
        global Ni = round(Int, C0/p_arr[i])
        println(Ni)
    end
    if(J_const)
        global C = Ni * p_arr[i]
        if(C_const)
            C=C0
        end
        global con_ii = C * J0
    end


    global param_Phi   = Dict("beta"=>beta, "theta"=>theta,"Phimax"=> Phimax)
    global param_lnp   = Dict("con_ii"=> con_ii, "con_ei"=> con_ei, "con_ie"=> con_ie, "con_ee"=> con_ee, "p_e"=> p_e, "p_i" => p_arr[i],
                        "taue"=> taue, "taui"=> taui, "taus_e"=> taus_e, "taus_i"=> taus_i, "mu0"=> mu0, "delay"=> delay_micro, "mu_dt" => mu_dt_micro, "mu_t_jump"=>mu_t_jump_micro, "mu_threshold"=> mu_threshold,
                            "mu_const"=>mu_const, "mu_a"=>mu_a, "mu_phi"=>mu_phi, "mu_om"=>mu_om, "dummy"=>[1,2,3], "sparse_limit_yn"=> sparse_limit_yn)

    global param_meso_corrected = Dict("theta"=>theta, "beta"=> beta, "Phimax"=> Phimax, "tmax"=> T, "dt"=> dt_meso, "p"=>p_arr[i],
                            "w"=>con_ii, "N"=> Ni, "C"=> Ni * p_arr[i], "tau"=>taui, "delta"=>delay, "mu"=> mu0, "round_yn"=>round_yn, "mu0"=>mu0, "max_search_for_fixed_point"=>max_search_for_fixed_point,
                            "F_smoothed"=> F_sigm_erf, "delFdels" => delFdels_sig_erf, "phi_prime" => del_sigm_erf_del_h, "mu_dt" => mu_dt, "mu_t_jump"=>mu_t_jump, "mu_threshold"=> mu_threshold,
                            "mu_const"=>mu_const, "mu_a"=>mu_a, "mu_phi"=>mu_phi, "mu_om"=>mu_om, "sparse_limit_yn"=>sparse_limit_yn)

    if(C_const)
        global param_meso_naive = Dict("theta"=>theta, "beta"=> beta, "Phimax"=> Phimax, "tmax"=> T, "dt"=> dt_meso, "p"=>1,
                            #"w"=>con_ii, "N"=> Ni0, "C"=> Ni0, "tau"=>taui, "delta"=>delay, "mu"=> mu0, "round_yn"=>round_yn, "mu0"=>mu0, "max_search_for_fixed_point"=>max_search_for_fixed_point,
                            "w"=>con_ii, "N"=> Ni0/p_arr[i], "C"=> Ni0, "tau"=>taui, "delta"=>delay, "mu"=> mu0, "round_yn"=>round_yn, "mu0"=>mu0, "max_search_for_fixed_point"=>max_search_for_fixed_point,
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
        if(!just_theory)
            global micro_annealed_res = Micro_Simulation_Annealed_fixedindegree(T, dt, Ne, Int(Ni), Nrecord, dt_record, Phi_sigm_erf, param_Phi, param_lnp, seed_quenched,fixed_in_degree=fixed_in_degree_annaeled, ini_v = ini_h_dis,
                                                         record_mean_h_i=true, ini_sig=ini_sig, record_var_h_i=true, mu_func = mu_func, record_skew_h_i=record_skew_h_i)
            mean_rr_ann[i]  = mean(micro_annealed_res.rpopI_record[n_relax_micro:end])
            var_rr_ann[i]   = var(micro_annealed_res.rpopI_record[n_relax_micro:end])
            var_hh_ann[i]   = var(micro_annealed_res.mean_h_i[n_relax_micro:end])
            mean_hh_ann[i]  = mean(micro_annealed_res.mean_h_i[n_relax_micro:end])
            var_ss_ann[i]   = var(micro_annealed_res.var_h_i[n_relax_micro:end])
            mean_ss_ann[i]  = mean(micro_annealed_res.var_h_i[n_relax_micro:end])
            cov_hs_ann[i]   = cov(micro_annealed_res.var_h_i[n_relax_micro:end], micro_annealed_res.mean_h_i[n_relax_micro:end])
        end
    end
    if(micro_yn)
        if(!just_theory)
            global micro_res = micro_simulation(T, dt, Ne, Int(Ni), Nrecord, dt_record, Phi_sigm_erf, param_Phi, param_lnp, seed_quenched,fixed_in_degree=fixed_in_degree, ini_v = ini_h_dis,
                                                         record_mean_h_i=true, ini_sig=ini_sig, record_var_h_i=true, mu_func = mu_func, record_skew_h_i=record_skew_h_i)
            mean_rr_micro[i]    = mean(micro_res.rpopI_record[n_relax_micro:end])
            var_rr_micro[i]     = var(micro_res.rpopI_record[n_relax_micro:end])
            var_hh_micro[i]     = var(micro_res.mean_h_i[n_relax_micro:end])
            mean_hh_micro[i]    = mean(micro_res.mean_h_i[n_relax_micro:end])
            var_ss_micro[i]     = var(micro_res.var_h_i[n_relax_micro:end])
            mean_ss_micro[i]    = mean(micro_res.var_h_i[n_relax_micro:end])
            cov_hs_micro[i]     = cov(micro_res.var_h_i[n_relax_micro:end], micro_res.mean_h_i[n_relax_micro:end])
        end
    end
    if(meso_yn)
        if(!just_theory)
            global meso_corrected_res = mesoscopic_model_correction_colored_noise_time_dep_stimulus(h0, s0, X0, F_sigm_erf, G_sigm_erf, mu_func, param_meso_corrected, model3d=1)
            mean_rr_meso[i] = mean(meso_corrected_res.r[n_relax_meso:end])
            var_rr_meso[i]  = var(meso_corrected_res.r[n_relax_meso:end])
            var_hh_meso[i]  = var(meso_corrected_res.h[n_relax_meso:end])
            mean_hh_meso[i] = mean(meso_corrected_res.h[n_relax_meso:end])
            var_ss_meso[i]  = var(meso_corrected_res.s[n_relax_meso:end])
            mean_ss_meso[i] = mean(meso_corrected_res.s[n_relax_meso:end])
            mean_xixi_meso[i]   = mean(meso_corrected_res.X[n_relax_meso:end])
            var_xixi_meso[i]    = var(meso_corrected_res.X[n_relax_meso:end])
            cov_hxi_meso[i]     = cov(meso_corrected_res.h[n_relax_meso:end], meso_corrected_res.X[n_relax_meso:end])
            cov_sxi_meso[i]     = cov(meso_corrected_res.s[n_relax_meso:end], meso_corrected_res.X[n_relax_meso:end])
            cov_hs_meso[i]  = cov(meso_corrected_res.s[n_relax_meso:end], meso_corrected_res.h[n_relax_meso:end])
        end
        mean_hh_meso_the[i] = h0
        mean_ss_meso_the[i] = s0
        mean_rr_meso_the[i] = r0
        var_hh_meso_the[i], cov_hs_meso_the[i], var_ss_meso_the[i], cov_hxi_meso_the[i],
            cov_sxi_meso_the[i], var_xixi_meso_the[i], var_rr_meso_the[i] = linearized_variance_color_corrected(param_meso_corrected, false, turn_off_xi_variable=false)
        var_hh_meso_the_simple[i], cov_hs_meso_the_simple[i], var_ss_meso_the_simple[i], cov_hxi_meso_the_simple[i],
            cov_sxi_meso_the_simple[i], var_xixi_meso_the_simple[i], var_rr_meso_the_simple[i] = linearized_variance_color_corrected(param_meso_corrected, false, turn_off_xi_variable=true)
        mean_hh_meso_analyt[i], _,  mean_rr_meso_analyt[i] = mean_complete_analytical_approximation(param_meso_corrected)
        var_rr_meso_analyt[i], var_hh_meso_analyt[i] = variance_complete_analytical_approximation(param_meso_corrected)
        mean_hh_meso_sparse_the[i] = h0_sparse
        mean_ss_meso_sparse_the[i] = s0_sparse
        mean_rr_meso_sparse_the[i] = r0_sparse

    end
    println("here2")
    if(naive_yn)
        global h0n, s0n, r0n, fix_foundn = PSD_get_one_fix_point(param_meso_naive, true)
        global X0n=0
        if(!just_theory)
            println("here")
            global naive_corrected_res = mesoscopic_model_correction_colored_noise_time_dep_stimulus(h0n, 0.0, 0.0, F_sigm_erf, G_sigm_erf, mu_func, param_meso_naive, naive_yn=naive_yn)
            mean_rr_naive[i]    = mean(naive_corrected_res.r[n_relax_naive:end])
            var_rr_naive[i]     = var(naive_corrected_res.r[n_relax_naive:end])
            var_hh_naive[i]     = var(naive_corrected_res.h[n_relax_meso:end])
            mean_hh_naive[i]    = mean(naive_corrected_res.h[n_relax_meso:end])
        end
        mean_hh_naive_the[i]= h0n
        mean_rr_naive_the[i]=r0n
        var_hh_naive_the[i], _, _, _, _, _, var_rr_naive_the[i] = linearized_variance_color_corrected(param_meso_naive, true, turn_off_xi_variable=false) # most of the variables are 0 by default
    end

    if(comparemode)
        println(i)
        G11[i], G12[i], G13[i], G21[i], G22[i], G23[i], G31[i], G32[i], G33[i], Fhv[i], Fsv[i], av[i], bv[i], gv[i], hv[i], sv[i], rv[i] = linearized_variance_color_corrected_Gamma_Matrix_Analysis(param_meso_corrected, false)
        #Fh1[i] = Phimax/sqrt(2*pi)*exp(-0.5*(sig_erf_inv(mu0/(-con_ii*Phimax)))^2)
        #Fh2[i] = 1/sqrt(1/beta^2 - con_ii*mu0*(1-p_arr[i]) /(2*taui*Ni*p_arr[i]))
        Fh_param[i] = Fh_in_param(param_meso_corrected)
        Fs_param[i] = Fs_in_param(param_meso_corrected)
        h0_param[i], s0_param[i], r0_param[i] = FP_approxed(param_meso_corrected)
         
        if(false)
        if(mu0 < -con_ii * Phimax)        
            r0n_simp[i] = -mu0/con_ii
            r0m_simp[i] = -mu0/con_ii
            h0n_simp[i] = 0
            s0m_simp[i] = con_ii^2 * (1-p_arr[i])/(2*taui*Ni*p_arr[i]) * r0m_simp[i]
            h0m_simp[i] = sig_erf_inv(r0m_simp[i]/Phimax) *sqrt(1/beta^2+s0m_simp[i])
            Fhn_simp[i] = Phimax*beta/sqrt(2*pi)
            Fhn_semi[i] = delFdelh_sig_erf(h0n, 0, param_meso_naive)
            Fhm_simp[i] = delFdelh_sig_erf(h0m_simp[i], s0m_simp[i], param_meso_corrected)
            Fh1[i] = Phimax/sqrt(2*pi)*exp(-0.5*(sig_erf_inv(mu0/(-con_ii*Phimax)))^2)
            Fh2[i] = 1/sqrt(1/beta^2 - con_ii*mu0*(1-p_arr[i]) /(2*taui*Ni*p_arr[i]))
            mu0wphimax[i] = -mu0/(con_ii * Phimax)
            phimu0wphimax[i] = sig_erf_inv(mu0wphimax[i])
            phimu0wphimaxsq[i] = (phimu0wphimax[i])^2
        else
            r0n_simp[i] = Phimax
            r0m_simp[i] = Phimax
            h0n_simp[i] = con_ii*Phimax + mu0
            s0m_simp[i] = con_ii^2 * (1-p_arr[i])/(2*taui*Ni*p_arr[i]) * r0m_simp[i]
            h0m_simp[i] = con_ii*Phimax + mu0
            Fhn_simp[i] = Phimax*beta/sqrt(2*pi) * exp(-0.5*beta^2*(con_ii^2*Phimax+mu0)^2)
            Fhn_semi[i] = delFdelh_sig_erf(h0n, 0, param_meso_naive)
            Fhm_simp[i] = delFdelh_sig_erf(h0m_simp[i], s0m_simp[i], param_meso_corrected)
            Fh1[i] = sqrt(abs(delFdelh_sig_erf(h0m_simp[i], s0m_simp[i], param_meso_corrected)))
            Fh2[i] = sqrt(abs(delFdelh_sig_erf(h0m_simp[i], s0m_simp[i], param_meso_corrected)))
        end
        end
    end
end

if(savedat_yn)
    NPZ.npzwrite(total_save_data_name, Dict("p_arr"=>p_arr, "mean_hh_ann"=>mean_hh_ann, "var_hh_ann"=>var_hh_ann, "mean_ss_ann"=> mean_ss_ann, "var_ss_ann"=>var_ss_ann,
                                            "cov_hs_ann"=>cov_hs_ann, "mean_rr_ann"=>mean_rr_ann, "var_rr_ann"=>var_rr_ann, "mean_rr_micro"=>mean_rr_micro, 
                                            "var_rr_micro"=> var_rr_micro, "mean_hh_micro"=> mean_hh_micro, "var_hh_micro"=> var_hh_micro, "mean_ss_micro"=>mean_ss_micro,
                                            "var_ss_micro"=>var_ss_micro, "cov_hs_micro"=>cov_hs_micro, "mean_rr_meso"=>mean_rr_meso, "var_rr_meso"=>var_rr_meso,
                                            "mean_hh_meso"=>mean_hh_meso, "var_hh_meso"=>var_hh_meso, "mean_ss_meso"=>mean_ss_meso, "var_ss_meso"=>var_ss_meso,
                                            "mean_xixi_meso"=>mean_xixi_meso, "var_xixi_meso"=>var_xixi_meso, "cov_hxi_meso"=>cov_hxi_meso, "cov_sxi_meso"=>cov_sxi_meso,
                                            "var_hh_meso_the"=>var_hh_meso_the, "cov_hs_meso_the"=>cov_hs_meso_the, "var_ss_meso_the"=>var_ss_meso_the, "cov_hxi_meso_the"=> cov_hxi_meso_the,
                                            "cov_sxi_meso_the"=>cov_sxi_meso_the, "var_xixi_meso_the"=>var_xixi_meso_the, "var_rr_meso_the"=>var_rr_meso_the,
                                            "var_hh_meso_the_simple"=>var_hh_meso_the_simple, "cov_hs_meso_the_simple"=>cov_hs_meso_the_simple, "var_ss_meso_the_simple"=>var_ss_meso_the_simple,
                                            "cov_hxi_meso_the_simple"=> cov_hxi_meso_the_simple, "cov_sxi_meso_the_simple"=>cov_sxi_meso_the_simple,
                                            "var_xixi_meso_the_simple"=>var_xixi_meso_the_simple, "var_rr_meso_the_simple"=>var_rr_meso_the_simple,
                                            "mean_rr_naive"=>mean_rr_naive, "var_rr_naive"=>var_rr_naive, "var_hh_naive"=>var_hh_naive, "mean_hh_naive"=>mean_hh_naive,
                                            "var_hh_naive_the"=>var_hh_naive_the, "var_rr_naive_the"=>var_rr_naive_the, "mean_hh_meso_the"=>mean_hh_meso_the, 
                                            "mean_ss_meso_the"=>mean_ss_meso_the, "mean_rr_meso_the"=>mean_rr_meso_the, "mean_hh_naive_the"=>mean_hh_naive_the, 
                                            "mean_rr_naive_the"=>mean_rr_naive_the, "mean_hh_meso_sparse_the"=>mean_hh_meso_sparse_the, "mean_ss_meso_sparse_the"=>mean_ss_meso_sparse_the,
                                            "mean_rr_meso_sparse_the"=>mean_rr_meso_sparse_the))
end
