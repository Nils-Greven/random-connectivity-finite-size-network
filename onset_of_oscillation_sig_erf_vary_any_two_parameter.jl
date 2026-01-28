#=
calculate onset of oscillation for sigmoidal error function
=#
using PyPlot, Printf, Statistics, Random, FFTW, StatsBase,LinearAlgebra, SpecialFunctions, Roots, NPZ
using PyCall, NLsolve, UUIDs
py"""
       import sys
       sys.path.insert(0, "./src")
       """
opt_py = pyimport("opt_py")
include("src/sim.jl")
include("src/sim_2.jl")

Ne          = 0             # number excitatory cells
Ni          = 1000#20000          # number inhibitory cells
Ncells      = Ne + Ni
Nrecord     = Ncells        # number neurons to be recorded

T           = 5             # simulation time [s]
dt          = 1e-4          # [s] simulation timestep
dt_record   = dt            # [s] record time step 
dt_meso     = dt_record/10
dt_naive    = dt_meso
alpha       = 0.1           # exponent in transfer function
beta        = 5             # [mV^-1]steepness of the sigmoidal-errf transfer function
theta       = 0.            # [mV] voltage of maximal increase in sigm-errf transfer function
Phi0        = 0.65          # [s^-1] prefactor of transfer function theta^(-alpha) = (1 mV)^(-alpha) 
Phimax      = 100           # [s^-1] maximal firing rate of neurons
con_ii      = -1#-0.0895#-0.01      # [mVs] connection strength from i to i
con_ie      = 0.            # [mVs] connection strength from e to i
con_ei      = 0.            # [mVs] connection strength from i to e
con_ee      = 0.            # [mVs] connection strength from e to e
p_e         = 0.000         # probability of connection from excitatory neuron
p_i         = 0.1#100/Ni#4*0.005         # probability of connection from inhibitory neuron
taue        = 0#0.1           # membrane time constant for exc. neurons [s]
taui        = 0.02          # membrane time constant for inh. neurons [s]
taus_e      = 0.00          # synaptic time constant, excitatory [s]
taus_i      = 0.00          # synaptic time constant, inhibitory [s]
mu0         = 10.0          # input current [mV]
delay       = 1.0           # delay [s], cannot be set to zero (for now)
a_string    = "delay"
a_max       = 0.010
a_min       = 0.00001#0.0001#0.001
da          = 0.00005
b_string    = "w"
b_min       = -2
b_max       = 0#2
db          = 0.01
a_label     = "delay [s]"
b_label     = "w [mVs]"
kmax        = 0         #maximal number of curves --> needs better comment!
max_search_for_fixed_point = 1000
naive_model_yn = true
meso_model_yn  = true
omega_max_search = 10000
rel_tol = 10^(-3)

shf_yn      = false         # sh(ow) f(igure) yes/no
svfig_yn    = false
svdat_yn    = false         # deprecated
savedat_yn  = true
sparse_limit_yn = true
scatterplot_yn = false
file_ending = ".png"





param_meso = Dict("delay"=>delay, "w"=>con_ii, "tau"=>taui, "d"=>delay, "p"=>p_i, "N"=>Ni, "mu0"=>mu0, "Phimax"=>Phimax, "theta"=>theta, "beta"=>beta, "delta"=>delay, "C"=>Ni*p_i, 
                "max_search_for_fixed_point" => max_search_for_fixed_point, "omega_max_search" => omega_max_search, "rel_tol" =>rel_tol, "sparse_limit_yn"=> sparse_limit_yn)
param_naive = Dict("delay"=> delay, "w"=>con_ii, "tau"=>taui, "d"=>delay, "p"=>1, "N"=>Ni, "mu0"=>mu0, "Phimax"=>Phimax, "theta"=>theta, "beta"=>beta, "delta"=>delay, "C"=>Ni*p_i, 
                "max_search_for_fixed_point" => max_search_for_fixed_point, "omega_max_search" => omega_max_search, "rel_tol" =>rel_tol)
para_string = "varied_a_"*a_string*"_"*string(a_min)*"_"*string(da)*"_"*string(a_max)*"varied_b_"*b_string*"_"*string(b_min)*"_"*string(db)*"_"*string(b_max)*"N_"*string(Ni)*"_w_"*string(con_ii)*"_p_"*string(p_i)*"_mu0_"*string(mu0)*"_tau_"*string(taui)*"_delay_"*string(delay)
para_string*= "_theta_"*string(theta)*"_beta_"*string(beta)*"_Phimax_"*string(Phimax)*"kmax"*string(kmax)


para_string = "N_"*string(Ni)*"_p_"*string(p_i)

save_location       = "data/onset_of_oscillation/Hopf_boundary/"
dat_file_name       = "Hopf_boundary_semianalytic_"
source_file_path    = "onset_of_oscillation_sig_erf_vary_any_two_parameter_COPY_FOR_PEER_REVIEW.jl"


if(savedat_yn)
    filename_uuid = save_simulation_result(save_location, source_file_path, dat_file_name*para_string; get_also_current_time=true)
    total_save_data_name = save_location*filename_uuid*".npz"
end



a_arr   = a_min:da:a_max
b_arr   = b_min:db:b_max

a_list = [[] for i=0:kmax]
b_list = [[] for i=0:kmax]

a_list_naive = [[] for i=0:kmax]
b_list_naive = [[] for i=0:kmax]

figure(1)

if(meso_model_yn)
    for k=0:kmax
        if(k==0)
            global map_condition, a_list[k+1], b_list[k+1], om_list, map_S, map_phase, map_om, map_phase_err, map_r0, map_Fhtil, map_Fstil = onset_of_oscillation_vary_any_parameter(a_arr, 
                                        b_arr, a_string, b_string, param_meso, k, false)
        else
             _, a_list[k+1], b_list[k+1], _, _, _, _, _, _, _ = onset_of_oscillation_vary_any_parameter(a_arr, b_arr, a_string, b_string, param_meso, k, false)
        end
        if(k==0)
            plot(a_list[0+1], b_list[0+1], ".", label=string(0)*" meso", color="blue")
        else
            plot(a_list[k+1], b_list[k+1], ".", label=string(k)*" meso")
        end
    end
end
if(naive_model_yn)
    for k=0:kmax
        if(k==0)
            global map_condition_naive, a_list_naive[k+1], b_list_naive[k+1], om_list_naive, map_S_naive, map_phase_naive, map_om_naive, map_phase_err_naive, map_r0_naive, map_Fhtil_naive, map_Fstil_naive = onset_of_oscillation_vary_any_parameter(a_arr, 
                                        b_arr, a_string, b_string, param_naive, k, true)
        else
             _, a_list_naive[k+1], b_list_naive[k+1], _, _, _, _, _, _,_ = onset_of_oscillation_vary_any_parameter(a_arr, b_arr, a_string, b_string, param_naive, k, naive_model_yn)
        end
        if(k==0)
            plot(a_list_naive[0+1], b_list_naive[0+1], ".", label=string(0)*" naive", color="orange")
        else
            plot(a_list_naive[k+1], b_list_naive[k+1], ".", label=string(k)*" naive")
        end
    end
end
xlabel(a_label)
ylabel(b_label)
legend()
saveplot("data/onset_of_oscillation/Hopf_boundary/", "Hopf_boundary_semianalytic_"*para_string, file_ending, svfig_yn; dpi=300)
if(savedat_yn)
#    NPZ.npzwrite("data/onset_of_oscillation/Hopf_boundary/Hopf_boundary_semianalytic_"*para_string*".npz", Dict(
    NPZ.npzwrite(total_save_data_name, Dict(
                    "a_arr" => a_arr, "b_arr"=>b_arr, "kmax"=>kmax, "a_list"=>convert(Array{Float64, 1}, a_list[1]), "b_list"=>convert(Array{Float64, 1}, b_list[1]), "om_list"=>convert(Array{Float64, 1}, om_list), 
                        "a_list_naive"=> convert(Array{Float64, 1}, a_list_naive[1]), "b_list_naive"=>convert(Array{Float64, 1}, b_list_naive[1]), "om_list_naive"=>convert(Array{Float64, 1}, om_list_naive)))
end
if(scatterplot_yn)
    figure(2)
    scatter(a_list[1], b_list[1], c=om_list, label="meso")
    scatter(a_list_naive[1], b_list_naive[1], c=om_list_naive, label="naive")
    colorbar()
    legend()
    xlabel(a_label)
    ylabel(b_label)
    saveplot("data/onset_of_oscillation/Hopf_boundary/", "Hopf_boundary_semianalytic_"*para_string*"_EXTRA_CMAP_Of_OM", file_ending, svfig_yn; dpi=300)
end
