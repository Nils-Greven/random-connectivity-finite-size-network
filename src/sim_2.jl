mutable struct Micro_Simulation_Results
    ns::Union{ Nothing, Vector{Int} }
    v_record::Union{ Nothing, Matrix{Float32}, Vector{Vector{Float32}} }
    spikes::Union{ Nothing, Matrix{Int8} }
    rpopE_record::Union{ Nothing, Vector{Float32} }
    rpopI_record::Union{ Nothing, Vector{Float32} }
    actpopE_record::Union{ Nothing, Vector{Float32} }
    actpopI_record::Union{ Nothing, Vector{Float32} }
    mean_h_i::Union{ Nothing, Vector{Float32} }
    var_h_i::Union{ Nothing, Vector{Float32} }
    skew_h_i::Union{ Nothing, Vector{Float32} }
end


        
function micro_simulation(T, dt, Ne, Ni, Nrecord, dt_rec, Phi, param_Phi, param_lnp, seed_quenched;fixed_in_degree=true, ini_v=0, record_mean_h_i=false, mu_func=mu_func_0, ini_sig=0, record_var_h_i=false, record_skew_h_i=false)
#=
Simulate microscopic model of spiking neurons, in general an IE-network
inputs:
    T:              total runtime [s]
    Ne:             number of exc. neurons
    Ni:             number of inhibitory neurons


results:
    ns:             array with the number of spikes for each neuron
    v_record:       array with number of neurons that are recorded and the time bins for which they record the voltage of the neurons
    spikes:         spike trains to be recorded
    rpopE_record:   population rate excitatory
    rpopI_record:   population rate inhibitory


=#

    if(seed_quenched == 0)
        rng = Random.MersenneTwister()
        param_lnp["rng"] = rng
    else
        rng = Random.MersenneTwister(seed_quenched)
        param_lnp["rng"] = rng
    end

    Ncells  = Ne + Ni

    if(!haskey(param_lnp, "sig_mu_e"))
        param_lnp["sig_mu_e"] = 0
    end
    sig_mu_e = param_lnp["sig_mu_e"]

    if(!haskey(param_lnp, "sig_mu_i"))
        param_lnp["sig_mu_i"] = 0
    end
    sig_mu_i = param_lnp["sig_mu_i"]
    
    taue    = param_lnp["taue"] #membrane time constant for exc. neurons (s)
    taui    = param_lnp["taui"]

    taus_e  = param_lnp["taus_e"] #synaptic time constant (s)
    taus_i  = param_lnp["taus_i"] #synaptic time constant (s)

    Etaus = zeros(Float16, Ncells)
    for m in 1:Ne
        if taus_e > 0
            Etaus[m] = exp(-dt / taus_e)
        end
    end
    for m in 1:Ni
        if taus_i > 0
            Etaus[Ne+m] = exp(-dt / taus_i)
        end
    end

    #connection probabilities
    p_i = param_lnp["p_i"]
    p_e = param_lnp["p_e"]
    
    Ke = round(Int, Ne * p_e)
    Ki = round(Int, Ni * p_i)
    
    jie = param_lnp["con_ie"] / Ke
    jei = param_lnp["con_ei"] / Ki
    jii = param_lnp["con_ii"] / Ki 
    jee = param_lnp["con_ee"] / Ke
    
    #stimulation
    mu0 = param_lnp["mu0"]
    mu  = zeros(Ncells)
    mu[1:Ne] = mu0 .+ randn(rng, (Ne,)).*sqrt(sig_mu_e)
    mu[(Ne+1):Ncells] = mu0 .+ randn(rng, (Ni,)).*sqrt(sig_mu_i)
    
    delay = param_lnp["delay"]
    n_delay = max(ceil(Int, delay/dt), 1)
        
    Nbin = round(Int, dt_rec/dt) 
    
    #membrane time constant
    tau = zeros(Ncells)
    dtau = zeros(Ncells)
    
    tau[1:Ne] .= taue
    tau[(1+Ne):Ncells] .= taui
    dtau[1:Ne] .= dt/taue
    dtau[(1+Ne):Ncells] .= dt/taui


    weights = zeros(Ncells,Ncells)
    if(fixed_in_degree)
   
        #random excitatory connections
        pre_neurons = zeros(Int, Ne)
        pre_neurons[1:Ke] = ones(Int, Ke)
        for n in 1:Ne
            weights[n,1:Ne] = jee * pre_neurons[randperm(rng, Ne)] 
        end
        for n in Ne+1:Ne+Ni
            weights[n,1:Ne] = jie * pre_neurons[randperm(rng, Ne)] 
        end

        #random inhibitory connections
        pre_neurons = zeros(Int, Ni)
        pre_neurons[1:Ki] = ones(Int, Ki)
        for n in 1:Ne
            weights[n,Ne+1:Ncells] = jei * pre_neurons[randperm(rng, Ni)] 
        end
        for n in Ne+1:Ncells
            weights[n,Ne+1:Ncells] = jii * pre_neurons[randperm(rng, Ni)]
        end
    else
        weights[1:Ne, 1:Ne]                 = jee .* rand(Bernoulli(p_e), (Ne,Ne))
        weights[1:Ne, 1+Ne:Ncells]          = jei .* rand(Bernoulli(p_i), (Ne,Ni))
        weights[1+Ne:Ncells, 1:Ne]          = jie .* rand(Bernoulli(p_e), (Ni,Ne))
        weights[1+Ne:Ncells, 1+Ne:Ncells]   = jii .* rand(Bernoulli(p_i), (Ni,Ni))
    end
    
    ns = zeros(Int,Ncells)
    
    forwardInputsE = zeros(Ncells)
    forwardInputsI = zeros(Ncells)

    delaylineE = zeros(Ncells, n_delay)
    delaylineI = zeros(Ncells, n_delay)
    start = 1

    
    xe = zeros(Ncells)                                      # synaptic variable for excitatory input
    xi = zeros(Ncells)                                      # synaptic variable for inhibitory input

    #v = 0 * ones(Ncells)                                    # voltage of neurons
    #v = ini_v * ones(Ncells)
    v = ini_v * ones(Ncells) .+ randn(Ncells) .* sqrt(ini_sig)
    #v = randn(Ncells).*20 .-30
    spikes_per_time_stepI = 0
    spikes_per_time_stepE = 0

    #quantities to be recorded
    Nsteps = round(Int,T/dt)
    Nsteps_rec = round(Int,T/dt_rec)
    # Ie_record = zeros(Float32,(Nrecord, Nsteps_rec))
    # Ii_record = zeros(Float32,(Nrecord, Nsteps_rec))
    # Itot_record = zeros(Float32,(Nrecord, Nsteps_rec))
    # Isyntot_record = zeros(Float32,(Nrecord, Nsteps_rec))
    v_record = zeros(Float32,(Nrecord, Nsteps_rec))         # voltage traces to be recorded
    if(record_mean_h_i)
        mean_h_i = zeros(Float32, Nsteps_rec)               # mean of distribution of voltage distribution for each recorded time step
    else
        mean_h_i = nothing
    end
    if(record_var_h_i)
        var_h_i = zeros(Float32, Nsteps_rec)               # mean of distribution of voltage distribution for each recorded time step
    else
        var_h_i = nothing
    end
    if(record_skew_h_i)
        skew_h_i = zeros(Float32, Nsteps_rec)               # mean of distribution of voltage distribution for each recorded time step
    else
        skew_h_i = nothing
    end
    rpopE_record    = zeros(Float32,Nsteps_rec)                # E population rate
    rpopI_record    = zeros(Float32,Nsteps_rec)                # I population rate
    spikes          = zeros(Int8, (Nrecord, Nsteps_rec))             # spike trains to be recorded
    actpopE_record  = zeros(Float32,Nsteps_rec)
    actpopI_record  = zeros(Float32,Nsteps_rec)
#    println("starting simulation")

    poprateE = 0
    poprateI = 0
    #begin main simulation loop
    for ti = 1:Nsteps
	    if mod(ti,Nsteps/100) == 1  #print percent complete
            ########@printf("\33[2K\r%d%% poprateE=%g poprateI=%g, abgelaufene Zeit=%g, rate=%g, Mu%g",round(Int,100*ti/Nsteps), poprateE, poprateI, ti*dt, sum(ns)/(Ni*ti*dt), mu0)
#            @printf("xe=%g xi=%g xetheo=%g xitheo=%g\n",xe[1],xi[1], poprateE*jee*taus*Ke, poprateI*jei*taus*Ki)
	    end
	    t = dt*ti
        i_rec = ceil(Int, ti/Nbin)
#        i_rec= floor(Int, t/dt_rec)+1
	    forwardInputsE[:] .= 0
	    forwardInputsI[:] .= 0
        spikes_per_time_stepI = 0
        spikes_per_time_stepE = 0

        ende = ((start + n_delay -2) % n_delay) + 1
        
        #mu = mu0                                            # just renaming

        poprateE = 0
        poprateI = 0

        mu_ext = mu_func(t, param_lnp)

	    for ci = 1:Ncells
            xe[ci] = xe[ci] * Etaus[ci] + delaylineE[ci, ende] / dt * (1 - Etaus[ci])
            xi[ci] = xi[ci] * Etaus[ci] + delaylineI[ci, ende] / dt * (1 - Etaus[ci])
	        # xe[ci] += -dt*xe[ci]/taus + delaylineE[ci, ende]  #mV
	        # xi[ci] += -dt*xi[ci]/taus + delaylineI[ci, ende]
                
	        synInputE = xe[ci]
            synInputI = xi[ci]
            synInput = synInputE + synInputI
                
	        v[ci] += dtau[ci]*(mu[ci]+mu_ext-v[ci] + synInput)

            #rate = Phi(v[ci], Phi0, alpha)
            rate = Phi(v[ci], param_Phi)
            if (ci<=Ne)
                poprateE += rate #r_e
            else
                poprateI += rate #r_i
            end
            
	        if (rand() < rate * dt)  #spike occurred
		        ns[ci] = ns[ci]+1
                if(ci<=Ne)
                    spikes_per_time_stepE += 1
                else
                    spikes_per_time_stepI += 1
                end
                if (ci<=Nrecord)
                    spikes[ci, i_rec] = 1
                end
                    
		        for j = 1:Ncells
                    w = weights[j,ci]
		            if  w > 0  #E synapse
			            forwardInputsE[j] += w
		            elseif w < 0  #I synapse
			            forwardInputsI[j] += w
		            end
		        end #end loop over synaptic projections
	        end #end if(spike occurred)
            
            if (ci <= Nrecord)
                # Ie_record[ci,i_rec] += synInputE  # excitatory current in mV
                # Ii_record[ci,i_rec] += synInputI  # inhibitory current in mV
                # Isyntot_record[ci,i_rec] += synInput # total synaptic current in mV
                # Itot_record[ci,i_rec] += synInput + mu + eta[ci] # total current in mV
                v_record[ci,i_rec] += v[ci]                # subthreshold membrane potential
            end #end loop over recorded neurons
            
	    end #end loop over neurons

        spikes_per_time_stepE /= Ne
        spikes_per_time_stepI /= Ni
        actpopE_record[i_rec] += spikes_per_time_stepE
        actpopI_record[i_rec] += spikes_per_time_stepI        
        poprateE /= Ne
        poprateI /= Ni
        rpopE_record[i_rec] += poprateE
        rpopI_record[i_rec] += poprateI
        if(record_mean_h_i) 
            mean_h_i[i_rec] += mean(v)
        end
        if(record_var_h_i) 
            var_h_i[i_rec] += var(v)
        end
        if(record_skew_h_i) 
            skew_h_i[i_rec] += calculate_emphirical_skewness(v)
        end
        
        for j in 1:Ncells
            # push!(delaylineE[j], forwardInputsE[j]) #add total input to delay line
            # push!(delaylineI[j], forwardInputsI[j])
            delaylineE[j, ende] = forwardInputsE[j]
            delaylineI[j, ende] = forwardInputsI[j]
        end #end loop over postsynaptic neurons
        start = ende
    end #end loop over time

    actpopI_record /= dt
    actpopE_record /= dt


    v_record /=Nbin
    rpopE_record /=Nbin
    rpopI_record /=Nbin
    actpopE_record /=Nbin
    actpopI_record /=Nbin

    if(record_mean_h_i)        
        mean_h_i /=Nbin
    end
    if(record_var_h_i)
        var_h_i /=Nbin  # Mean variance of h_i over the Nbins
    end
    if(record_skew_h_i)
        skew_h_i /=Nbin  # Mean variance of h_i over the Nbins
    end

    #@printf("\r")
    return Micro_Simulation_Results(ns, v_record, spikes, rpopE_record, rpopI_record, actpopE_record, actpopI_record, mean_h_i, var_h_i, skew_h_i)
end

function get_git_commit_id(get_also_current_time=false)
    try
        commit_id = strip(read(`git rev-parse HEAD`, String))
    catch
        commit_id = "UNKNOWN"
    end
    if(get_also_current_time)
        commit_id = commit_id * " and date " * strip(read(`date`, String))
    end
    return commit_id
end

function save_simulation_result(data_folder, source_file_path, filename; get_also_current_time=false)
    filename_uuid = filename * "_$(uuid4())"

    # Generate a unique identifier for the result file
    result_filename = filename_uuid * ".para"
    
    # Create the destination path
    destination_path = joinpath(data_folder, result_filename)
    
    # Copy the source file to the destination folder
    cp(source_file_path, destination_path)
    
    # Get the current commit ID
    commit_id = get_git_commit_id(get_also_current_time)
    
    # Prepend the commit ID to the copied file
    prepend_commit_id(destination_path, commit_id)
    
    println("Simulation result saved to: $destination_path")
    return filename_uuid
end

function prepend_commit_id(file_path, commit_id)
    # Read the existing content of the file
    content = read(file_path, String)
    
    # Prepend the commit ID to the content
    new_content = "# Commit ID $commit_id\n$content"
    
    # Write the updated content back to the file
    open(file_path, "w") do file
        write(file, new_content)
    end
end

mutable struct Micro_Simulation_Annealed_fixedindegree_Results
    ns::Union{ Nothing, Vector{Int} }
    v_record::Union{ Nothing, Matrix{Float32}, Vector{Vector{Float32}} }
    spikes::Union{ Nothing, Matrix{Int8} }
    rpopE_record::Union{ Nothing, Vector{Float32} }
    rpopI_record::Union{ Nothing, Vector{Float32} }
    actpopE_record::Union{ Nothing, Vector{Float32} }
    actpopI_record::Union{ Nothing, Vector{Float32} }
    mean_h_i::Union{ Nothing, Vector{Float32} }
    var_h_i::Union{ Nothing, Vector{Float32} }
    skew_h_i::Union{ Nothing, Vector{Float32} }
end


        
function Micro_Simulation_Annealed_fixedindegree(T, dt, Ne, Ni, Nrecord, dt_rec, Phi, param_Phi, param_lnp, seed_quenched; fixed_in_degree=true, ini_v=0, record_mean_h_i=false, 
                                                    record_var_h_i=false, mu_func=mu_func_0, ini_sig=0, record_skew_h_i=false, annealed_inefficient=false)
#= 
Despite the program name you need to specify whether you want fixed-indegree, which is really inefficient as it redraws the entire connectivity matri for each spike
=#

    if(seed_quenched == 0)
        rng = Random.MersenneTwister()
        println(rng)
        param_lnp["rng"] = rng
    else
        rng = Random.MersenneTwister(seed_quenched)
        param_lnp["rng"] = rng
    end
#    println("setting up parameters")
    #Ne = 0
    #Ni = 1000
    Ncells  = Ne + Ni
    #alpha=2.

    if(!haskey(param_lnp, "sig_mu_e")) #old versions of this program do not have this parameter, the parameter would be zero
        param_lnp["sig_mu_e"] = 0
    end
    sig_mu_e = param_lnp["sig_mu_e"]

    if(!haskey(param_lnp, "sig_mu_i"))
        param_lnp["sig_mu_i"] = 0
    end
    sig_mu_i = param_lnp["sig_mu_i"]
    
    taue    = param_lnp["taue"] #membrane time constant for exc. neurons (s)
    taui    = param_lnp["taui"]

    taus_e  = param_lnp["taus_e"] #synaptic time constant (s)
    taus_i  = param_lnp["taus_i"] #synaptic time constant (s)

    Etaus = zeros(Float16, Ncells)
    for m in 1:Ne
        if taus_e > 0
            Etaus[m] = exp(-dt / taus_e)
        end
    end
    for m in 1:Ni
        if taus_i > 0
            Etaus[Ne+m] = exp(-dt / taus_i)
        end
    end

    #connection probabilities
    p_i = param_lnp["p_i"]
    p_e = param_lnp["p_e"]
    
    Ke = round(Int, Ne * p_e)
    Ki = round(Int, Ni * p_i)
    
    jie = param_lnp["con_ie"] / Ke   
    jei = param_lnp["con_ei"] / Ki
    jii = param_lnp["con_ii"] / Ki 
    jee = param_lnp["con_ee"] / Ke
    
    #stimulation
    mu0 = param_lnp["mu0"]
    mu  = zeros(Ncells)
    mu[1:Ne] = mu0 .+ randn(rng, (Ne,)).*sqrt(sig_mu_e)
    mu[(Ne+1):Ncells] = mu0 .+ randn(rng, (Ni,)).*sqrt(sig_mu_i)
    
    delay = param_lnp["delay"]
    #n_delay = round(Int, delay/dt)
    n_delay = max(ceil(Int, delay/dt), 1)
        
    Nbin = round(Int, dt_rec/dt) #nin size for recording voltage and currents
    
    #membrane time constant
    tau = zeros(Ncells)
    dtau = zeros(Ncells)
    
    tau[1:Ne] .= taue
    tau[(1+Ne):Ncells] .= taui
    dtau[1:Ne] .= dt/taue
    dtau[(1+Ne):Ncells] .= dt/taui

#=
    weights = zeros(Ncells,Ncells)
    if(fixed_in_degree)
   
        #random excitatory connections
        pre_neurons = zeros(Int, Ne)
        pre_neurons[1:Ke] = ones(Int, Ke)
        for n in 1:Ne
            weights[n,1:Ne] = jee * pre_neurons[randperm(rng, Ne)] 
        end
        for n in Ne+1:Ne+Ni
            weights[n,1:Ne] = jie * pre_neurons[randperm(rng, Ne)] 
        end

        #random inhibitory connections
        pre_neurons = zeros(Int, Ni)
        pre_neurons[1:Ki] = ones(Int, Ki)
        for n in 1:Ne
            weights[n,Ne+1:Ncells] = jei * pre_neurons[randperm(rng, Ni)] # Vergleiche mit oben!
        end
        for n in Ne+1:Ncells
            weights[n,Ne+1:Ncells] = jii * pre_neurons[randperm(rng, Ni)]
        end
    else
        weights[1:Ne, 1:Ne]                 = jee .* rand(Bernoulli(p_e), (Ne,Ne))
        weights[1:Ne, 1+Ne:Ncells]          = jei .* rand(Bernoulli(p_i), (Ne,Ni))
        weights[1+Ne:Ncells, 1:Ne]          = jie .* rand(Bernoulli(p_e), (Ni,Ne))
        weights[1+Ne:Ncells, 1+Ne:Ncells]   = jii .* rand(Bernoulli(p_i), (Ni,Ni))
    end
=#
    if(annealed_inefficient)
        Jmatrix                             = zeros(Ncells, Ncells)
        Jmatrix[1:Ne, 1:Ne]                 = jee .* ones(Ne, Ne)
        Jmatrix[1:Ne, 1+Ne:Ncells]          = jie .* ones(Ne, Ni)                       
        Jmatrix[1+Ne:Ncells, 1:Ne]          = jei .* ones(Ni, Ne)                       
        Jmatrix[1+Ne:Ncells, 1+Ne:Ncells]   = jii .* ones(Ni, Ni)
    end
    ns = zeros(Int,Ncells)
    
    forwardInputsE = zeros(Ncells)
    forwardInputsI = zeros(Ncells)

    delaylineE = zeros(Ncells, n_delay)
    delaylineI = zeros(Ncells, n_delay)
    start = 1

    
    xe = zeros(Ncells)                                      # synaptic variable for excitatory input
    xi = zeros(Ncells)                                      # synaptic variable for inhibitory input

    #v = 0 * ones(Ncells)                                    # voltage of neurons
    #v = ini_v * ones(Ncells)
    v = ini_v * ones(Ncells) .+ randn(Ncells) .* sqrt(ini_sig) 
   #v = randn(Ncells).*20 .-30
    spikes_per_time_stepI = 0
    spikes_per_time_stepE = 0

    #quantities to be recorded
    Nsteps = round(Int,T/dt)
    Nsteps_rec = round(Int,T/dt_rec)
    # Ie_record = zeros(Float32,(Nrecord, Nsteps_rec))
    # Ii_record = zeros(Float32,(Nrecord, Nsteps_rec))
    # Itot_record = zeros(Float32,(Nrecord, Nsteps_rec))
    # Isyntot_record = zeros(Float32,(Nrecord, Nsteps_rec))
    v_record = zeros(Float32,(Nrecord, Nsteps_rec))         # voltage traces to be recorded
    if(record_mean_h_i)
        mean_h_i = zeros(Float32, Nsteps_rec)               # mean of distribution of voltage distribution for each recorded time step
    else
        mean_h_i = nothing
    end
    if(record_var_h_i)
        var_h_i = zeros(Float32, Nsteps_rec)               # mean of distribution of voltage distribution for each recorded time step
    else
        var_h_i = nothing
    end
    if(record_skew_h_i)
        skew_h_i = zeros(Float32, Nsteps_rec)               # mean of distribution of voltage distribution for each recorded time step
    else
        skew_h_i = nothing
    end
    rpopE_record    = zeros(Float32,Nsteps_rec)                # E population rate
    rpopI_record    = zeros(Float32,Nsteps_rec)                # I population rate
    spikes          = zeros(Int8, (Nrecord, Nsteps_rec))             # spike trains to be recorded
    actpopE_record  = zeros(Float32,Nsteps_rec)
    actpopI_record  = zeros(Float32,Nsteps_rec)
#    println("starting simulation")

    poprateE = 0
    poprateI = 0
    #begin main simulation loop
    for ti = 1:Nsteps
	    if mod(ti,Nsteps/100) == 1  #print percent complete
            @printf("\33[2K\r%d%% poprateE=%g poprateI=%g, abgelaufene Zeit=%g, rate=%g, Mu%g",round(Int,100*ti/Nsteps), poprateE, poprateI, ti*dt, sum(ns)/(Ni*ti*dt), mu0)
#            @printf("xe=%g xi=%g xetheo=%g xitheo=%g\n",xe[1],xi[1], poprateE*jee*taus*Ke, poprateI*jei*taus*Ki)
	    end
	    t = dt*ti
        i_rec = ceil(Int, ti/Nbin)
#        i_rec= floor(Int, t/dt_rec)+1
	    forwardInputsE[:] .= 0
	    forwardInputsI[:] .= 0
        spikes_per_time_stepI = 0
        spikes_per_time_stepE = 0

        ende = ((start + n_delay -2) % n_delay) + 1
        
        #mu = mu0                                            # just renaming

        poprateE = 0
        poprateI = 0

        # for all neurons the same external fluctuations
        mu_ext = mu_func(t, param_lnp)

	    for ci = 1:Ncells
            xe[ci] = xe[ci] * Etaus[ci] + delaylineE[ci, ende] / dt * (1 - Etaus[ci])
            xi[ci] = xi[ci] * Etaus[ci] + delaylineI[ci, ende] / dt * (1 - Etaus[ci])
	        # xe[ci] += -dt*xe[ci]/taus + delaylineE[ci, ende]  #mV
	        # xi[ci] += -dt*xi[ci]/taus + delaylineI[ci, ende]
                
	        synInputE = xe[ci]
            synInputI = xi[ci]
            synInput = synInputE + synInputI

            #println(v[ci])                
	        v[ci] += dtau[ci]*(mu[ci]+mu_ext-v[ci] + synInput)

            #rate = Phi(v[ci], Phi0, alpha)
            rate = Phi(v[ci], param_Phi)
            if (ci<=Ne)
                poprateE += rate #r_e
            else
                poprateI += rate #r_i
            end
            
	        if (rand() < rate * dt)  #spike occurred
		        ns[ci] = ns[ci]+1
                if(ci<=Ne)
                    spikes_per_time_stepE += 1
                else
                    spikes_per_time_stepI += 1
                end
                if (ci<=Nrecord)
                    spikes[ci, i_rec] = 1
                end

                if(fixed_in_degree)
                    weights = zeros(Ncells,Ncells)
                    pre_neurons = zeros(Int, Ne)
                    pre_neurons[1:Ke] = ones(Int, Ke)
                    for n in 1:Ne
                        weights[n,1:Ne] = jee * pre_neurons[randperm(rng, Ne)] 
                    end
                    for n in Ne+1:Ne+Ni
                        weights[n,1:Ne] = jie * pre_neurons[randperm(rng, Ne)] 
                    end

                    #random inhibitory connections
                    pre_neurons = zeros(Int, Ni)
                    pre_neurons[1:Ki] = ones(Int, Ki)
                    for n in 1:Ne
                        weights[n,Ne+1:Ncells] = jei * pre_neurons[randperm(rng, Ni)] # Vergleiche mit oben!
                    end
                    for n in Ne+1:Ncells
                        weights[n,Ne+1:Ncells] = jii * pre_neurons[randperm(rng, Ni)]
                    end
                end
                  
		        for j = 1:Ncells
                    if(fixed_in_degree)
                        w = weights[j,ci]
                    else
                        w = 0.
                        if(annealed_inefficient) # this is just legacy code, not wrong, but also very inefficient memory wise      
                            if(ci <= Ne)
                                if(rand() < p_e)
                                    w = Jmatrix[ci, j]
                                end
                            else
                                if(rand() < p_i)
                                    w = Jmatrix[ci, j]
                                end
                            end
                        else
                            if(ci <= Ne)
                                if(rand() < p_e)
                                    if(j <= Ne)
                                        w = jee
                                    else
                                        w = jie
                                    end 
                                end
                            else
                                if(rand() < p_i)
                                    if(j <= Ne)
                                        w = jei
                                    else
                                        w = jii
                                    end
                                end
                            end
                        end
                    end                 
		            if  w > 0  #E synapse
			            forwardInputsE[j] += w
		            elseif w < 0  #I synapse
			            forwardInputsI[j] += w
		            end
		        end #end loop over synaptic projections
	        end #end if(spike occurred)
            
            if (ci <= Nrecord)
                # Ie_record[ci,i_rec] += synInputE  # excitatory current in mV
                # Ii_record[ci,i_rec] += synInputI  # inhibitory current in mV
                # Isyntot_record[ci,i_rec] += synInput # total synaptic current in mV
                # Itot_record[ci,i_rec] += synInput + mu + eta[ci] # total current in mV
                v_record[ci,i_rec] += v[ci]                # subthreshold membrane potential
            end #end loop over recorded neurons
            
	    end #end loop over neurons

        spikes_per_time_stepE /= Ne
        spikes_per_time_stepI /= Ni
        actpopE_record[i_rec] += spikes_per_time_stepE
        actpopI_record[i_rec] += spikes_per_time_stepI        
        poprateE /= Ne
        poprateI /= Ni
        rpopE_record[i_rec] += poprateE
        rpopI_record[i_rec] += poprateI
        if(record_mean_h_i) # if we go over to IE-networks, this needs to be adapted!!!
            mean_h_i[i_rec] += mean(v)
        end
        if(record_var_h_i) # if we go over to IE-networks, this needs to be adapted!!!
            var_h_i[i_rec] += var(v)
        end
        if(record_skew_h_i) # if we go over to IE-networks, this needs to be adapted!!!
            skew_h_i[i_rec] += calculate_emphirical_skewness(v)
        end
        
        for j in 1:Ncells
            delaylineE[j, ende] = forwardInputsE[j]
            delaylineI[j, ende] = forwardInputsI[j]
        end #end loop over postsynaptic neurons
        start = ende
    end #end loop over time

    actpopI_record /= dt
    actpopE_record /= dt

    v_record /=Nbin
    rpopE_record /=Nbin
    rpopI_record /=Nbin
    actpopE_record /=Nbin
    actpopI_record /=Nbin

    if(record_mean_h_i)        
        mean_h_i /=Nbin
    end
    if(record_var_h_i)
        var_h_i /=Nbin  # Mean variance of h_i over the Nbins
    end
    if(record_skew_h_i)
        skew_h_i /=Nbin  # Mean variance of h_i over the Nbins
    end

    @printf("\r")
    return Micro_Simulation_Annealed_fixedindegree_Results(ns, v_record, spikes, rpopE_record, rpopI_record, actpopE_record, actpopI_record, mean_h_i, var_h_i, skew_h_i)
end
