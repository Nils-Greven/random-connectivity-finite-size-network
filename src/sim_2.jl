mutable struct Micro_Simulation_Results
    ns::Union{ Nothing, Vector{Int} }                                       #Int8???
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
    
    jie = param_lnp["con_ie"] / Ke   # mV ??? not mVs?
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
            # push!(delaylineE[j], forwardInputsE[j]) #add total input to delay line
            # push!(delaylineI[j], forwardInputsI[j])
            delaylineE[j, ende] = forwardInputsE[j]
            delaylineI[j, ende] = forwardInputsI[j]
        end #end loop over postsynaptic neurons
        start = ende
    end #end loop over time

    actpopI_record /= dt
    actpopE_record /= dt

    # Ie_record /=Nbin
    # Ii_record /=Nbin
    # Itot_record /=Nbin
    # Isyntot_record /=Nbin
    v_record /=Nbin
    rpopE_record /=Nbin
    rpopI_record /=Nbin
    actpopE_record /=Nbin
    actpopI_record /=Nbin
    #nbins   = 25
    #hist    = fit(Histogram, v, nbins=nbins)
    #hist    = normalize(hist, mode=:pdf)
    #kurt    = calculate_emphirical_curtosis(v)
    #skew    = calculate_emphirical_skewness(v)
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


mutable struct Micro_Simulation_Results_Deleteme
    ns::Union{ Nothing, Vector{Int} }                                       #Int8???
    v_record::Union{ Nothing, Matrix{Float64}, Vector{Vector{Float64}} }
    spikes::Union{ Nothing, Matrix{Int8} }
    rpopE_record::Union{ Nothing, Vector{Float64} }
    rpopI_record::Union{ Nothing, Vector{Float64} }
    actpopE_record::Union{ Nothing, Vector{Float64} }
    actpopI_record::Union{ Nothing, Vector{Float64} }
    mean_h_i::Union{ Nothing, Vector{Float64} }
    var_h_i::Union{ Nothing, Vector{Float64} }
    skew_h_i::Union{ Nothing, Vector{Float64} }
end


        
function micro_simulation_Deleteme(T, dt, Ne, Ni, Nrecord, dt_rec, Phi, param_Phi, param_lnp, seed_quenched;fixed_in_degree=true, ini_v=0, record_mean_h_i=false, mu_func=mu_func_0, ini_sig=0, record_var_h_i=false, record_skew_h_i=false)
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
    
    jie = param_lnp["con_ie"] / Ke   # mV ??? not mVs?
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
    v_record = zeros(Float64,(Nrecord, Nsteps_rec))         # voltage traces to be recorded
    if(record_mean_h_i)
        mean_h_i = zeros(Float64, Nsteps_rec)               # mean of distribution of voltage distribution for each recorded time step
    else
        mean_h_i = nothing
    end
    if(record_var_h_i)
        var_h_i = zeros(Float64, Nsteps_rec)               # mean of distribution of voltage distribution for each recorded time step
    else
        var_h_i = nothing
    end
    if(record_skew_h_i)
        skew_h_i = zeros(Float64, Nsteps_rec)               # mean of distribution of voltage distribution for each recorded time step
    else
        skew_h_i = nothing
    end
    rpopE_record    = zeros(Float64,Nsteps_rec)                # E population rate
    rpopI_record    = zeros(Float64,Nsteps_rec)                # I population rate
    spikes          = zeros(Int8, (Nrecord, Nsteps_rec))             # spike trains to be recorded
    actpopE_record  = zeros(Float64,Nsteps_rec)
    actpopI_record  = zeros(Float64,Nsteps_rec)
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
            # push!(delaylineE[j], forwardInputsE[j]) #add total input to delay line
            # push!(delaylineI[j], forwardInputsI[j])
            delaylineE[j, ende] = forwardInputsE[j]
            delaylineI[j, ende] = forwardInputsI[j]
        end #end loop over postsynaptic neurons
        start = ende
    end #end loop over time

    actpopI_record /= dt
    actpopE_record /= dt

    # Ie_record /=Nbin
    # Ii_record /=Nbin
    # Itot_record /=Nbin
    # Isyntot_record /=Nbin
    v_record /=Nbin
    rpopE_record /=Nbin
    rpopI_record /=Nbin
    actpopE_record /=Nbin
    actpopI_record /=Nbin
    #nbins   = 25
    #hist    = fit(Histogram, v, nbins=nbins)
    #hist    = normalize(hist, mode=:pdf)
    #kurt    = calculate_emphirical_curtosis(v)
    #skew    = calculate_emphirical_skewness(v)
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
    return Micro_Simulation_Results_Deleteme(ns, v_record, spikes, rpopE_record, rpopI_record, actpopE_record, actpopI_record, mean_h_i, var_h_i, skew_h_i)
end



mutable struct Micro_Simulation_Annealed_reset_Results
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


        
function Micro_Simulation_Annealed_reset(T, dt, Ne, Ni, Nrecord, dt_rec, Phi, param_Phi, param_lnp, seed_quenched; fixed_in_degree=true, ini_v=0, record_mean_h_i=false, 
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

    v_R = param_lnp["v_R"]             # reset voltage
    #v_R = 0
    #println(v_R)
    rough_prob_estimate_yn = param_lnp["rough_prob_estimate_yn"]

    tau_ref = param_lnp["tau_ref"]

    t_ref_vec = ones(Ncells)*tau_ref


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
    
    jie = param_lnp["con_ie"] / Ke   # mV ??? not mVs?
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
        Jmatrix[1:Ne, 1+Ne:Ncells]          = jie .* ones(Ne, Ni)                       # This probably needs fixing
        Jmatrix[1+Ne:Ncells, 1:Ne]          = jei .* ones(Ni, Ne)                       # This probably needs fixing
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
        t_ref_vec = t_ref_vec .+ dt
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

           rate = Phi(v[ci], param_Phi)
            if (ci<=Ne)
                poprateE += rate #r_e
            else
                poprateI += rate #r_i
            end
            
            if(rough_prob_estimate_yn)
                rough_est_prob = rate * dt
            else
                rough_est_prob = 1.0 - exp(-rate * dt)
            end

	        #if (rand() < rate * dt)  #spike occurred
	        #if (rand() < rough_est_prob)  #spike occurred
            if ((rand() < rough_est_prob) && (t_ref_vec[ci] > tau_ref))  #spike occurred
		        ns[ci] = ns[ci]+1
                if(ci<=Ne)
                    spikes_per_time_stepE += 1
                    v[ci] = v_R
                    t_ref_vec[ci] = 0
                else
                    spikes_per_time_stepI += 1
                    v[ci] = v_R
                    t_ref_vec[ci] = 0
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
    return Micro_Simulation_Annealed_reset_Results(ns, v_record, spikes, rpopE_record, rpopI_record, actpopE_record, actpopI_record, mean_h_i, var_h_i, skew_h_i)
end


mutable struct Micro_Simulation_Annealed_SRM_Results
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


function Micro_Simulation_Annealed_SRM(T, dt, Ne, Ni, Nrecord, dt_rec, Phi, param_Phi, param_lnp, seed_quenched; fixed_in_degree=true, ini_v=0, record_mean_h_i=false, 
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

   dv_reset = param_lnp["dv_reset"]              # reset voltage by dv_reset at spike
    #v_R = 0
    #println(v_R)
    rough_prob_estimate_yn = param_lnp["rough_prob_estimate_yn"]

    tau_ref = param_lnp["tau_ref"]

    t_ref_vec = ones(Ncells)*tau_ref


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
    
    jie = param_lnp["con_ie"] / Ke   # mV ??? not mVs?
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
        Jmatrix[1:Ne, 1+Ne:Ncells]          = jie .* ones(Ne, Ni)                       # This probably needs fixing
        Jmatrix[1+Ne:Ncells, 1:Ne]          = jei .* ones(Ni, Ne)                       # This probably needs fixing
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
        t_ref_vec = t_ref_vec .+ dt
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

           rate = Phi(v[ci], param_Phi)
            if (ci<=Ne)
                poprateE += rate #r_e
            else
                poprateI += rate #r_i
            end
            
            if(rough_prob_estimate_yn)
                rough_est_prob = rate * dt
            else
                rough_est_prob = 1.0 - exp(-rate * dt)
            end

	        #if (rand() < rate * dt)  #spike occurred
	        #if (rand() < rough_est_prob)  #spike occurred
            if ((rand() < rough_est_prob) && (t_ref_vec[ci] > tau_ref))  #spike occurred
		        ns[ci] = ns[ci]+1
                if(ci<=Ne)
                    spikes_per_time_stepE += 1
                    v[ci] = v[ci] - dv_reset
                    t_ref_vec[ci] = 0
                else
                    spikes_per_time_stepI += 1
                    v[ci] = v[ci] - dv_reset
                    t_ref_vec[ci] = 0
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
    return Micro_Simulation_Annealed_SRM_Results(ns, v_record, spikes, rpopE_record, rpopI_record, actpopE_record, actpopI_record, mean_h_i, var_h_i, skew_h_i)
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
    
    jie = param_lnp["con_ie"] / Ke   # mV ??? not mVs?
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
        Jmatrix[1:Ne, 1+Ne:Ncells]          = jie .* ones(Ne, Ni)                       # This probably needs fixing
        Jmatrix[1+Ne:Ncells, 1:Ne]          = jei .* ones(Ni, Ne)                       # This probably needs fixing
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


mutable struct Micro_Simulation_large_N_Results
    ns::Union{ Nothing, Vector{Int} }                                       #Int8???
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

function Micro_Simulation_large_N(T, dt, Ne, Ni, Nrecord, dt_rec, Phi, param_Phi, param_lnp, seed_quenched;fixed_in_degree=true, ini_v=0, ini_sig=0, record_mean_h_i=false, record_var_h_i=false,
                                    record_skew_h_i=false, mu_func=mu_func_0)
#=
Simulate microscopic model of spiking neurons, in general an IE-network. This is a mirror version of micro_simulation,
but optimized for large (sparse) network: only save connections not the entire matrix
=#

    if(seed_quenched == 0)
        rng =   Random.MersenneTwister()
        param_lnp["rng"] = rng
    else
        rng     = Random.MersenneTwister(seed_quenched)
        rng2    = Random.MersenneTwister()
        param_lnp["rng"] = rng2
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
    
    jie = param_lnp["con_ie"] / Ke   # mV ??? not mVs?
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

    # set up vector of vectos:
    #con_vec = [[] for _ in 1:(Ni+Ne)]
    con_vec = [Vector{UInt16}() for _ in 1:(Ni + Ne)]

    if(fixed_in_degree)
        # loop over all neurons: from which neurons do the receive spikes
        for n in (1:Ncells)
            
            # excitatory neurons: from which neurons do they receive spikes?
            list_innervating_neuron_n = sample(rng, 1:Ne, Ke, replace=false)

            # reverse that: make list of which neurons get a spike from neuron j:
            for j in list_innervating_neuron_n
                #push!(con_vec[j], n)
                push!(con_vec[j], UInt16(n))
            end

            # same as above, just inhibitory neurons
            list_innervating_neuron_n = sample(rng, (Ne+1):(Ni+Ne), Ki, replace=false)
        
            # same as above, just inhibitory neurons
            for j in list_innervating_neuron_n
                #ush!(con_vec[j], n)
                push!(con_vec[j], UInt16(n))
            end
        end
    else
        # loop over all neurons: from which neurons do the receive spikes
        for n in (1:Ncells)

            # excitatory neurons: from which neurons do they receive spikes?
            list_innervating_neuron_n = [i for i in 1:Ne if rand(rng) < p_e]

            # reverse that: make list of which neurons get a spike from neuron j:
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end

            # same as above, just inhibitory neurons
            list_innervating_neuron_n = [i for i in (Ne+1):(Ne+Ni) if rand(rng) < p_i]
        
            # same as above, just inhibitory neurons
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end
        end
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
    spikes_per_time_stepI = 0
    spikes_per_time_stepE = 0

    #quantities to be recorded
    Nsteps = round(Int,T/dt)
    Nsteps_rec = round(Int,T/dt_rec)
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
            #@printf("\33[2K\r%d%% poprateE=%g poprateI=%g, abgelaufene Zeit=%g, rate=%g, Mu%g",round(Int,100*ti/Nsteps), poprateE, poprateI, ti*dt, sum(ns)/(Ni*ti*dt), mu0)
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
                
	        v[ci] += dtau[ci]*(mu[ci]+mu_ext-v[ci] + synInput)

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
              
                if(ci <= Ne) # from excitatory ...
                    for cj in con_vec[ci]
                        if(cj <= Ne) # ... to excitatory
                            forwardInputsE[cj] += jee
                        else # ... to inhibitory
                            forwardInputsE[cj] += jie
                        end
                    end
                else  # from inhibitory ...
                    for cj in con_vec[ci]
                        if(cj <= Ne) # ... to excitatory
                            forwardInputsI[cj] += jei
                        else # ... to inhibitory
                            forwardInputsI[cj] += jii
                        end
                    end
                end
	        end #end if(spike occurred)
            
            if (ci <= Nrecord)
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
    return Micro_Simulation_large_N_Results(ns, v_record, spikes, rpopE_record, rpopI_record, actpopE_record, actpopI_record, mean_h_i,var_h_i, skew_h_i)
end


mutable struct Euler_integrator_second_layer_quick_Results
    h::Union{ Nothing, Vector{Float32} }
    s::Union{ Nothing, Vector{Float32} }
    h1::Union{ Nothing, Vector{Float32} }
    A::Union{ Nothing, Vector{Float32} }
    r::Union{ Nothing, Vector{Float32} }
end


function Euler_integrator_second_layer_quick(h0, s0, h10, f_h, f_s, F, para_Euler_integrator; supress_noise_yn=false, give_out_xi=false)
#=
    F       : full integral of Gaussian distribution over transfer function
=#
    tmax    = para_Euler_integrator["tmax"]
    dt      = para_Euler_integrator["dt"]
    delta   = para_Euler_integrator["delta"]
    N       = para_Euler_integrator["N"]
    mu0     = para_Euler_integrator["mu0"]
    w       = para_Euler_integrator["w"]
    C       = para_Euler_integrator["p"]*N
    tau     = para_Euler_integrator["tau"]
    nSteps  = trunc(Int, tmax/dt)
    h       = zeros(nSteps+1)
    h1      = zeros(nSteps+1)
    s       = zeros(nSteps+1)
    A       = zeros(nSteps+1)
    r       = zeros(nSteps+1)
    noise   = zeros(nSteps+1)
    if(!supress_noise_yn)
        rng = Random.MersenneTwister()
        noise = randn(rng, Float64, (1, nSteps+1))
    end
    delay   = trunc(Int, delta/dt)
    h[1:delay+1] .= h0
    h1[1:delay+1].= h10
    s[1:delay+1] .= s0
    A[1:delay+1] .= F(h0, s0, para_Euler_integrator)
    r[1:delay+1] .= F(h0, s0, para_Euler_integrator)

    for i=1:(nSteps-delay)
        if(s[i]<0)
            println("here")
        end
        h[i+1+delay]    = h[i+delay] + f_h(h[i+delay], s[i+delay], r[i], para_Euler_integrator, noise=noise[i]) * dt
        #h[i+1+delay]    = h[i+delay] + ( -h[i+delay] + mu0 + w * F(h[i], s[i], para_Euler_integrator) + w *0* sqrt(F(h[i], s[i], para_Euler_integrator)/N) * noise[i]/sqrt(dt) ) * dt/tau
        s[i+1+delay]    = s[i+delay] + f_s(h[i+delay], s[i+delay], r[i], para_Euler_integrator, noise=noise[i]) * dt
        #s[i+1+delay]    = s[i+delay] + (-2*s[i+delay] + w^2*(1-C/N)/(tau * C)*F(h[i], s[i], para_Euler_integrator))

        h1[i+1+delay]   = h1[i+delay] + ( -h1[i+delay] + mu0 + w * F(h[i], s[i], para_Euler_integrator) ) * dt/tau
        if(s[i+1+delay]<0)
            print("here")
        end
        r_inter         = F(h[i+1+delay], s[i+1+delay], para_Euler_integrator)
        r[i+1+delay]    = r_inter
        A[i+1+delay]    = (r_inter + sqrt(r_inter/N)*noise[i+1+delay]/sqrt(dt))
    end
    Euler_integrator_second_layer_quick_Results(h, s, h1, A, r)
end


function F_taylor_correction_OLD(h, s, xi, param)
    F           = param["F_smoothed"]               # is a function F(h, s, param)
    delFdels    = param["delFdels"]                 # is a function dFds(h, s, param)
    delphidelh  = param["phi_prime"]                # is a function delphidelh(h, s, param)

    F_val = F(h, s, param)
    return  F_val + param["w"] * sqrt((1-param["p"])/(param["C"]*param["N"]) * F_val) * xi * delphidelh(h, param) - delFdels(h, 0, param)*s
end

function del_sigm_erf_del_h(h, param)
    beta = param["beta"]
    return 0.5 /sqrt(2*pi) * param["Phimax"] * beta *exp(-beta^2*(h-theta)^2/2)
end

mutable struct mesoscopic_model_correction_colored_noise_OLD_Results
    h::Union{ Nothing, Vector{Float64} }
    s::Union{ Nothing, Vector{Float64} }
    A::Union{ Nothing, Vector{Float64} }
    r::Union{ Nothing, Vector{Float64} }
    r_neg::Union{ Nothing, Float64 }
end

function mesoscopic_model_correction_colored_noise_OLD(h0, s0, xi0, F_corrected, param, supress_eta_yn=false, supress_zeta_yn=false, r_replace=0)

    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = param["N"]
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = param["p"]
    tau     = param["tau"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)

    r_neg   = 0.                                    # counter for how often r was negative
                                                    # aka how often did I need to intervene,
                                                    # I replace the negative r with zero  
    h       = zeros(nSteps+1)
    s       = zeros(nSteps+1)
    A       = zeros(nSteps+1)
    r       = zeros(nSteps+1)
    xi      = zeros(nSteps+1)                       # colored noise
    eta     = zeros(nSteps+1)                       # Noise term finite-size noise
    zeta    = zeros(nSteps+1)                       # Noise term colored noise correction
    if(!supress_eta_yn)
        rng = Random.MersenneTwister()
        eta = randn(rng, Float64, (1, nSteps+1))
    end
    if(!supress_zeta_yn)
        zeta = randn(rng, Float64, (1, nSteps+1))
    end

    h[1:delay+1]    .= h0
    s[1:delay+1]    .= s0
    xi[1:delay+1]   .= xi0
    A[1:delay+1]    .= F_corrected(h0, s0, xi0, param)
    r[1:delay+1]    .= F_corrected(h0, s0, xi0, param)

    for i=1:(nSteps-delay)
        h[i+1+delay]    = h[i+delay] + dt/tau * ( -h[i+delay] + mu0 + w * A[i] )
        s[i+1+delay]    = s[i+delay] + dt/tau * ( -2*s[i+delay] +w^2/tau * (1-p)/C * r[i] )
        xi[i+1+delay]   = xi[i+delay]+ dt/tau * ( -xi[i+delay] + zeta[i+delay]/sqrt(dt) )


        r_inter         = F_corrected(h[i+1+delay], s[i+1+delay], xi[i+1+delay], param)
        if(r_inter < 0)
            r_neg  += 1
            r_inter = r_replace         
        end
        r[i+1+delay]    = r_inter
        A[i+1+delay]    = r_inter + sqrt(r_inter/N) * eta[i+1+delay]/sqrt(dt)
    end
    println(r_neg)
    #println()
return mesoscopic_model_correction_colored_noise_OLD_Results(h, s, A, r, r_neg)
end

function calculate_fixed_point_sigmoidal_erf_color_corrected_OLD(param, hmin, hmax)
    w   = param["w"]
    mu0 = param["mu0"]
    if((abs(w) < 1e-10))
        return mu0
    end
    Phimax  = param["Phimax"]
    N       = param["N"]
    tau     = param["tau"]
    beta    = param["beta"]
    theta   = param["theta"]
    p       = param["p"]
    A = w*(1-p)/(2*tau*N*p)

    f = x-> Phimax*sigm_erf(beta*(x-theta)/(sqrt(abs(1+beta^2*A*(x-mu0)))))+Phimax/sqrt(8*pi)*exp(-0.5*(x-theta)^2*beta^2)*beta^3*(x-theta)*A*(x-mu0)-(x-mu0)/w

    rts = find_zeros(f, (hmin, hmax), atol=1e-10, rtol=1e-10)
    #rts = roots(f, hmin..hmax)
    return rts
end


function G_sigm_erf(h, s, param)
    # needs scipy_special.py to be loaded
    beta = param["beta"]
    Phimax = param["Phimax"]
    F = F_sigm_erf(h, s, param)
    return Phimax*F - 2*Phimax^2*scipy_special.O_T(beta*(h-param["theta"])/sqrt(1+beta^2*s), 1/sqrt(1+2*beta^2*s)) -F^2
end


mutable struct mesoscopic_model_correction_colored_noise_Results
    h::Union{ Nothing, Vector{Float64} }
    s::Union{ Nothing, Vector{Float64} }
    A::Union{ Nothing, Vector{Float64} }
    r::Union{ Nothing, Vector{Float64} }
    r_neg::Union{ Nothing, Float64 }
    X::Union{ Nothing, Vector{Float64} }
end

function mesoscopic_model_correction_colored_noise(h0, s0, X0, F, G, param, supress_eta_yn=false, supress_zeta_yn=false, r_replace=0)

    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = param["N"]
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = param["p"]
    tau     = param["tau"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)

    r_neg   = 0.                                    # counter for how often r was negative
                                                    # aka how often did I need to intervene,
                                                    # I replace the negative r with zero  
    h       = zeros(nSteps+1)
    s       = zeros(nSteps+1)
    A       = zeros(nSteps+1)
    r       = zeros(nSteps+1)
    X       = zeros(nSteps+1)                       # Fluctuating correction to the firing rate 
    slamb   = zeros(nSteps+1)                       # Assume X is an OU process, slamb is sigma_lambda the variance
    eta     = zeros(nSteps+1)                       # Noise term finite-size noise
    zeta    = zeros(nSteps+1)                       # Noise term for the OU for X
    if(!supress_eta_yn)
        rng = Random.MersenneTwister()
        eta = randn(rng, Float64, (1, nSteps+1))
    end
    if(!supress_zeta_yn)
        zeta = randn(rng, Float64, (1, nSteps+1))
    end

    h[1:delay+1]    .= h0
    s[1:delay+1]    .= s0
    X[1:delay+1]    .= X0
    A[1:delay+1]    .= F(h0, s0, param)
    r[1:delay+1]    .= F(h0, s0, param)
    slamb[1:delay+1].= G(h0, s0, param)

    for i=1:(nSteps-delay)
        h[i+1+delay]    = h[i+delay] + dt/tau * ( -h[i+delay] + mu0 + w * A[i] )
        s[i+1+delay]    = s[i+delay] + dt/tau * ( -2*s[i+delay] +w^2/tau * (1-p)/C * r[i] )
        X[i+1+delay]    = X[i+delay] + dt/tau * ( -X[i+delay] + sqrt(2*tau*slamb[i+delay])*zeta[i+delay]/sqrt(dt) )

        slamb[i+1+delay]    = G(h[i+1+delay], s[i+1+delay], param)
        r_inter             = F(h[i+1+delay], s[i+1+delay], param) + 1/sqrt(N)*X[i+1+delay]
        if(r_inter < 0)
            r_neg  += 1
            r_inter = r_replace         
        end
        r[i+1+delay]        = r_inter
        A[i+1+delay]        = r_inter + sqrt(r_inter/N) * eta[i+1+delay]/sqrt(dt)
    end
    println(r_neg)
    #println()
    return mesoscopic_model_correction_colored_noise_Results(h, s, A, r, r_neg, X)
end

function mu_func_sin(t, param)
    return param["mu_a"]*sin(param["mu_om"]*t+param["mu_phi"])+param["mu_const"]
end

function mu_func_cos(t, param)
    return param["mu_a"]*cos(param["mu_om"]*t+param["mu_phi"])+param["mu_const"]
end

function mu_func_sin_step(t, param)
        if(t < param["mu_threshold"])
        return 0.
    else
        return param["mu_a"]*sin(param["mu_om"]*t+param["mu_phi"])+param["mu_const"]
    end
end

function mu_func_step(t, param)
    if(t < param["mu_threshold"])
        return 0.
    else
        return param["mu_const"]
    end
end

function mu_func_delta(t, param)
    # use mu_t_jump = dt if you want delta(t)
    dt = param["mu_dt"]
    t_j= param["mu_t_jump"]    
    if(((t-dt/2) <= t_j) && (t_j <= (t+dt/2)))
        print("here")
        return param["mu_a"]/dt
    else
        return 0
    end
end

function mu_func_0(t, param)
    return 0
end

function mu_func_WN(t, param)
    i = round(Int, t/param["dt"])
    return param["noise_vec"][i]
end

function mu_func_block(t, param)
    if(t<= param["mu_thresh_1"])
        return param["mu_C1"]
    elseif(t<= param["mu_thresh_2"])
        return param["mu_C2"]
    else
        return param["mu_C3"]
    end
end


function mu_func_block_and_WN(t, param)
    i = round(Int, t/param["dt"])
    if(t<= param["mu_thresh_1"])
        return param["mu_C1"]+param["noise_vec"][i]
    elseif(t<= param["mu_thresh_2"])
        return param["mu_C2"]+param["noise_vec"][i]
    else
        return param["mu_C3"]+param["noise_vec"][i]
    end
end

function mu_func_block_and_WN_draw_direct(t, param) # draw the random number directly here
    i = round(Int, t/param["dt"])
    if(t<= param["mu_thresh_1"])
        return param["mu_C1"]+sqrt(param["tau"]*param["mu_sigma_ext"])*randn(param["rng"], Float64)/sqrt(param["dt"])
    elseif(t<= param["mu_thresh_2"])
        return param["mu_C2"]+sqrt(param["tau"]*param["mu_sigma_ext"])*randn(param["rng"], Float64)/sqrt(param["dt"])
    else
        return param["mu_C3"]+sqrt(param["tau"]*param["mu_sigma_ext"])*randn(param["rng"], Float64)/sqrt(param["dt"])
    end
end
"""
function mu_func_block_and_WN_draw_direct(t, param) # draw the random number directly here
    i = round(Int, t/param["dt"])
    if(t<= param["mu_thresh_1"])
        return param["mu_C1"]+sqrt(param["mu_sigma_ext"])*randn()
    elseif(t<= param["mu_thresh_2"])
        return param["mu_C2"]+sqrt(param["mu_sigma_ext"])*randn()
    else
        return param["mu_C3"]+sqrt(param["mu_sigma_ext"])*randn()
    end
end
"""
function F_sigm_erf_mui(h, s, param)
    return F_sigm_erf(h, s+param["sig_mu_i"], param)
end

function G_sigm_erf_mui(h, s, param)
    return G_sigm_erf(h, s+param["sig_mu_i"], param)
end

function kronecker_delta(i, j)
    if(i==j)
        return 1
    end
    return 0
end

function mesoscopic_model_correction_colored_noise_time_dep_stimulus(h0, s0, X0, F, G, mu_func, param; supress_eta_yn=false, supress_zeta_yn=false, r_replace=0,
                                                                        model3d=1, naive_yn=false, div_N=1)
#=
Same function as above, bu this time with a $\mu = \mu(t)$
model3d set to zero to get the model witout the colored finite size correction. I also do this in the naive model to ensure that G(h, 0)=0 without rounding errors
=#
    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = param["N"]
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = param["p"]
    tau     = param["tau"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)
    if(!haskey(param, "extra_scaling"))
        param["extra_scaling"] = false
    end

    if(param["extra_scaling"])
        C = param["extra_C"]
        p = 0
        div_N = 0
    end

    r_neg   = 0.                                    # counter for how often r was negative
                                                    # aka how often did I need to intervene,
                                                    # I replace the negative r with zero  
    h       = zeros(nSteps+1)
    s       = zeros(nSteps+1)
    A       = zeros(nSteps+1)
    r       = zeros(nSteps+1)
    X       = zeros(nSteps+1)                       # Fluctuating correction to the firing rate 
    slamb   = zeros(nSteps+1)                       # Assume X is an OU process, slamb is sigma_lambda the variance
    eta     = zeros(nSteps+1)                       # Noise term finite-size noise
    zeta    = zeros(nSteps+1)                       # Noise term for the OU for X

    if(!supress_eta_yn)
        rng = Random.MersenneTwister()
        param["rng"] = rng
        eta = randn(rng, Float64, (1, nSteps+1))
    end
    if(!supress_zeta_yn)
        zeta = randn(rng, Float64, (1, nSteps+1))
    end

    X0 = X0*model3d
    if(naive_yn)
        s0 = 0
        model3d = 0
        X0 = 0
    end

    h[1:delay+1]    .= h0
    s[1:delay+1]    .= s0
    X[1:delay+1]    .= X0
    A[1:delay+1]    .= F(h0, s0, param)
    r[1:delay+1]    .= F(h0, s0, param)
    G0               = G(h0, s0, param)*model3d
    if(abs(G0) < 1e-10)
        G0 = 0
    end
    slamb[1:delay+1].= G0#abs(G(h0, s0, param))*model3d    # apparently there can be numerical errors like -1.124910706472534e-264 \approx 0
    #println(G(h0, s0, param)*model3d)

    for i=1:(nSteps-delay)
        h[i+1+delay]    = h[i+delay] + dt/tau * ( -h[i+delay] + mu0 + mu_func((i+delay)*dt, param) + w * A[i] )
        s[i+1+delay]    = s[i+delay] + dt/tau * ( -2*s[i+delay] +w^2/tau * (1-p)/C * r[i] )
        X[i+1+delay]    = X[i+delay] + dt/tau * ( -X[i+delay] + sqrt(2*tau*slamb[i+delay])*zeta[i+delay]/sqrt(dt) )

        if(naive_yn)
            s[i+1+delay] = 0
            X[i+1+delay] = 0
        end

        slamb[i+1+delay]    = G(h[i+1+delay], s[i+1+delay], param)*model3d
        #if(abs(slamb[i+1+delay]) < 1e-10)
        if(slamb[i+1+delay] < 0)
            #println(slamb[i+1+delay])
            slamb[i+1+delay] = 0
        end
        r_inter             = F(h[i+1+delay], s[i+1+delay], param) + 1/sqrt(N)*X[i+1+delay]*model3d*div_N
        if(r_inter < 0)
            r_neg  += 1
            r_inter = r_replace         
        end
        r[i+1+delay]        = r_inter
        A[i+1+delay]        = r_inter + sqrt(r_inter/N) * eta[i+1+delay]/sqrt(dt)*div_N
    end
    println(r_neg)
    #println()
    return mesoscopic_model_correction_colored_noise_Results(h, s, A, r, r_neg, X)
end

mutable struct mesoscopic_model_correction_colored_noise_Results_with_delayline
    h::Union{ Nothing, Vector{Float64} }
    s::Union{ Nothing, Vector{Float64} }
    A::Union{ Nothing, Vector{Float64} }
    r::Union{ Nothing, Vector{Float64} }
    r_neg::Union{ Nothing, Float64 }
    X::Union{ Nothing, Vector{Float64} }
    eta::Union{ Nothing, Vector{Float64} }
end


function mesoscopic_model_correction_colored_noise_time_dep_stimulus_and_delayline(h0, s0, X0, F, G, mu_func, param; eta_factor=1, zeta_factor=1, r_replace=0,
                                                                        model3d=1, naive_yn=false, div_N=1, record_eta_yn=true)
#=
Same function as above, but this time I have a "delay-line", now I do not need to save the entire time series. THIS PROGRAM IS NOT FINISHED!!!
=#
    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = param["N"]
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = param["p"]
    tau     = param["tau"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)+1
    dt_rec  = param["dt_rec"]
    if(dt_rec < dt)
        print("dt_rec is set too small, now reset to dt")
        dt_rec = dt
    end
    n_rec   = round(dt_rec/dt)
    N_rec   = Int(ceil(tmax/dt_rec))

    h_dline     = zeros(delay)
    s_dline     = zeros(delay)
    X_dline     = zeros(delay)
    r_dline     = zeros(delay)
    A_dline     = zeros(delay)
    slamb_dline = zeros(delay)
    h_rec   = zeros(N_rec)
    s_rec   = zeros(N_rec)
    X_rec   = zeros(N_rec)
    r_rec   = zeros(N_rec)
    A_rec   = zeros(N_rec)
    if(record_eta_yn)
        eta_rec = zeros(N_rec)
    else
        eta_rec = Nothing
    end

    pos_now = delay
    pos_del = 1


    if(!haskey(param, "extra_scaling"))
        param["extra_scaling"] = false
    end

    if(param["extra_scaling"])
        C = param["extra_C"]
        p = 0
        div_N = 0
    end

    r_neg   = 0.                                    # counter for how often r was negative
                                                    # aka how often did I need to intervene,
                                                    # I replace the negative r with zero  
                                                    # Fluctuating correction to the firing rate 

    rng = Random.MersenneTwister()
    param["rng"] = rng


    X0 = X0*model3d
    if(naive_yn)
        s0 = 0
        model3d = 0
        X0 = 0
    end

    r0 = F(h0, s0, param)
    G0 = G(h0, s0, param)*model3d
    if(abs(G0) < 1e-10)
        G0 = 0
    end

    for i=1:delay
        h_dline[i] = h0
        s_dline[i] = s0
        X_dline[i] = X0
        r_dline[i] = r0
        A_dline[i] = r0
        slamb_dline[i] = G0
    end

    h_inter = 0
    s_inter = 0
    X_inter = 0
    r_inter = 0
    slamb_inter = 0

    for i=1:(nSteps)
        pos_write = (pos_now+1)%delay
        if(pos_write==0)            # we go through the delayline from 1 to max-value not 0 to max-value -1 
            pos_write=delay
        end
        i_rec       = ceil(Int, i/n_rec)

        zeta = randn(rng, Float64)*eta_factor
        eta  = randn(rng, Float64)*zeta_factor

        #h_inter    = h_dline[pos_now] + dt/tau * ( -h_dline[pos_now] + mu0 + mu_func((i+delay)*dt, param) + w * A_dline[pos_del] )
        h_inter    = h_dline[pos_now] + dt/tau * ( -h_dline[pos_now] + mu0 + mu_func(i*dt, param) + w * A_dline[pos_del] )
        s_inter    = s_dline[pos_now] + dt/tau * ( -2*s_dline[pos_now] +w^2/tau * (1-p)/C * r_dline[pos_del] )
        X_inter    = X_dline[pos_now] + dt/tau * ( -X_dline[pos_now] + sqrt(2*tau*slamb_dline[pos_now])*zeta/sqrt(dt) )

        if(naive_yn)
            s_inter = 0
            X_inter = 0
        end

        slamb_inter    = G(h_inter, s_inter, param)*model3d
        if(slamb_inter < 0)
            slamb_inter = 0
        end
        r_inter = F(h_inter, s_inter, param) + 1/sqrt(N)*X_inter*model3d*div_N
        if(r_inter < 0)
            r_neg  += 1
            r_inter = r_replace         
        end
        r_dline[pos_write] = r_inter
        A_dline[pos_write] = r_inter + sqrt(r_inter/N) * eta/sqrt(dt)*div_N
        h_dline[pos_write] = h_inter
        s_dline[pos_write] = s_inter
        X_dline[pos_write] = X_inter

        slamb_dline[pos_write] = slamb_inter

        h_rec[i_rec]    += h_inter/n_rec
        s_rec[i_rec]    += s_inter/n_rec
        X_rec[i_rec]    += X_inter/n_rec
        r_rec[i_rec]    += r_inter/n_rec
        A_rec[i_rec]    += A_dline[pos_write]/n_rec
        if(record_eta_yn)
            eta_rec[i_rec]  += (eta/sqrt(dt))/n_rec
        end

        pos_now = (pos_now+1)%delay
        if(pos_now==0)
            pos_now=delay
        end
        pos_del = (pos_del+1)%delay
        if(pos_del==0)
            pos_del = delay
        end
    end
    println(r_neg)
    #println()
    return mesoscopic_model_correction_colored_noise_Results_with_delayline(h_rec, s_rec, A_rec, r_rec, r_neg, X_rec, eta_rec)
end


function mesoscopic_model_correction_colored_noise_time_dep_stimulus_OU_noise(h0, s0, X0, F, G, mu_func, param; supress_eta_yn=false, supress_zeta_yn=false, r_replace=0, model3d=1, naive_yn=false)
#=
Same function as above, bu this time with a $\mu = \mu(t)$ as an OU process/ colored Gaussian noise
model3d set to zero to get the model witout the colored finite size correction. I also do this in the naive model to ensure that G(h, 0)=0 without rounding errors
=#
    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = param["N"]
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = param["p"]
    tau     = param["tau"]
    tau_mu  = param["tau_mu"]
    D       = param["D"]
    muext0  = param["muext0"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)

    r_neg   = 0.                                    # counter for how often r was negative
                                                    # aka how often did I need to intervene,
                                                    # I replace the negative r with zero  
    h       = zeros(nSteps+1)
    s       = zeros(nSteps+1)
    A       = zeros(nSteps+1)
    r       = zeros(nSteps+1)
    X       = zeros(nSteps+1)                       # Fluctuating correction to the firing rate 
    slamb   = zeros(nSteps+1)                       # Assume X is an OU process, slamb is sigma_lambda the variance
    eta     = zeros(nSteps+1)                       # Noise term finite-size noise
    zeta    = zeros(nSteps+1)                       # Noise term for the OU for X
    xi_mu   = zeros(nSteps+1) 
    muext   = zeros(nSteps+1)

    if(!supress_eta_yn)
        rng = Random.MersenneTwister()
        param["rng"] = rng
        eta = randn(rng, Float64, (1, nSteps+1))
        xi_mu = randn(rng, Float64, (1, nSteps+1))
    end
    if(!supress_zeta_yn)
        zeta = randn(rng, Float64, (1, nSteps+1))
    end

    X0 = X0*model3d
    if(naive_yn)
        s0 = 0
        model3d = 0
        X0 = 0
    end

    muext[1:delay+1].= muext0
    h[1:delay+1]    .= h0
    s[1:delay+1]    .= s0
    X[1:delay+1]    .= X0
    A[1:delay+1]    .= F(h0, s0, param)
    r[1:delay+1]    .= F(h0, s0, param)
    G0               = G(h0, s0, param)*model3d
    if(abs(G0) < 1e-10)
        G0 = 0
    end
    slamb[1:delay+1].= G0#abs(G(h0, s0, param))*model3d    # apparently there can be numerical errors like -1.124910706472534e-264 \approx 0
    #println(G(h0, s0, param)*model3d)

    for i=1:(nSteps-delay)
        h[i+1+delay]    = h[i+delay] + dt/tau * ( -h[i+delay] + mu0 + mu_func((i+delay)*dt, param) + muext[i+delay] + w * A[i] )
        s[i+1+delay]    = s[i+delay] + dt/tau * ( -2*s[i+delay] +w^2/tau * (1-p)/C * r[i] )
        X[i+1+delay]    = X[i+delay] + dt/tau * ( -X[i+delay] + sqrt(2*tau*slamb[i+delay])*zeta[i+delay]/sqrt(dt) )
        muext[i+1+delay]= muext[i+delay] +dt/tau_mu * ( -muext[i+delay] +sqrt(2*D)*xi_mu[i+delay]/sqrt(dt) )

        if(naive_yn)
            s[i+1+delay] = 0
            X[i+1+delay] = 0
        end

        slamb[i+1+delay]    = G(h[i+1+delay], s[i+1+delay], param)*model3d
        #if(abs(slamb[i+1+delay]) < 1e-10)
        if(slamb[i+1+delay] < 0)
            #println(slamb[i+1+delay])
            slamb[i+1+delay] = 0
        end
        r_inter             = F(h[i+1+delay], s[i+1+delay], param) + 1/sqrt(N)*X[i+1+delay]*model3d
        if(r_inter < 0)
            r_neg  += 1
            r_inter = r_replace         
        end
        r[i+1+delay]        = r_inter
        A[i+1+delay]        = r_inter + sqrt(r_inter/N) * eta[i+1+delay]/sqrt(dt)
    end
    println(r_neg)
    #println()
    return mesoscopic_model_correction_colored_noise_Results(h, s, A, r, r_neg, X)
end



mutable struct Langevin_model_intermediate_Results
    h_rec::Union{ Nothing, Matrix{Float64}, Vector{Vector{Float64}} }
    A_rec::Union{ Nothing, Vector{Float64} }
    r_rec::Union{ Nothing, Vector{Float64} }
end


function Langevin_model_intermediate(Phi, mu_func, param)
#=
intermediate model between annealed model and mesoscopic model
=#
    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = Int(param["N"])
    println(N)
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = param["p"]
    tau     = param["tau"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)+1

    dt_rec  = param["dt_rec"]
    if(dt_rec < dt)
        print("dt_rec is set too small, now reset to dt")
        dt_rec = dt
    end
    n_rec   = round(dt_rec/dt)
    N_rec   = Int(ceil(tmax/dt_rec))
    println(N_rec)    

    h_dline = zeros((N, delay))                     # input potentials

    r_dline = zeros(delay)                          # firing rate
    A_dline = zeros(delay)
    h_rec   = zeros((N, N_rec))
    r_rec   = zeros(N_rec)
    A_rec   = zeros(N_rec)
    pos_now = delay
    pos_del = 1

    rng = Random.MersenneTwister()

    h_ini = param["ini_v"] * ones(N) .+ randn(N) .* sqrt(param["ini_sig"])
    for i=1:delay
        h_dline[:, i] = h_ini
    end
    r_inter = 0
    for j=1:N
        r_inter += Phi(h_dline[j, 1], param)/N
    end
    for i=1:delay
        r_dline[i]  = r_inter
        A_dline[i]  = r_inter
    end
    h_inter = zeros(N)

    for i=1:(nSteps)
        pos_write   = (pos_now+1)%delay
        if(pos_write==0)            # we go through the delayline from 1 to max-value not 0 to max-value -1 
            pos_write=delay
        end
        i_rec       = ceil(Int, i/n_rec)

        for j=1:N
            zeta_j = randn(rng, Float64)
            h_inter[j] = h_dline[j, pos_now] + dt/tau * (-h_dline[j, pos_now] + mu0 + mu_func((i+delay)*dt, param) + w *A_dline[pos_del]
                                                            + w*sqrt((1/(N*p) - 1/N) * r_dline[pos_del])/sqrt(dt)*zeta_j)
        end
        r_inter=0
        for j=1:N
            r_inter += Phi(h_inter[j], param)/N
        end
        eta = randn(rng, Float64)
        A_dline[pos_write]      = r_inter + sqrt(r_inter/N)*eta/sqrt(dt)
        r_dline[pos_write]      = r_inter
        h_dline[:, pos_write]   = h_inter

        h_rec[:, i_rec] += h_inter ./ n_rec
        r_rec[i_rec]    += r_inter ./ n_rec
        A_rec[i_rec]    += A_dline[pos_write] ./ n_rec

        pos_now = (pos_now+1)%delay
        if(pos_now==0)
            pos_now=delay
        end
        pos_del = (pos_del+1)%delay
        if(pos_del==0)
            pos_del = delay
        end 
    end
    return Langevin_model_intermediate_Results(h_rec, A_rec, r_rec)
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

function recursive_multi_layer_correction(nr_layer, param; naive_yn=false)
    h0, s0, r0, fix_found = PSD_get_one_fix_point(param, naive_yn)
    h_r         = zeros(nr_layer+1)
    shh_r       = zeros(nr_layer+1)
    r_r         = zeros(nr_layer+1)
    sig_lamb    = zeros(nr_layer+1)

    h_r[1]      = h0
    shh_r[1]    = s0
    r_r[1]      = r0
    sig_lamb[1] = G_sigm_erf(h0, s0, param)

    w = param["w"]
    tau = param["tau"]
    C = param["C"]
    mu0 = param["mu0"]

    for i =1:(nr_layer)
        ii = i+1                                        # number in vector
        #shh_ralt[ii]    = w^2/C*( sig_lamb[ii-1]/2 +sqrt(r_r[ii-1]*sig_lamb[ii-1]/(2*tau)) +r_r[ii-1]/tau )
        shh_r[ii]       = w^2(2*tau*C)*( r_r[ii-1] + tau*sig_lamb[ii-1] )
        h_r[ii]         = mu0 + w * r_r[ii-1]
        r_r[ii]         = F_sigm_erf(h_r[ii], shh_r[ii], param)
        sig_lamb[ii]    = G_sigm_erf(h_r[ii], shh_r[ii], param)
        
    end
    return h_r, shh_r, r_r, sig_lamb
end

function recursive_multi_layer_correction_alt(nr_layer, param; naive_yn=false)
    h0, s0, r0, fix_found = PSD_get_one_fix_point(param, naive_yn)
    h_r         = zeros(nr_layer+1)
    shh_r       = zeros(nr_layer+1)
    r_r         = zeros(nr_layer+1)
    sig_lamb    = zeros(nr_layer+1)

    h_r[1]      = h0
    shh_r[1]    = s0
    r_r[1]      = r0
    sig_lamb[1] = G_sigm_erf(h0, s0, param)

    w = param["w"]
    tau = param["tau"]
    C = param["C"]
    mu0 = param["mu0"]

    for i =1:(nr_layer)
        ii = i+1                                        # number in vector
        shh_r[ii]       = w^2/C*( sig_lamb[ii-1]/2 +sqrt(r_r[ii-1]*sig_lamb[ii-1]/(2*tau)) +r_r[ii-1]/tau )
        #shh_r[ii]       = w^2(2*tau*C)*( r_r[ii-1]/2 +2*tau*sig_lamb[ii-1] )
        h_r[ii]         = mu0 + w * r_r[ii-1]
        r_r[ii]         = F_sigm_erf(h_r[ii], shh_r[ii], param)
        sig_lamb[ii]    = G_sigm_erf(h_r[ii], shh_r[ii], param)
        
    end
    return h_r, shh_r, r_r, sig_lamb
end

function give_susceptibility_matrix(om, param; naive_yn=false)
#=
    give susceptibility matrix chi
    om: angular frequency omega
=#
    tau = param["tau"]
    w   = param["w"]
    N   = param["N"]
    C   = param["C"]
    d   = param["delay"]
    chi = complex(zeros((3,3)))
    h0, s0, r0, fix_found = PSD_get_one_fix_point(param, naive_yn)
    Fh = delFdelh_sig_erf(h0, s0, param)
    Fs = delFdels_sig_erf(h0, s0, param)

    alpha = w/tau
    beta = w^2 * (1/C-1/N)/tau^2
    gamma = 1/sqrt(N)
    E = exp(-1im*om*d)
    D = (1im * om +1/tau) * (1im * om +2/tau) - (1im * om +2/tau)*Fh*alpha*E-(1im * om +1/tau)*Fs*beta*E

    chi[1, 1] = (1im*om+2/tau-Fs*beta*E)/D
    chi[1, 2] = Fs*alpha*E/D
    chi[1, 3] = gamma*alpha*E/D*(1im*om+2/tau)/(1im*om+1/tau)
    chi[2, 1] = Fh*beta*E/D
    chi[2, 2] = (1im*om+1/tau-Fh*alpha*E)/D
    chi[2, 3] = gamma*beta*E/D
    chi[3, 3] = (1im*om+2/tau)/D-Fs*beta*E/D-Fh*alpha*E/D*(1im*om+2/tau)/(1im*om+1/tau)

    chi_r = 1/tau * (Fh * chi[1,1] + Fs * chi[2, 1])

    return chi[1,1]
end

function give_meso_susceptibility_r(om, param; naive_yn=false)
#=
    give susceptibility matrix chi
    om: angular frequency omega
=#
    tau = param["tau"]
    w   = param["w"]
    N   = param["N"]
    C   = param["C"]
    d   = param["delay"]
    chi = complex(zeros((3,3)))
    h0, s0, r0, fix_found = PSD_get_one_fix_point(param, naive_yn)
    if(naive_yn)
        s0 = 0
    end
    Fh = delFdelh_sig_erf(h0, s0, param)
    #println(Fh)
    Fs = delFdels_sig_erf(h0, s0, param)
    #println(Fs)
    if(naive_yn)
        Fs = 0
    end

    alpha = w/tau
    beta = w^2 * (1/C-1/N)/tau^2
    if(naive_yn)
        beta=0
    end
    gamma = 1/sqrt(N)
    E = exp(-1im*om*d)
    D = (1im * om +1/tau) * (1im * om +2/tau) - (1im * om +2/tau)*Fh*alpha*E-(1im * om +1/tau)*Fs*beta*E

    chi[1, 1] = (1im*om+2/tau-Fs*beta*E)/D
    #chi[1, 2] = Fs*alpha*E/D
    #chi[1, 3] = gamma*alpha*E/D*(1im*om+2/tau)/(1im*om+1/tau)
    chi[2, 1] = Fh*beta*E/D
    #chi[2, 2] = (1im*om+1/tau-Fh*alpha*E)/D
    #chi[2, 3] = gamma*beta*E/D
    #chi[3, 3] = (1im*om+2/tau)/D-Fs*beta*E/D-Fh*alpha*E/D*(1im*om+2/tau)/(1im*om+1/tau)

    chi_r = 1/tau * (Fh * chi[1,1] + Fs * chi[2, 1])

    return chi_r
end

function give_meso_susceptibility_chi_ij(om, param, i_index, j_index; naive_yn=false)
#=
    give susceptibility matrix chi
    om: angular frequency omega
=#
    tau = param["tau"]
    w   = param["w"]
    N   = param["N"]
    C   = param["C"]
    d   = param["delay"]
    chi = complex(zeros((3,3)))
    h0, s0, r0, fix_found = PSD_get_one_fix_point(param, naive_yn)
    if(naive_yn)
        s0 = 0
    end
    Fh = delFdelh_sig_erf(h0, s0, param)
    #println(Fh)
    Fs = delFdels_sig_erf(h0, s0, param)
    #println(Fs)
    if(naive_yn)
        Fs = 0
    end

    alpha = w/tau
    beta = w^2 * (1/C-1/N)/tau^2
    if(naive_yn)
        beta=0
    end
    gamma = 1/sqrt(N)
    E = exp(-1im*om*d)
    D = (1im * om +1/tau) * (1im * om +2/tau) - (1im * om +2/tau)*Fh*alpha*E-(1im * om +1/tau)*Fs*beta*E

    chi[1, 1] = (1im*om+2/tau-Fs*beta*E)/D
    chi[1, 2] = Fs*alpha*E/D
    chi[1, 3] = gamma*alpha*E/D*(1im*om+2/tau)/(1im*om+1/tau)
    chi[2, 1] = Fh*beta*E/D
    chi[2, 2] = (1im*om+1/tau-Fh*alpha*E)/D
    chi[2, 3] = gamma*beta*E/D
    chi[3, 3] = (1im*om+2/tau)/D-Fs*beta*E/D-Fh*alpha*E/D*(1im*om+2/tau)/(1im*om+1/tau)

    chi_r = 1/tau * (Fh * chi[1,1] + Fs * chi[2, 1])
    if((i_index==0) || (j_index==0))
        return chi_r
    end
    return chi[i_index, j_index]
end

function linearized_variance_color_corrected(param, naive_model_yn; turn_off_xi_variable=false)     # has this been checked ever? # eh yes?
    w   = param["w"]
    tau = param["tau"]
    N   = param["N"]
    C   = param["C"]
    p   = param["p"]
    if(naive_yn)
        p = 1
    end
    if(!haskey(param, "mu_sigma_ext"))
        param["mu_sigma_ext"] = 0
    end
    mu_sigma_ext = param["mu_sigma_ext"]

    # fixed points and partial derivatives of F at the fixed points
    hfix, sfix, r0, fix_found = PSD_get_one_fix_point(param, naive_model_yn)
    Fh  = delFdelh_sig_erf(hfix, sfix, param)
    if(!naive_model_yn)
        Fs = delFdels_sig_erf(hfix, sfix, param)
        G0 = G_sigm_erf(hfix, sfix, param)
    else
        Fs = 0
        G0 = 0
    end
    
    # abbreviations
    alpha   = w/tau
    beta    = w^2/(tau^2*C)*(1-p)
    if(naive_yn)
        beta = 0
    end
    gamma   = 1/sqrt(N)

    # Gamma matrix elements
    G11 = Fh*alpha - 1/tau 
    G12 = Fs*alpha
    G13 = gamma*alpha
    G21 = Fh*beta
    G22 = Fs*beta-2/tau
    G23 = gamma*beta
    G33 = -1/tau
    E   = -4*(G11+G22)*(G11*G22-G12*G21)    # Determinant of the subsystem [[G11, G12], [G21, G22]]
    D   = (G11+G33)*(G22+G33)-G12*G21
    #pre = -2*w^2*r0/(tau^2*E*N)
    pre = -2 * (mu_sigma_ext/tau + r0*w^2/tau^2*1/N)/E

    if(turn_off_xi_variable)
        s13 = 0
        s23 = 0
        s33 = 0
    else
        s33 = G0
        #s13 = -G0/D *( G13*(G22+G33) - G23*G12)
        s13 = G0/D * ( G23*G12 - G13*(G22+G33) )
        #s23 = -G0/D *( G13*G21 + G23*(G11+G33) )
        s23 = G0/D * ( G13*G21 - G23*(G11+G33) )
    end

    s11 = pre * ( G12*G21 - G22*(G11+G22) ) -4*s13/E *( G13*(G21*G12 - G22*(G11+G22)) + G23*G12*G22 ) -4*s23/E * ( G23*G12^2 + G13*G12*G22 )
    s12 = pre * ( G21 * G22 ) + 4*s13/E * ( -G21*G22*G13 + G11*G22*G23 ) + 4*s23/E * ( G23*G11*G12 + G13*G11*G22 )
    s22 = pre * ( -G21^2 ) + 4*s13/E * ( G21^2*G13-G11*G21*G23 ) + 4*s23/E * ( (G11*(G11+G22)-G21*G12)*G23 -G11*G21*G13 )

    # now for the firing rate
    sr = Fh^2*s11 + Fs^2*s22 + s33/N + 2*Fh*Fs*s12 + 2*Fh*gamma*s13 + 2*Fs*gamma*s23 
    #println(sr)
    return s11, s12, s22, s13, s23, s33, sr
end

function approximation_of_linearized_variance(param; naive_model_yn=false, approximate_FP=true, approximate_Fh=true, debug_mode=false)
    #=
        program works for w < 0 
    =#

    w   = param["w"]
    tau = param["tau"]
    N   = param["N"]
    C   = param["C"]
    p   = param["p"]     
    mu  = param["mu0"]   
    Phimax = param["Phimax"]
    sig_ext = param["mu_sigma_ext"]

    if(approximate_FP)
        if(mu < 0)
            r0 = 0
        elseif(0 <= mu0 <= abs(w)*Phimax)
            r0 = -mu/w
        else
            r0 = Phimax
        end
    else
        h0, s0, r0, fix_found = PSD_get_one_fix_point(param, naive_model_yn)
        Fh  = delFdelh_sig_erf(h0, s0, param)
    end

    pref1 = -w*r0/(2*tau*N)
    pref2 = sig_ext/(2)

    shh = (pref1+pref2)/Fh
    srr = (pref1+pref2)*Fh

    if(debug_mode)
        return shh, srr, h0, s0, r0, fix_found, pref1, pref2, Fh
    end

    return shh, srr
end

function linearized_variance_color_corrected_Gamma_Matrix_Analysis(param, naive_model_yn; turn_off_xi_variable=false)
    w   = param["w"]
    tau = param["tau"]
    N   = param["N"]
    C   = param["C"]
    p   = param["p"]
    if(naive_model_yn)
        p = 1
    end

    # fixed points and partial derivatives of F at the fixed points
    hfix, sfix, r0, fix_found = PSD_get_one_fix_point(param, naive_model_yn)
    Fh  = delFdelh_sig_erf(hfix, sfix, param)
    if(!naive_model_yn)
        Fs = delFdels_sig_erf(hfix, sfix, param)
        G0 = G_sigm_erf(h0, s0, param)
    else
        Fs = 0
        G0 = 0
    end
    
    # abbreviations
    alpha   = w/tau
    beta    = w^2/(tau^2*C)*(1-p)
    println(p)
    if(naive_model_yn)
        beta = 0
    end
    gamma   = 1/sqrt(N)

    # Gamma matrix elements
    G11 = Fh*alpha - 1/tau 
    G12 = Fs*alpha
    G13 = gamma*alpha
    G21 = Fh*beta
    G22 = Fs*beta-2/tau
    G23 = gamma*beta
    G33 = -1/tau

    return G11, G12, G13, G21, G22, G23, 0, 0, G33, Fh, Fs, alpha, beta, gamma, hfix, sfix, r0 
end

function calculate_chi(x; dt=1, n=100, full_length=true, t_relax=0.)
    # this program has full_length = true, because splitting up chuncks without the delta-disturbance does not make sense
    n_relax = trunc(Int, t_relax/dt)
    y       = x[(n_relax+1):end]
    N_max   = length(y)
    if(full_length)
        n   = N_max
    end
    recast_length   = N_max-(N_max%n)
    nr_cols         = trunc(Int, round(recast_length/n))
    y   = reshape(y[1:recast_length], (n, nr_cols))   # in julia: (a, b) = (rows, col), columnwise sorted!
    yf  = FFTW.fft(y, (1,))
    df  = 1.0/(n*dt)
    f   = df .* (1:(trunc(Int, n/2)-1))
    chi = mean(yf*dt, dims=2)[2:trunc(Int, n/2)]  # why the dt again?
    return f, chi
end

function calculate_chi_h_simple(om, param;naive_model_yn=false)
    w   = param["w"]
    tau = param["tau"]
    N   = param["N"]
    C   = param["C"]
    p   = param["p"]
    if(naive_yn)
        p = 1
    end

    # fixed points and partial derivatives of F at the fixed points
    hfix, sfix, r0, fix_found = PSD_get_one_fix_point(param, naive_model_yn)
    Fh  = delFdelh_sig_erf(hfix, sfix, param)
    if(!naive_model_yn)
        Fs = delFdels_sig_erf(hfix, sfix, param)
        G0 = G_sigm_erf(h0, s0, param)
    else
        Fs = 0
        G0 = 0
    end
    
    # abbreviations
    alpha   = w/tau
    beta    = w^2/(tau^2*C)*(1-p)
    if(naive_model_yn)
        beta = 0
    end
    gamma   = 1/sqrt(N)

    
    
    if(!naive_model_yn)
        D = (1im*om+1/tau-Fh*alpha)*(1im+2/tau-Fs*beta)-alpha*beta*Fh*Fs
        return (1im+2/tau-beta*Fs)/D
    else
        return Fh/(1im*om*tau+1-w*Fh) #susc. of r needs to be changed back
        #return 1/(1im*om*tau+1-w*Fh)
    end
end



function Linear_mesoscopic_model_correction_colored_noise_time_dep_stimulus(h0, s0, X0, F, G, mu_func, param; supress_eta_yn=false, supress_zeta_yn=false, r_replace=0, model3d=1, naive_yn=false, s_func=mu_func_0)
#=
Same function as above, bu this time with a $\mu = \mu(t)$
model3d set to zero to get the model witout the colored finite size correction. I also do this in the naive model to ensure that G(h, 0)=0 without rounding errors
This is the linearized (around thefixed points) version of the model
=#
    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = param["N"]
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = param["p"]
    tau     = param["tau"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)

    r_neg   = 0.                                    # counter for how often r was negative
                                                    # aka how often did I need to intervene,
                                                    # I replace the negative r with zero  
    h       = zeros(nSteps+1)
    s       = zeros(nSteps+1)
    A       = zeros(nSteps+1)
    r       = zeros(nSteps+1)
    X       = zeros(nSteps+1)                       # Fluctuating correction to the firing rate 
    slamb   = zeros(nSteps+1)                       # Assume X is an OU process, slamb is sigma_lambda the variance
    eta     = zeros(nSteps+1)                       # Noise term finite-size noise
    zeta    = zeros(nSteps+1)                       # Noise term for the OU for X
    if(!supress_eta_yn)
        rng = Random.MersenneTwister()
        eta = randn(rng, Float64, (1, nSteps+1))
    end
    if(!supress_zeta_yn)
        zeta = randn(rng, Float64, (1, nSteps+1))
    end

    X0 = X0*model3d
    if(naive_yn)
        s0 = 0
        model3d = 0
        X0 = 0
    end

    hfix, sfix, rfix, fix_found = PSD_get_one_fix_point(param, naive_yn)
    if(naive_yn)
        sfix = 0.
    end
    Fh = delFdelh_sig_erf(hfix, sfix, param)
    Fs = delFdels_sig_erf(hfix, sfix, param)
    if(naive_yn)
        Fs = 0.
    end
    G0 = G(hfix, sfix, param)*model3d


    h[1:delay+1]    .= h0
    s[1:delay+1]    .= s0
    X[1:delay+1]    .= X0
    A[1:delay+1]    .= F(h0, s0, param)
    r[1:delay+1]    .= F(h0, s0, param)
    slamb[1:delay+1].= G(h0, s0, param)*model3d

    for i=1:(nSteps-delay)
        r[i]            = rfix+Fh * (h[i]-hfix) + Fs * (s[i]-sfix) +1/sqrt(N) * X[i]*model3d
        A[i]            = r[i] + sqrt(rfix/N)*eta[i]/sqrt(dt)
        h[i+1+delay]    = h[i+delay] + dt/tau * ( -h[i+delay] + mu0 + mu_func((i)*dt, param) + w * (A[i]) )
        s[i+1+delay]    = s[i+delay] + dt/tau * ( -2*s[i+delay] +w^2/tau * (1-p)/C * r[i] )
        X[i+1+delay]    = X[i+delay] + dt/tau * ( -X[i+delay] + sqrt(2*tau*G0)*zeta[i+delay]/sqrt(dt) )

        if(naive_yn)
            s[i+1+delay] = 0
            X[i+1+delay] = 0
        end
    end
    #println(r_neg)
    #println()
    return mesoscopic_model_correction_colored_noise_Results(h, s, A, r, r_neg, X)
end


function calculate_coeff_DFT(x, dt, om; t_relax=0)
    # how long is one cycle    
    T_cycle = 2*pi/om

    # how many time steps does this make
    nr_steps = trunc(Int, T_cycle/dt)

    # cut off relaxation time
    n_relax = trunc(Int, t_relax/dt)
    y = x[(n_relax+1):end]
    
    # chunk time-series into pieces of length nr_steps
    len_y       = length(y)
    if(len_y < nr_steps)
        println(len_y)
        println(nr_steps)
        error("You have to measure longer enough!")
    end
    len_recast  = len_y-(len_y%nr_steps)
    nr_cols     = trunc(Int, round(len_recast/nr_steps))
    y   = reshape(y[1:len_recast], (nr_steps, nr_cols))

    # FFT and an their coefficients
    yf  = FFTW.fft(y, (1,))  
    coeff = mean(yf, dims=2)/nr_steps
    return coeff
end

function PSD_theory_with_susceptibility_matrix(f, Smumu, param; naive_yn=false)
#=
    calculate PSD for the mesoscopic models, we use the form of the susceptibility matrix
    f:      frequency at whcih we evaluate
    Smumu:  PSD of the external drive $\mu(t)$, at the frequency at which we evaluate
    param:  system parameter
=#
    om  = 2*pi*f    
    tau = param["tau"]
    w   = param["w"]
    N   = param["N"]
    C   = param["C"]
    d   = param["delay"]
    chi = complex(zeros((3,3)))
    S   = complex(zeros((3,3)))
    h0, s0, r0, fix_found = PSD_get_one_fix_point(param, naive_yn)
    if(naive_yn)
        s0 = 0
    end
    Fh = delFdelh_sig_erf(h0, s0, param)
    Fs = delFdels_sig_erf(h0, s0, param)
    if(naive_yn)
        Fs = 0
    end

    alpha = w/tau
    beta = w^2 * (1/C-1/N)/tau^2
    if(naive_yn)
        beta=0
    end
    if(naive_yn)
        G0 = 0
    else
        G0 = G_sigm_erf(h0, s0, param)
    end
    gamma = 1/sqrt(N)
    if(naive_yn)
        gamma = 0
    end
    naive_factor = (1-naive_yn)
    E = exp(-1im*om*d)
    D = (1im * om +1/tau) * (1im * om +2/tau) - (1im * om +2/tau)*Fh*alpha*E-(1im * om +1/tau)*Fs*beta*E

    chi[1, 1] = (1im*om+2/tau-Fs*beta*E)/D
    chi[1, 2] = Fs*alpha*E/D*naive_factor
    chi[1, 3] = gamma*alpha*E/D*(1im*om+2/tau)/(1im*om+1/tau)*naive_factor
    chi[2, 1] = Fh*beta*E/D*naive_factor
    chi[2, 2] = (1im*om+1/tau-Fh*alpha*E)/D*naive_factor
    chi[2, 3] = gamma*beta*E/D*naive_factor
    chi[3, 3] = ( (1im*om+2/tau)/D-Fs*beta*E/D-Fh*alpha*E/D*(1im*om+2/tau)/(1im*om+1/tau) ) * naive_factor

    for i=1:3
        for j=1:3
             S[i, j] = 1/tau^2*(Smumu + w^2*r0/N) * chi[i, 1]*conj(chi[j, 1]) + 2*G0 /(tau) *chi[i, 3]*conj(chi[j, 3])
        end
    end

    if(naive_yn)
        for i =1:3
            for j=1:3
                if(!(i==1) && !(j==1))
                    S[i, j] = 0
                end
            end
        end
    end

    
    Srr     = Fh^2 * real(S[1,1]) + Fs^2 * real(S[2,2]) + 1/N * real(S[3,3]) + 2*Fh*Fs*real(S[2,1]) + 2*Fh/sqrt(N)*real(S[1,3]) + 2*Fs/sqrt(N)*real(S[2,3]) # real(Sii) as a precaution                             
                                                                                             # for rounding errors
    Sreta   = w/tau * sqrt(r0/N) * (Fh*chi[1,1]+Fs*chi[2,1])
    return S, Srr, Sreta, Srr+r0/N+2*sqrt(r0/N)*real(Sreta)    # == SAA
end

function control_PSD_Theory_naive_only(f, Smumu, param)
    om = 2*pi*f
    tau=param["tau"]
    w = param["w"]
    N = param["N"]

    h0, s0, r0, fix_found = PSD_get_one_fix_point(param, true)
    Fh = delFdelh_sig_erf(h0, 0, param)
    E = exp(-1im*om*param["delta"])
    tau_chi_h = 1/(1im*om*tau+1-w*Fh*E)

    Shh     = (Smumu+w^2*r0/N)*tau_chi_h * conj(tau_chi_h)
    Srr     = Fh^2 * Shh
    Setar   = Fh*w*sqrt(r0/N)*tau_chi_h
    SAA     = Srr + r0/N + 2 * real(Setar)*sqrt(r0/N)

    return real(Shh), real(Srr), Setar, real(SAA)
end

mutable struct control_linear_theory_naive_Results
    h::Union{ Nothing, Vector{Float64} }
    A::Union{ Nothing, Vector{Float64} }
    r::Union{ Nothing, Vector{Float64} }
    eta::Union{ Nothing, Vector{Float64} }
end

function control_linear_theory_naive(F, G, mu_func, param; eta_factor=1, div_N=1, record_eta_yn=true, add_noise=true)
#=
naive control
=#
    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = param["N"]
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = 1
    tau     = param["tau"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)+1
    dt_rec  = param["dt_rec"]
    if(dt_rec < dt)
        print("dt_rec is set too small, now reset to dt")
        dt_rec = dt
    end
    n_rec   = round(dt_rec/dt)
    N_rec   = Int(ceil(tmax/dt_rec))

    h_dline     = zeros(delay)
    r_dline     = zeros(delay)
    A_dline     = zeros(delay)
    h_rec   = zeros(N_rec)
    r_rec   = zeros(N_rec)
    A_rec   = zeros(N_rec)
    if(record_eta_yn)
        eta_rec = zeros(N_rec)
    else
        eta_rec = Nothing
    end

    pos_now = delay
    pos_del = 1


    if(!haskey(param, "extra_scaling"))
        param["extra_scaling"] = false
    end

    if(param["extra_scaling"])
        C = param["extra_C"]
        p = 0
        div_N = 0
    end

    r_neg   = 0.                                    # counter for how often r was negative
                                                    # aka how often did I need to intervene,
                                                    # I replace the negative r with zero  
                                                    # Fluctuating correction to the firing rate 

    rng = Random.MersenneTwister()
    param["rng"] = rng

    
    h0, s0, r0, fix_found = PSD_get_one_fix_point(param, true)
    Fh = delFdelh_sig_erf(h0, 0, param)

    for i=1:delay
        h_dline[i] = h0
        r_dline[i] = r0
        A_dline[i] = r0
    end

    h_inter = 0
    r_inter = 0

    for i=1:(nSteps)
        pos_write = (pos_now+1)%delay
        if(pos_write==0)            # we go through the delayline from 1 to max-value not 0 to max-value -1 
            pos_write=delay
        end
        i_rec       = ceil(Int, i/n_rec)

        eta  = randn(rng, Float64)*eta_factor

        h_inter    = h_dline[pos_now] + dt/tau * ( -h_dline[pos_now] + mu0 + mu_func(i*dt, param) + w * A_dline[pos_del] )


        r_inter = r0 + Fh * (h_inter - h0)
        r_dline[pos_write] = r_inter
        if(add_noise)        
            A_dline[pos_write] = r_inter + sqrt(r0/N) * eta/sqrt(dt)*div_N
        else
            A_dline[pos_write] = r_inter + sqrt(r_inter/N) * eta/sqrt(dt)*div_N
        end
        h_dline[pos_write] = h_inter

        h_rec[i_rec]    += h_inter/n_rec
        r_rec[i_rec]    += r_inter/n_rec
        A_rec[i_rec]    += A_dline[pos_write]/n_rec
        if(record_eta_yn)
            eta_rec[i_rec]  += (eta/sqrt(dt))/n_rec
        end

        pos_now = (pos_now+1)%delay
        if(pos_now==0)
            pos_now=delay
        end
        pos_del = (pos_del+1)%delay
        if(pos_del==0)
            pos_del = delay
        end
    end
    return control_linear_theory_naive_Results(h_rec, A_rec, r_rec, eta_rec)
end


function control_linear_naive_model_dummed_down(param)
    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = param["N"]
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = 1
    tau     = param["tau"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)+1
    dt_rec  = param["dt_rec"]
    A       = param["A"]
    B       = param["B"]
    gamma   = param["gamma"]
    if(dt_rec < dt)
        print("dt_rec is set too small, now reset to dt")
        dt_rec = dt
    end
    n_rec   = round(dt_rec/dt)
    N_rec   = Int(ceil(tmax/dt_rec))

    x_dline     = zeros(delay)
    y_dline     = zeros(delay)
    x_rec   = zeros(N_rec)
    y_rec   = zeros(N_rec)
    record_eta_yn = true
    if(record_eta_yn)
        eta_rec = zeros(N_rec)
    else
        eta_rec = Nothing
    end

    rng = Random.MersenneTwister()
    param["rng"] = rng

    pos_now = delay
    pos_del = 1

    for i=1:(nSteps)
        pos_write = (pos_now+1)%delay
        if(pos_write==0)            # we go through the delayline from 1 to max-value not 0 to max-value -1 
            pos_write=delay
        end
        i_rec       = ceil(Int, i/n_rec)

        eta  = randn(rng, Float64)

        x_inter    = x_dline[pos_now] + dt * (y_dline[pos_del] )

        y_dline[pos_write] = gamma + A * x_inter + B * eta/sqrt(dt)
        x_dline[pos_write] = x_inter

        x_rec[i_rec]    += x_inter/n_rec
        y_rec[i_rec]    += y_dline[pos_write]/n_rec
        if(record_eta_yn)
            eta_rec[i_rec]  += (eta/sqrt(dt))/n_rec
        end

        pos_now = (pos_now+1)%delay
        if(pos_now==0)
            pos_now=delay
        end
        pos_del = (pos_del+1)%delay
        if(pos_del==0)
            pos_del = delay
        end
    end
    return x_rec, y_rec, eta_rec
end

function control_linear_naive_model_dummed_down_even_more(param)
    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = param["N"]
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = 1
    tau     = param["tau"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)+1
    dt_rec  = param["dt_rec"]
    A       = param["A"]
    B       = param["B"]
    gamma   = param["gamma"]
    if(dt_rec < dt)
        print("dt_rec is set too small, now reset to dt")
        dt_rec = dt
    end
    n_rec   = round(dt_rec/dt)
    N_rec   = Int(ceil(tmax/dt_rec))

    x_dline     = zeros(delay)
    y_dline     = zeros(delay)
    x_rec   = zeros(N_rec)
    y_rec   = zeros(N_rec)
    record_eta_yn = true
    if(record_eta_yn)
        eta_rec = zeros(N_rec)
    else
        eta_rec = Nothing
    end

    rng = Random.MersenneTwister()
    param["rng"] = rng

    pos_now = delay
    pos_del = 1

    for i=1:(nSteps)
        pos_write = (pos_now+1)%delay
        if(pos_write==0)            # we go through the delayline from 1 to max-value not 0 to max-value -1 
            pos_write=delay
        end
        i_rec       = ceil(Int, i/n_rec)

        eta  = randn(rng, Float64)

        x_inter    = x_dline[pos_now] + dt * ( -gamma * x_dline[pos_now] +  B * eta/sqrt(dt))

        #y_dline[pos_write] = gamma+A * x_inter + B * eta/sqrt(dt)
        x_dline[pos_write] = x_inter

        x_rec[i_rec]    += x_inter/n_rec
        #y_rec[i_rec]    += y_dline[pos_write]/n_rec
        if(record_eta_yn)
            eta_rec[i_rec]  += (eta/sqrt(dt))/n_rec
        end

        pos_now = (pos_now+1)%delay
        if(pos_now==0)
            pos_now=delay
        end
        pos_del = (pos_del+1)%delay
        if(pos_del==0)
            pos_del = delay
        end
    end
    return x_rec#, y_rec, eta_rec
end

function PSD_theory_with_susceptibility_matrix_hsxir(f, Smumu, param; naive_yn=false)
    # same as above, but give out psd of h and s and ...
    om  = 2*pi*f    
    tau = param["tau"]
    w   = param["w"]
    N   = param["N"]
    C   = param["C"]
    d   = param["delay"]
    chi = complex(zeros((3,3)))
    S   = complex(zeros((3,3)))
    h0, s0, r0, fix_found = PSD_get_one_fix_point(param, naive_yn)
    if(naive_yn)
        s0 = 0
    end
    Fh = delFdelh_sig_erf(h0, s0, param)
    Fs = delFdels_sig_erf(h0, s0, param)
    if(naive_yn)
        Fs = 0
    end

    alpha = w/tau
    beta = w^2 * (1/C-1/N)/tau^2
    if(naive_yn)
        beta=0
    end

    if(naive_yn)
        G0 = 0
    else
        G0 = G_sigm_erf(h0, s0, param)
    end
    gamma = 1/sqrt(N)
    if(naive_yn)
        gamma = 0
    end
    E = exp(-1im*om*d)
    D = (1im * om +1/tau) * (1im * om +2/tau) - (1im * om +2/tau)*Fh*alpha*E-(1im * om +1/tau)*Fs*beta*E

    chi[1, 1] = (1im*om+2/tau-Fs*beta*E)/D
    chi[1, 2] = Fs*alpha*E/D
    chi[1, 3] = gamma*alpha*E/D*(1im*om+2/tau)/(1im*om+1/tau)
    chi[2, 1] = Fh*beta*E/D
    chi[2, 2] = (1im*om+1/tau-Fh*alpha*E)/D
    chi[2, 3] = gamma*beta*E/D
    chi[3, 3] = (1im*om+2/tau)/D-Fs*beta*E/D-Fh*alpha*E/D*(1im*om+2/tau)/(1im*om+1/tau)

    for i=1:3
        for j=1:3
            S[i, j] = 1/tau^2*(Smumu*kronecker_delta(i, 1)*kronecker_delta(j, 1) + w^2*r0/N) * chi[i, 1]*conj(chi[j, 1]) + 2*G0 /(tau) *chi[i, 3]*conj(chi[j, 3])
        end
    end
    
    Srr     = Fh^2 * real(S[1,1]) + Fs^2 * real(S[2,2]) + 1/N * real(S[3,3]) + 2*Fh*Fs*real(S[2,1]) + 2*Fh/sqrt(N)*real(S[1,3]) + 2*Fs/sqrt(N)*real(S[2,3]) # real(Sii) as a precaution
      
    if(naive_yn)
        Srr = Fh^2 * real(S[1,1])
    end                                                                                                                                                      # for rounding errors
    Sreta   = w/tau * sqrt(r0/N) * (Fh*chi[1,1]+Fs*chi[2,1])
    return S, Srr, Sreta, Srr+r0/N+2*sqrt(r0/N)*real(Sreta)    # == SAA
end

function calculate_fixed_point_sigmoidal_erf_sig_mu_i(param, hmin, hmax; naive_model = false)
    w   = param["w"]
    mu0 = param["mu0"]
    if((abs(w) < 1e-10))
        return mu0
    end
    Phimax  = param["Phimax"]
    N       = param["N"]
    tau     = param["tau"]
    beta    = param["beta"]
    theta   = param["theta"]
    p       = param["p"]
    A = w*(1-p)/(2*tau*N*p)
    
    if(naive_model)
        f = x -> Phimax*sigm_erf(beta*(x-theta)/sqrt(1+beta^2*param["sig_mu_i"]))-(x-mu0)/w
    else
        f = x-> Phimax*sigm_erf(beta*(x-theta)/(sqrt(abs(1+beta^2*param["sig_mu_i"]+beta^2*A*(x-mu0)))))-(x-mu0)/w
    end
    rts = find_zeros(f, (hmin, hmax), atol=1e-7, rtol=1e-7)
    #rts = roots(f, hmin..hmax)
    return rts
end

function PSD_get_one_fix_point_sig_mu_i(param, naive_model_yn; replacement_h =0, replacement_s =0, replacement_r = 0)
    # replacement_x: if no fixed point s found it is likely the case where h0 \approx \mu_0 otherwise put it to zero, so that I know that there might be a problem
    w           = param["w"]
    mu0         = param["mu0"]
    max_search  = param["max_search_for_fixed_point"]
    p           = param["p"]
    tau         = param["tau"]
    C           = param["C"]
    hfix        = 0
    sfix        = 0
    rfix        = 0
    fix_found   = false

    if(w>0)
        hmin = mu0
        hmax = abs(max_search)
    else
        hmin = -abs(max_search)
        hmax = mu0
    end
    fp = calculate_fixed_point_sigmoidal_erf_sig_mu_i(param, hmin, hmax, naive_model = naive_model_yn)
    println(fp)
    # there can be 1 or 3 fixed points: take the fixed point with the smallest firing rate
    len_fp = length(fp)
    if(len_fp == 0)
        return replacement_h, replacement_s, replacement_r, fix_found
    end
    fix_found = true
    for i=1:len_fp
        hi = fp[i]
        if(naive_model_yn)
            si = 0
        else
            si = w * (1-p)/(2*tau*C)*(hi - mu0)
        end
        ri = F_sigm_erf(hi, si+sig_mu_i, param)
        #if((i==1)|| (ri <= r0))
         if((i==1) || (ri <= rfix))
            hfix = hi
            sfix = si
            rfix = ri
        end
    end
    return hfix, sfix, rfix, fix_found
end

function sig_erf_inv(x)
    return sqrt(2)*erfinv(2*x-1)
end

function Q_exp_sig_erf_inv(x)
    if((x<= 0) || (x>= 1))
        return 0.
    else
        return exp(-0.5*(sig_erf_inv(x))^2)
    end
end

function Fh_hat_in_scaling(muhat, betahat, tau, N, p)
    # Fh_hat = Fh * sqrt(-w*2*pi/(beta*Phimax))
    if(muhat < 0)
        return 0
    end
    return 1/sqrt(1+(1-p)*muhat*betahat^2/(2*tau*N*p*Phimax))*Q_exp_sig_erf_inv(muhat)
end


function Fh_approxed_FP(param)
    mu = param["mu0"]
    w  = param["w"]
    Phimax = param["Phimax"] 
    p       = param["p"]
    beta    = param["beta"]
    N = param["N"]
    tau = param["tau"]
    if(mu <= 0)
        return 0
    end
    if(mu < (abs(w)*Phimax))
        return Phimax/sqrt(2*pi)*exp(-0.5*(sig_erf_inv(mu/(abs(w)*Phimax)))^2)/sqrt(1/(param["beta"])^2 + abs(w)*mu*(1-p)/(2*param["tau"]*param["N"]*p))
    else
        return Phimax/sqrt(2*pi)*exp(-0.5*(w*Phimax +mu0)^2/(1/beta^2+Phimax*w^2*(1-p)/(2*tau*p*N)))/sqrt(1/beta^2+Phimax*w^2*(1-p)/(2*tau*p*N))
    end
end

function FP_approxed(param)
    mu0     = param["mu0"]
    w       = param["w"]
    Phimax  = param["Phimax"]
    N       = param["N"]
    p       = param["p"]    
    if(mu0 < 0)
        r0 = 0
    elseif(0< mu0 < abs(w)*Phimax)
        r0 = mu0/abs(w)
    else
        r0 = Phimax
    end
    s0 = w^2 * (1-p)/(2*param["tau"]*N*p) * r0
    if(mu0 > abs(w)*Phimax)
        h0 = mu0 + w*Phimax
    else
        h0 = sig_erf_inv(r0/Phimax)*sqrt(1/(param["beta"])^2 + s0)
    end
    return h0, s0, r0
end


function heaviside(x; a=0)
    if(x<0)
        return 0
    else
        return 1
    end
end


function Fh_in_param(param)
    mu=param["mu0"]
    rm=param["Phimax"]
    p = param["p"]
    w =param["w"]
    return rm*Q_exp_sig_erf_inv(-mu/(w*rm))/sqrt(2*pi) / sqrt(1/beta^2 - w*mu*(1-p)/(2*param["tau"]*p*param["N"]))
end

function Fs_in_param(param)
    mu=param["mu0"]
    rm=param["Phimax"]
    p = param["p"]
    w =param["w"]
    beta= param["beta"]
    tau = param["tau"]
    N = param["N"]    
   #return -rm*beta^2*sig_erf_inv(-mu/(w*Phimax))*Q_exp_sig_erf_inv(-mu/(w*Phimax))/sqrt(2*pi)/sqrt(1/beta^2-w*mu*(1-p)/(2*param["tau"]*p*param["N"]))
    return -rm/(2*sqrt(2*pi)) * Q_exp_sig_erf_inv(-mu0/(w*rm))*sig_erf_inv(-mu0/(w*rm))/(1/beta^2-w*mu0*(1-p)/(2*tau*N*p))
end


mutable struct naive_with_shot_noise_Results
    h::Union{ Nothing, Vector{Float64} }
    r::Union{ Nothing, Vector{Float64} }
end

function simulator_mean_connected_network_with_shot_noise(h0, param, mu_func)
#=
    use for the simulation of the breakdown of the simple mean-field networks
=#
    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = Int(param["N"])
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = param["p"]
    tau     = param["tau"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)+1

    dt_rec  = param["dt_rec"]
    if(dt_rec < dt)
        print("dt_rec is set too small, now reset to dt")
        dt_rec = dt
    end
    n_rec   = round(dt_rec/dt)                  # number of steps for averaging
    N_rec   = Int(ceil(tmax/dt_rec))            # number of time steps in simulation

    h_dline = zeros(delay)                      # input potentials
    r_dline = zeros(delay)                      # firing rate
    h_rec   = zeros(N_rec)
    r_rec   = zeros(N_rec)
    pos_now = delay
    pos_del = 1

    rng = Random.MersenneTwister()
    param["rng"] =rng

    for j=1:delay
        h_dline[j] = h0
    end
    r_inter = Phi_sigm_erf(h_dline[1], param)
    r_dline[1]  = r_inter
    h_inter = 0

    for j=1:(nSteps)
        pos_write   = (pos_now+1)%delay
        if(pos_write==0)            # we go through the delayline from 1 to max-value not 0 to max-value -1 
            pos_write=delay
        end
        j_rec       = ceil(Int, j/n_rec)

        r_inter             = Phi_sigm_erf(h_dline[pos_del], param)
        #nr_of_spikes        = rand(rng, Poisson(r_inter*dt))
        nr_of_spikes        = rand(rng, Poisson(N*r_inter*dt))
        h_inter             = h_dline[pos_now] + dt/tau *(-h_dline[pos_now] + mu0 + mu_func((j+delay)*dt, param)) + w/(N*tau) * nr_of_spikes                 
        r_dline[pos_write]  = r_inter
        h_dline[pos_write]  = h_inter

        h_rec[j_rec] += h_inter / n_rec
        r_rec[j_rec]    += r_inter / n_rec

        pos_now = (pos_now+1)%delay
        if(pos_now==0)
            pos_now=delay
        end
        pos_del = (pos_del+1)%delay
        if(pos_del==0)
            pos_del = delay
        end 
    end
    return naive_with_shot_noise_Results(h_rec, r_rec)

end


mutable struct Micro_Simulation_with_reset_Results
    ns::Union{ Nothing, Vector{Int} }                                       #Int8???
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

function Micro_Simulation_with_reset(T, dt, Ne, Ni, Nrecord, dt_rec, Phi, param_Phi, param_lnp, seed_quenched;fixed_in_degree=true, ini_v=0, ini_sig=0, record_mean_h_i=false, record_var_h_i=false,
                                    record_skew_h_i=false, mu_func=mu_func_0)
#=
Simulate microscopic model of spiking neurons, in general an IE-network. This is a mirror version of micro_simulation,
but optimized for large (sparse) network: only save connections not the entire matrix
=#

    if(seed_quenched == 0)
        rng =   Random.MersenneTwister()
        param_lnp["rng"] = rng
    else
        rng     = Random.MersenneTwister(seed_quenched)
        rng2    = Random.MersenneTwister()
        param_lnp["rng"] = rng2
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

    delta_v = param_lnp["delta_v"]
    rough_prob_estimate_yn = param_lnp["rough_prob_estimate_yn"]

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
    
    jie = param_lnp["con_ie"] / Ke   # mV ??? not mVs?
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

    # set up vector of vectos:
    con_vec = [[] for _ in 1:(Ni+Ne)]

    if(fixed_in_degree)
        # loop over all neurons: from which neurons do the receive spikes
        for n in (1:Ncells)
            
            # excitatory neurons: from which neurons do they receive spikes?
            list_innervating_neuron_n = sample(rng, 1:Ne, Ke, replace=false)

            # reverse that: make list of which neurons get a spike from neuron j:
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end

            # same as above, just inhibitory neurons
            list_innervating_neuron_n = sample(rng, (Ne+1):(Ni+Ne), Ki, replace=false)
        
            # same as above, just inhibitory neurons
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end
        end
    else
        # loop over all neurons: from which neurons do the receive spikes
        for n in (1:Ncells)

            # excitatory neurons: from which neurons do they receive spikes?
            list_innervating_neuron_n = [i for i in 1:Ne if rand(rng) < p_e]

            # reverse that: make list of which neurons get a spike from neuron j:
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end

            # same as above, just inhibitory neurons
            list_innervating_neuron_n = [i for i in (Ne+1):(Ne+Ni) if rand(rng) < p_i]
        
            # same as above, just inhibitory neurons
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end
        end
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
    spikes_per_time_stepI = 0
    spikes_per_time_stepE = 0

    #quantities to be recorded
    Nsteps = round(Int,T/dt)
    Nsteps_rec = round(Int,T/dt_rec)
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
            #@printf("\33[2K\r%d%% poprateE=%g poprateI=%g, abgelaufene Zeit=%g, rate=%g, Mu%g",round(Int,100*ti/Nsteps), poprateE, poprateI, ti*dt, sum(ns)/(Ni*ti*dt), mu0)
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
                
	        v[ci] += dtau[ci]*(mu[ci]+mu_ext-v[ci] + synInput)

            rate = Phi(v[ci], param_Phi)
            if (ci<=Ne)
                poprateE += rate #r_e
            else
                poprateI += rate #r_i
            end
            
            if(rough_prob_estimate_yn)
                rough_est_prob = rate * dt
            else
                rough_est_prob = 1.0 - exp(-rate * dt)
            end

	        #if (rand() < rate * dt)  #spike occurred
	        if (rand() < rough_est_prob)  #spike occurred
		        ns[ci] = ns[ci]+1
                #first test place the reset here
                if(ci<=Ne)
                    spikes_per_time_stepE += 1
                    v[ci]-= delta_v *taue
                else
                    spikes_per_time_stepI += 1
                    v[ci]-=delta_v * taui
                end
                if (ci<=Nrecord)
                    spikes[ci, i_rec] = 1
                end
              
                if(ci <= Ne) # from excitatory ...
                    for cj in con_vec[ci]
                        if(cj <= Ne) # ... to excitatory
                            forwardInputsE[cj] += jee
                        else # ... to inhibitory
                            forwardInputsE[cj] += jie
                        end
                    end
                else  # from inhibitory ...
                    for cj in con_vec[ci]
                        if(cj <= Ne) # ... to excitatory
                            forwardInputsI[cj] += jei
                        else # ... to inhibitory
                            forwardInputsI[cj] += jii
                        end
                    end
                end
	        end #end if(spike occurred)
            
            if (ci <= Nrecord)
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
    return Micro_Simulation_with_reset_Results(ns, v_record, spikes, rpopE_record, rpopI_record, actpopE_record, actpopI_record, mean_h_i,var_h_i, skew_h_i)
end

function exp_escape(h, param)
    return param["c"]*exp((h-param["h_thresh"])/param["Delta_h"])
end

function F_exp_escape(h, s, param)
    return exp_escape(h+s/(2*param["Delta_h"]), param)
end

function G_exp_escape(h, s, param)
    x = s/param["Delta_h"]
    return exp_escape(2*h+2*x, param)*(1-exp(-x))
end

mutable struct meso_model_with_reset_Results
    h::Union{ Nothing, Vector{Float64} }
    s::Union{ Nothing, Vector{Float64} }
    A::Union{ Nothing, Vector{Float64} }
    r::Union{ Nothing, Vector{Float64} }
    r_neg::Union{ Nothing, Float64 }
    X::Union{ Nothing, Vector{Float64} }
    eta::Union{ Nothing, Vector{Float64} }
end


function meso_model_with_reset(h0, s0, X0, F, G, mu_func, param; eta_factor=1, zeta_factor=1, r_replace=0,
                                                                        model3d=1, naive_yn=false, div_N=1, record_eta_yn=true)
#=
Same function as above, but this time I have a "delay-line", now I do not need to save the entire time series. THIS PROGRAM IS NOT FINISHED!!!
=#
    tmax    = param["tmax"]
    dt      = param["dt"]
    delta   = param["delta"]
    N       = param["N"]
    mu0     = param["mu0"]
    w       = param["w"]
    C       = param["C"]
    p       = param["p"]
    tau     = param["tau"]
    Dh      = param["h_thresh"]-param["h_reset"]
    nSteps  = trunc(Int, tmax/dt)
    delay   = trunc(Int, delta/dt)+1
    dt_rec  = param["dt_rec"]
    if(dt_rec < dt)
        print("dt_rec is set too small, now reset to dt")
        dt_rec = dt
    end
    n_rec   = round(dt_rec/dt)
    N_rec   = Int(ceil(tmax/dt_rec))

    h_dline     = zeros(delay)
    s_dline     = zeros(delay)
    X_dline     = zeros(delay)
    r_dline     = zeros(delay)
    A_dline     = zeros(delay)
    slamb_dline = zeros(delay)
    h_rec   = zeros(N_rec)
    s_rec   = zeros(N_rec)
    X_rec   = zeros(N_rec)
    r_rec   = zeros(N_rec)
    A_rec   = zeros(N_rec)
    if(record_eta_yn)
        eta_rec = zeros(N_rec)
    else
        eta_rec = Nothing
    end

    pos_now = delay
    pos_del = 1


    if(!haskey(param, "extra_scaling"))
        param["extra_scaling"] = false
    end

    if(param["extra_scaling"])
        C = param["extra_C"]
        p = 0
        div_N = 0
    end

    r_neg   = 0.                                    # counter for how often r was negative
                                                    # aka how often did I need to intervene,
                                                    # I replace the negative r with zero  
                                                    # Fluctuating correction to the firing rate 

    rng = Random.MersenneTwister()
    param["rng"] = rng

    turn_off_if_naive = 1-naive_yn

    X0 = X0*model3d
    if(naive_yn)
        s0 = 0
        model3d = 0
        X0 = 0
    end

    r0 = F(h0, s0, param)
    G0 = G(h0, s0, param)*model3d
    if(abs(G0) < 1e-10)
        G0 = 0
    end

    for i=1:delay
        h_dline[i] = h0
        s_dline[i] = s0
        X_dline[i] = X0
        r_dline[i] = r0
        A_dline[i] = r0
        slamb_dline[i] = G0
    end

    h_inter = 0
    s_inter = 0
    X_inter = 0
    r_inter = 0
    slamb_inter = 0

    for i=1:(nSteps)
        pos_write = (pos_now+1)%delay
        if(pos_write==0)            # we go through the delayline from 1 to max-value not 0 to max-value -1 
            pos_write=delay
        end
        i_rec       = ceil(Int, i/n_rec)

        zeta = randn(rng, Float64)*eta_factor
        eta  = randn(rng, Float64)*zeta_factor

        h_inter    = h_dline[pos_now] + dt/tau * ( -h_dline[pos_now] + mu0 + mu_func(i*dt, param) + w * A_dline[pos_del] - tau * Dh * r_dline[pos_now]*turn_off_if_naive)
        s_inter    = s_dline[pos_now] + dt/tau * ( -2*s_dline[pos_now] +w^2/tau * (1-p)/C * r_dline[pos_del] )
        X_inter    = X_dline[pos_now] + dt/tau * ( -X_dline[pos_now] + sqrt(2*tau*slamb_dline[pos_now])*zeta/sqrt(dt) )

        if(naive_yn)
            s_inter = 0
            X_inter = 0
        end

        slamb_inter    = G(h_inter, s_inter, param)*model3d
        if(slamb_inter < 0)
            slamb_inter = 0
        end
        r_inter = F(h_inter, s_inter, param) + 1/sqrt(N)*X_inter*model3d*div_N
        if(r_inter < 0)
            r_neg  += 1
            r_inter = r_replace         
        end
        r_dline[pos_write] = r_inter
        A_dline[pos_write] = r_inter + sqrt(r_inter/N) * eta/sqrt(dt)*div_N
        h_dline[pos_write] = h_inter
        s_dline[pos_write] = s_inter
        X_dline[pos_write] = X_inter

        slamb_dline[pos_write] = slamb_inter

        h_rec[i_rec]    += h_inter/n_rec
        s_rec[i_rec]    += s_inter/n_rec
        X_rec[i_rec]    += X_inter/n_rec
        r_rec[i_rec]    += r_inter/n_rec
        A_rec[i_rec]    += A_dline[pos_write]/n_rec
        if(record_eta_yn)
            eta_rec[i_rec]  += (eta/sqrt(dt))/n_rec
        end

        pos_now = (pos_now+1)%delay
        if(pos_now==0)
            pos_now=delay
        end
        pos_del = (pos_del+1)%delay
        if(pos_del==0)
            pos_del = delay
        end
    end
    println(r_neg)
    #println()
    return meso_model_with_reset_Results(h_rec, s_rec, A_rec, r_rec, r_neg, X_rec, eta_rec)
end

function exp_hazard(h, param)
#=
    exponential hazard rate, assume theta =0
=#
    return param["alpha"]*exp(param["beta"]*h)
end


mutable struct IF_simple_model_Results
    ns::Union{ Nothing, Vector{Int} }                                       #Int8???
    v_record::Union{ Nothing, Matrix{Float32}, Vector{Vector{Float32}} }
    spikes::Union{ Nothing, Matrix{Int8} }
    #rpopE_record::Union{ Nothing, Vector{Float32} }
    #rpopI_record::Union{ Nothing, Vector{Float32} }
    rpopE_record::Union{ Nothing, Vector{Float64} }
    rpopI_record::Union{ Nothing, Vector{Float64} }
    actpopE_record::Union{ Nothing, Vector{Float32} }
    actpopI_record::Union{ Nothing, Vector{Float32} }
    mean_h_i::Union{ Nothing, Vector{Float32} }
    var_h_i::Union{ Nothing, Vector{Float32} }
    skew_h_i::Union{ Nothing, Vector{Float32} }
    mean_u_i::Union{ Nothing, Vector{Float32} }
end

function IF_simple_model(T, dt, Ne, Ni, Nrecord, dt_rec, Phi, param_Phi, param_lnp, seed_quenched;fixed_in_degree=true, ini_v=0, ini_sig=0, record_mean_h_i=false, record_var_h_i=false,
                                    record_skew_h_i=false, mu_func=mu_func_0)
#=
    Simulate microscopic model of spiking neurons, in general an IE-network. This is a mirror version of micro_simulation
=#

    if(seed_quenched == 0)
        rng =   Random.MersenneTwister()
        param_lnp["rng"] = rng
    else
        rng     = Random.MersenneTwister(seed_quenched)
        rng2    = Random.MersenneTwister()
        param_lnp["rng"] = rng2
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

    v_R = param_lnp["v_R"]              # reset voltage
    #v_R = 0
    #println(v_R)
    rough_prob_estimate_yn = param_lnp["rough_prob_estimate_yn"]

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
    
    jie = param_lnp["con_ie"] / Ke   # mV ??? not mVs?
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
    println(Nbin)
    
    #membrane time constant
    tau = zeros(Ncells)
    dtau = zeros(Ncells)
    
    tau[1:Ne] .= taue
    tau[(1+Ne):Ncells] .= taui
    dtau[1:Ne] .= dt/taue
    dtau[(1+Ne):Ncells] .= dt/taui

    tau_ref = param_lnp["tau_ref"]

    t_ref_vec = ones(Ncells)*tau_ref


    # set up vector of vectos:
    con_vec = [[] for _ in 1:(Ni+Ne)]

    if(fixed_in_degree)
        # loop over all neurons: from which neurons do the receive spikes
        for n in (1:Ncells)
            
            # excitatory neurons: from which neurons do they receive spikes?
            list_innervating_neuron_n = sample(rng, 1:Ne, Ke, replace=false)

            # reverse that: make list of which neurons get a spike from neuron j:
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end

            # same as above, just inhibitory neurons
            list_innervating_neuron_n = sample(rng, (Ne+1):(Ni+Ne), Ki, replace=false)
        
            # same as above, just inhibitory neurons
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end
        end
    else
        # loop over all neurons: from which neurons do the receive spikes
        for n in (1:Ncells)

            # excitatory neurons: from which neurons do they receive spikes?
            list_innervating_neuron_n = [i for i in 1:Ne if rand(rng) < p_e]

            # reverse that: make list of which neurons get a spike from neuron j:
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end

            # same as above, just inhibitory neurons
            list_innervating_neuron_n = [i for i in (Ne+1):(Ne+Ni) if rand(rng) < p_i]
        
            # same as above, just inhibitory neurons
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end
        end
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
    u = ini_v * ones(Ncells) .+ randn(Ncells) .* sqrt(ini_sig)
    spikes_per_time_stepI = 0
    spikes_per_time_stepE = 0

    #quantities to be recorded
    Nsteps = round(Int,T/dt)
    Nsteps_rec = round(Int,T/dt_rec)
    v_record = zeros(Float32,(Nrecord, Nsteps_rec))         # voltage traces to be recorded
    if(record_mean_h_i)
        mean_h_i = zeros(Float32, Nsteps_rec)               # mean of distribution of voltage distribution for each recorded time step
        mean_u_i = zeros(Float64, Nsteps_rec)
    else
        mean_h_i = nothing
        mean_u_i = nothing
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
    #rpopE_record    = zeros(Float32,Nsteps_rec)                # E population rate
    #rpopI_record    = zeros(Float32,Nsteps_rec)                # I population rate
    rpopE_record    = zeros(Float64,Nsteps_rec)                # E population rate
    rpopI_record    = zeros(Float64,Nsteps_rec)                # I population rate
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
	    end
	    t = dt*ti
        t_ref_vec = t_ref_vec .+ dt

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
                
	        v[ci] += dtau[ci]*(mu[ci]+mu_ext-v[ci] + synInput)
            u[ci] += dtau[ci]*(mu[ci]+mu_ext-u[ci] + synInput)

            rate = Phi(v[ci], param_Phi)
            if (ci<=Ne)
                poprateE += rate #r_e
            else
                poprateI += rate #r_i
            end
            
            if(rough_prob_estimate_yn)
                rough_est_prob = rate * dt
            else
                rough_est_prob = 1.0 - exp(-rate * dt)
            end

	        #if (rand() < rate * dt)  #spike occurred
	        #if (rand() < rough_est_prob)  #spike occurred
            if ((rand() < rough_est_prob) && (t_ref_vec[ci] > tau_ref))  #spike occurred
		        ns[ci] = ns[ci]+1
                if(ci<=Ne)
                    spikes_per_time_stepE += 1
                    v[ci] = v_R
                    t_ref_vec[ci] = 0
                else
                    spikes_per_time_stepI += 1
                    v[ci] = v_R
                    t_ref_vec[ci] = 0
                    #println(v[ci])
                end
                if (ci<=Nrecord)
                    spikes[ci, i_rec] = 1
                end
              
                if(ci <= Ne) # from excitatory ...
                    for cj in con_vec[ci]
                        if(cj <= Ne) # ... to excitatory
                            forwardInputsE[cj] += jee
                        else # ... to inhibitory
                            forwardInputsE[cj] += jie
                        end
                    end
                else  # from inhibitory ...
                    for cj in con_vec[ci]
                        if(cj <= Ne) # ... to excitatory
                            forwardInputsI[cj] += jei
                        else # ... to inhibitory
                            forwardInputsI[cj] += jii
                        end
                    end
                end
	        end #end if(spike occurred)
            
            if (ci <= Nrecord)
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
            mean_u_i[i_rec] += mean(u)
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
        mean_u_i /=Nbin
    end
    if(record_var_h_i)
        var_h_i /=Nbin  # Mean variance of h_i over the Nbins
    end
    if(record_skew_h_i)
        skew_h_i /=Nbin  # Mean variance of h_i over the Nbins
    end
    println()
    @printf("\r")
    return IF_simple_model_Results(ns, v_record, spikes, rpopE_record, rpopI_record, actpopE_record, actpopI_record, mean_h_i,var_h_i, skew_h_i, mean_u_i)
end


mutable struct SRM_simple_model_Results
    ns::Union{ Nothing, Vector{Int} }                                       #Int8???
    v_record::Union{ Nothing, Matrix{Float32}, Vector{Vector{Float32}} }
    spikes::Union{ Nothing, Matrix{Int8} }
    #rpopE_record::Union{ Nothing, Vector{Float32} }
    #rpopI_record::Union{ Nothing, Vector{Float32} }
    rpopE_record::Union{ Nothing, Vector{Float64} }
    rpopI_record::Union{ Nothing, Vector{Float64} }
    actpopE_record::Union{ Nothing, Vector{Float32} }
    actpopI_record::Union{ Nothing, Vector{Float32} }
    mean_h_i::Union{ Nothing, Vector{Float32} }
    var_h_i::Union{ Nothing, Vector{Float32} }
    skew_h_i::Union{ Nothing, Vector{Float32} }
    mean_u_i::Union{ Nothing, Vector{Float32} }
end

function SRM_simple_model(T, dt, Ne, Ni, Nrecord, dt_rec, Phi, param_Phi, param_lnp, seed_quenched;fixed_in_degree=true, ini_v=0, ini_sig=0, record_mean_h_i=false, record_var_h_i=false,
                                    record_skew_h_i=false, mu_func=mu_func_0)
#=
    Simulate microscopic model of spiking neurons, in general an IE-network. This is a mirror version of micro_simulation
=#

    if(seed_quenched == 0)
        rng =   Random.MersenneTwister()
        param_lnp["rng"] = rng
    else
        rng     = Random.MersenneTwister(seed_quenched)
        rng2    = Random.MersenneTwister()
        param_lnp["rng"] = rng2
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

    dv_reset = param_lnp["dv_reset"]              # reset voltage
    #v_R = 0
    #println(v_R)
    rough_prob_estimate_yn = param_lnp["rough_prob_estimate_yn"]

    tau_ref = param_lnp["tau_ref"]

    t_ref_vec = ones(Ncells)*tau_ref

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
    
    jie = param_lnp["con_ie"] / Ke   # mV ??? not mVs?
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
    println(Nbin)
    
    #membrane time constant
    tau = zeros(Ncells)
    dtau = zeros(Ncells)
    
    tau[1:Ne] .= taue
    tau[(1+Ne):Ncells] .= taui
    dtau[1:Ne] .= dt/taue
    dtau[(1+Ne):Ncells] .= dt/taui

    # set up vector of vectos:
    con_vec = [[] for _ in 1:(Ni+Ne)]

    if(fixed_in_degree)
        # loop over all neurons: from which neurons do the receive spikes
        for n in (1:Ncells)
            
            # excitatory neurons: from which neurons do they receive spikes?
            list_innervating_neuron_n = sample(rng, 1:Ne, Ke, replace=false)

            # reverse that: make list of which neurons get a spike from neuron j:
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end

            # same as above, just inhibitory neurons
            list_innervating_neuron_n = sample(rng, (Ne+1):(Ni+Ne), Ki, replace=false)
        
            # same as above, just inhibitory neurons
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end
        end
    else
        # loop over all neurons: from which neurons do the receive spikes
        for n in (1:Ncells)

            # excitatory neurons: from which neurons do they receive spikes?
            list_innervating_neuron_n = [i for i in 1:Ne if rand(rng) < p_e]

            # reverse that: make list of which neurons get a spike from neuron j:
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end

            # same as above, just inhibitory neurons
            list_innervating_neuron_n = [i for i in (Ne+1):(Ne+Ni) if rand(rng) < p_i]
        
            # same as above, just inhibitory neurons
            for j in list_innervating_neuron_n
                push!(con_vec[j], n)
            end
        end
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
    u = ini_v * ones(Ncells) .+ randn(Ncells) .* sqrt(ini_sig)
    spikes_per_time_stepI = 0
    spikes_per_time_stepE = 0

    #quantities to be recorded
    Nsteps = round(Int,T/dt)
    Nsteps_rec = round(Int,T/dt_rec)
    v_record = zeros(Float32,(Nrecord, Nsteps_rec))         # voltage traces to be recorded
    if(record_mean_h_i)
        mean_h_i = zeros(Float32, Nsteps_rec)               # mean of distribution of voltage distribution for each recorded time step
        mean_u_i = zeros(Float64, Nsteps_rec)
    else
        mean_h_i = nothing
        mean_u_i = nothing
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
    #rpopE_record    = zeros(Float32,Nsteps_rec)                # E population rate
    #rpopI_record    = zeros(Float32,Nsteps_rec)                # I population rate
    rpopE_record    = zeros(Float64,Nsteps_rec)                # E population rate
    rpopI_record    = zeros(Float64,Nsteps_rec)                # I population rate
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
	    end
	    t = dt*ti
        t_ref_vec = t_ref_vec .+ dt
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
                
	        v[ci] += dtau[ci]*(mu[ci]+mu_ext-v[ci] + synInput)
            u[ci] += dtau[ci]*(mu[ci]+mu_ext-u[ci] + synInput)

            rate = Phi(v[ci], param_Phi)
            if (ci<=Ne)
                poprateE += rate #r_e
            else
                poprateI += rate #r_i
            end
            
            if(rough_prob_estimate_yn)
                rough_est_prob = rate * dt
            else
                rough_est_prob = 1.0 - exp(-rate * dt)
            end

	        #if (rand() < rate * dt)  #spike occurred
	        #if (rand() < rough_est_prob)  #spike occurred
            if ((rand() < rough_est_prob) && (t_ref_vec[ci] > tau_ref))  #spike occurred
		        ns[ci] = ns[ci]+1
                if(ci<=Ne)
                    spikes_per_time_stepE += 1
                    v[ci] = v[ci] - dv_reset
                    t_ref_vec[ci] = 0
                else
                    spikes_per_time_stepI += 1
                    v[ci] = v[ci] - dv_reset
                    t_ref_vec[ci] = 0
                    #println(v[ci])
                end
                if (ci<=Nrecord)
                    spikes[ci, i_rec] = 1
                end
              
                if(ci <= Ne) # from excitatory ...
                    for cj in con_vec[ci]
                        if(cj <= Ne) # ... to excitatory
                            forwardInputsE[cj] += jee
                        else # ... to inhibitory
                            forwardInputsE[cj] += jie
                        end
                    end
                else  # from inhibitory ...
                    for cj in con_vec[ci]
                        if(cj <= Ne) # ... to excitatory
                            forwardInputsI[cj] += jei
                        else # ... to inhibitory
                            forwardInputsI[cj] += jii
                        end
                    end
                end
	        end #end if(spike occurred)
            
            if (ci <= Nrecord)
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
            mean_u_i[i_rec] += mean(u)
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
        mean_u_i /=Nbin
    end
    if(record_var_h_i)
        var_h_i /=Nbin  # Mean variance of h_i over the Nbins
    end
    if(record_skew_h_i)
        skew_h_i /=Nbin  # Mean variance of h_i over the Nbins
    end
    println()
    @printf("\r")
    return SRM_simple_model_Results(ns, v_record, spikes, rpopE_record, rpopI_record, actpopE_record, actpopI_record, mean_h_i,var_h_i, skew_h_i, mean_u_i)
end

function exp_hazard_integrated(h, s, param)
#=
    integrated version of the exponential hazard rate, we assume theta = 0
    h           : mean input potential [mV]
    s           : sigma^2 of input potential [mV^2]
=#
    beta = param["beta"]
    return param["alpha"]*exp(beta*h+beta^2/2*s)

end

function circular_index(x, L)
#=
    cyclical position of indices 1=>1, L=>L, L+1=>1, -1 => L-1, 0=> L and so on
=#
    return mod(x-1, L)+1            # some modulus magic
end

mutable struct integral_equation_for_closed_reset_RESULTS
    A::Union{ Nothing, Vector{Float64} }
    SA::Union{ Nothing, Vector{Float64} }
end

function integral_equation_for_closed_reset(param)
#=
    here comes some comments
=#    

    # call parameters
    tmax    = param["tmax"]                                     # max simulation time [s]
    T_mem   = param["T_mem"]                                    # max time where trajectories remember last spike time [s]
    dt_sim  = param["dt_sim"]                                   # time step for simulation [s]
    dt_rec  = param["dt_rec"]                                   # time step for recording [s]
    mu_0    = param["mu_0"]                                     # a constant part of the external drive [mV]
    tau     = param["tau"]                                      # membrane time constant [s]
    w       = param["w"]                                        # connection strength, rescaled [mVs]
    alpha   = param["alpha"]                                    # phi = alpha exp(beta v), [s^(-1)]
    beta    = param["beta"]                                     # phi = alpha exp(beta v), [mV^(-1)]
    C       = param["C"]                                        # fixed in-degree number of synapses [1]
    p       = param["p"]                                        # connection probability [1]
    n_delay = round(Int, param["delay"]/dt_sim)                 # steps of transmission delay (delay in [s])
    record_int_SA_yn = param["record_int_SA_yn"]                # test: record int SA = 1

    n_tsim = round(Int, tmax/dt_sim)    
    n_T = round(Int, T_mem/dt_sim)
    if(n_delay > n_T)
        println("error, delay is assumed to be shorter than T_mem")
    end
    n_rec   = round(dt_rec/dt_sim)              # number of steps for averaging
    N_rec   = Int(ceil(tmax/dt_rec))            # number of recorded time steps in simulation
    
    # initialize recorded values
    A_rec   = zeros(N_rec)
    if(record_int_SA_yn)
        int_SA_inter    = 0
        int_SA_rec      = zeros(N_rec)
    else
        int_SA_rec      = nothing
    end

    # initialize calculation of lambda_free
    h_free  = param["h_free_0"]
    h_free_inter = h_free
    s_free  = param["s_free_0"]
    s_free_inter = s_free
    lam_free = exp_hazard_integrated(h_free, s_free, param)

    # initialize memory arrays and intermediate
    h_mem       = zeros(n_T)
    h_inter     = zeros(n_T)
    s_mem       = zeros(n_T)
    s_inter     = zeros(n_T)
    Surv_mem    = zeros(n_T)
    Surv_inter  = zeros(n_T)
    lamb_mem    = zeros(n_T)

    lamb_mem[1]     = alpha
    Surv_mem[1]     = 1
    Surv_inter[1]   = 1
    A_mem       = zeros(n_T)
    A_inter     = 0
    i_rec       = 0

    # initialize pos for memeory array
    pos_now     = n_T                       # current position in mem arrays
    pos_del     = n_T - n_delay             # position of delay
    pos_write   = 1                         # where to write in the new values

    for i = 1:n_tsim
	    if mod(i,n_tsim/100) == 1  #print percent complete
            @printf("\33[2K\r%d%%",round(Int,100*i/n_tsim))
	    end
        pos_write   = circular_index(pos_now+1, n_T)
        i_rec       = ceil(Int, i/n_rec)

        # update trajectories of last spikes
        for j = 2:n_T
            lamb_mem[j]     = exp_hazard_integrated(h_mem[j-1], s_mem[j-1], param)
            #println(s_mem[j])
            h_inter[j]      = h_mem[j-1] + dt_sim * ( ( mu_0 + w*A_mem[pos_del] - h_mem[j-1])/tau - s_mem[j-1] * beta *lamb_mem[j])  
            #s_inter[j]      = s_mem[j-1] + dt_sim/tau * (-2*s_mem[j-1] + 2*w^2*(1-p)/(tau*C)*A_mem[pos_del] - beta^2*(s_mem[j-1])^2*lamb_mem[j])
            #experimental line
            s_inter[j]      = s_mem[j-1] + dt_sim/tau * (-2*s_mem[j-1] + w^2*(1-p)/(tau*C)*A_mem[pos_del]) - dt_sim * beta^2*(s_mem[j-1])^2*lamb_mem[j]
            Surv_inter[j]   = Surv_mem[j-1] - dt_sim * lamb_mem[j] * Surv_mem[j-1]
        end

        # update lambda_free 
        h_free_inter = h_free + dt_sim/tau * ( mu_0 + w*A_mem[pos_del] - h_free) - dt_sim * s_free *beta * lam_free
        s_free_inter = s_free - dt_sim/tau *(2*s_free + w^2*(1-p)/(tau*C) * A_mem[pos_del]) - dt_sim * beta^2 * s_free^2*lam_free
        lam_free = exp_hazard_integrated(h_free_inter, s_free_inter, param)
        #println(lam_free>=0)

        # Summation for the integral
        A_inter = 0
        if(record_int_SA_yn)
            int_SA_inter  =0
        end
        for j =1:n_T
            # calculate pos in A array
            pos_j = circular_index(j-n_delay, n_T)
            
            # summation
            A_inter += (lamb_mem[j] - lam_free) * Surv_inter[j] * A_mem[pos_j]
            if(record_int_SA_yn)
                int_SA_inter += Surv_inter[j] * A_mem[pos_j]
            end
        end
        if(record_int_SA_yn)
            int_SA_inter *= dt_sim
        end
        A_inter *=  dt_sim
        A_inter += lam_free
    
        # write for next step
        A_mem[pos_write]    = A_inter
        h_free              = h_free_inter
        s_free              = s_free_inter


        for j=2:n_T
            h_mem[j]    = h_inter[j]
            s_mem[j]    = s_inter[j]
            Surv_mem[j] = Surv_inter[j]
        end

        # write down for output
        A_rec[i_rec] += A_inter/n_rec
        if(record_int_SA_yn)
            int_SA_rec[i_rec] += int_SA_inter/n_rec
        end

        # update current position for next time step
        pos_now = circular_index(pos_now+1, n_T)
        pos_del = circular_index(pos_del+1, n_T)
    end
    return integral_equation_for_closed_reset_RESULTS(A_rec, int_SA_rec)
end

mutable struct integral_equation_with_reset_no_refractory_RESULTS
    A::Union{ Nothing, Vector{Float64} }
    m::Union{ Nothing, Vector{Float64} }
end


function integral_equation_with_reset_no_refractory(mu_func, param)
#=
    same as above, but hopefully better algorithm introduceage vector and I move this density around explicitly
=#

    # call parameters
    tmax    = param["tmax"]                                     # max simulation time [s]
    T_mem   = param["T_mem"]                                    # max time where trajectories remember last spike time [s]
    dt_sim  = param["dt_sim"]                                   # time step for simulation [s]
    dt_rec  = param["dt_rec"]                                   # time step for recording [s]
    mu_0    = param["mu_0"]                                     # a constant part of the external drive [mV]
    tau     = param["tau"]                                      # membrane time constant [s]
    w       = param["w"]                                        # connection strength, rescaled [mVs]
    alpha   = param["alpha"]                                    # phi = alpha exp(beta v), [s^(-1)]
    beta    = param["beta"]                                     # phi = alpha exp(beta v), [mV^(-1)]
    C       = param["C"]                                        # fixed in-degree number of synapses [1]
    p       = param["p"]                                        # connection probability [1]
    n_delay = round(Int, param["delay"]/dt_sim)                 # steps of transmission delay (delay in [s])
    A_bin   = param["bin_weight"]
    B_bin   = 1 - A_bin    


    n_tsim = round(Int, tmax/dt_sim)            # number of simulation time steps
    n_T = round(Int, T_mem/dt_sim)              # number of steps where history is recorded
    if(n_delay > n_T)
        println("error, delay is assumed to be shorter than T_mem")
    end
    n_rec   = round(dt_rec/dt_sim)              # number of steps for averaging
    N_rec   = Int(ceil(tmax/dt_rec))            # number of recorded time steps in simulation
    
    # initialize recorded values
    A_rec   = zeros(N_rec)
    m_rec   = zeros(N_rec)
    m_inter = 0
    
    # initialize calculation of lambda_free
    h_free      = param["h_free_0"]
    s_free      = param["s_free_0"]
    m_free      = 0
    lamb_free   = exp_hazard_integrated(h_free, s_free, param)

    # initialize memory arrays and intermediate
    h_T       = zeros(n_T)
    s_T       = zeros(n_T)
    m_T       = zeros(n_T)                          # normalized fraction m(t, t')  
    lamb_T    = zeros(n_T)
    lamb_inter= 0

    m_T[2]  = 1                                   # start all neurons as having just recently fired
    A_T     = zeros(n_T)
    N_T     = zeros(n_T)                          # n = A*dt  
    i_rec   = 0  

    # initialize pos for memeory array
    pos_now     = n_T                       # current position in mem arrays
    pos_del     = n_T - n_delay             # position of delay
    pos_write   = 1                         # where to write in the new values

    dm_free_inter = 0


    for i_t = 1:n_tsim
	    if mod(i_t,n_tsim/100) == 1  #print percent complete
            @printf("\33[2K\r%d%%",round(Int,100*i_t/n_tsim))
	    end
        pos_write   = circular_index(pos_now+1, n_T)
        i_rec       = ceil(Int, i_t/n_rec)
        N_T[pos_write] = 0
        m_free += m_T[n_T]                  # farthest state gets shifted into the free state
        for j in reverse(1:(n_T-1))
            h_T[j+1]    = h_T[j] + dt_sim/tau * (mu_0 + mu_func(i_t*dt_sim, param) + w * A_T[pos_del] - h_T[j]) 
                            - dt_sim * s_T[j] * beta * lamb_T[j]*1
            s_T[j+1]    = s_T[j] + dt_sim/tau * (-2*s_T[j] + w^2*(1-p)/(tau*C)*A_T[pos_del])
                            - dt_sim * (s_T[j])^2 * beta^2 * lamb_T[j] * 1
            lamb_inter  = exp_hazard_integrated(h_T[j+1], s_T[j+1], param)
            P_fire      = (1 - exp(-dt_sim*(A_bin*lamb_T[j]+B_bin*lamb_inter)))*1
            lamb_T[j+1] = lamb_inter
            dm_T = P_fire * m_T[j]
            m_T[j+1] = m_T[j] - dm_T
            N_T[pos_write] += dm_T
        end

        # evolve free solution
        h_free = h_free + dt_sim/tau * (mu_0 + mu_func(i_t*dt_sim, param) + w * A_T[pos_del] - h_free) 
                            - dt_sim * s_free * beta * lamb_free * 1
        s_free = s_free + dt_sim/tau * (-2*s_free + w^2*(1-p)/(tau*C)*A_T[pos_del])
                            - dt_sim * (s_free)^2 * beta^2 * lamb_free * 1
        lamb_inter = exp_hazard_integrated(h_free, s_free, param)
        P_fire = (1 - exp(-dt_sim*(A_bin*lamb_free+B_bin*lamb_inter)))
        lamb_free = lamb_inter
        dm_free = P_fire * m_free
        m_free -= dm_free
            #A_T[pos_write] += dm_free/dt_sim
        N_T[pos_write] += dm_free
            #m_T[1] = A_T[pos_write] * dt_sim    # for new time step: all neurons that spiked are in the first bin
        m_T[1] = N_T[pos_write]
            #m_free+=dm_free_inter
        A_T[pos_write] = N_T[pos_write]/dt_sim        


        # write down for output
        A_rec[i_rec] += A_T[pos_write]/n_rec

        m_inter = 0
        for j_T = 1:n_T
            m_inter+=m_T[j_T]
        end
        m_inter += m_free
        #m_free = (1-m_inter) # test
        #m_inter=1
        m_rec[i_rec] += m_inter/n_rec

        # test: print out some of the data
        if(i_t==(n_tsim-1))
            println(s_T)
            println(s_free)
            println("mfree: ", m_free)
            println("mtotal = 1 vs ", m_rec[i_rec])
            println("m_T[n_T]: ", m_T[n_T])
            println("dm_free: ", dm_free)
        end

        # update current position for next time step
        pos_now = circular_index(pos_now+1, n_T)
        pos_del = circular_index(pos_del+1, n_T)
    end
    return integral_equation_with_reset_no_refractory_RESULTS(A_rec, m_rec)
end

function hybrid_model_annealed_micro(param)
#=
    some comments
=#
   
    # call parameters
    tmax    = param["tmax"]                                     # max simulation time [s]
    dt_sim  = param["dt_sim"]                                   # time step for simulation [s]
    dt_rec  = param["dt_rec"]                                   # time step for recording [s]
    mu_0    = param["mu_0"]                                     # a constant part of the external drive [mV]
    tau     = param["tau"]                                      # membrane time constant [s]
    w       = param["w"]                                        # connection strength, rescaled [mVs]
    alpha   = param["alpha"]                                    # phi = alpha exp(beta v), [s^(-1)]
    beta    = param["beta"]                                     # phi = alpha exp(beta v), [mV^(-1)]
    C       = param["C"]                                        # fixed in-degree number of synapses [1]
    p       = param["p"]                                        # connection probability [1]
    n_delay = round(Int, param["delay"]/dt_sim)                 # steps of transmission delay (delay in [s])
    N       = param["N"]                                        #

    n_tsim  = round(Int, tmax/dt_sim)            # number of simulation time steps
    n_rec   = round(dt_rec/dt_sim)              # number of steps for averaging
    N_rec   = Int(ceil(tmax/dt_rec))            # number of recorded time steps in simulation
    
    # initialize recorded values
    A_rec   = zeros(N_rec)
    r_rec   = zeros(N_rec)
    A_inter = 0

    r_dline     = zeros(delay)
    A_dline     = zeros(delay)

    h_old       = zeros(N)
    h_new       = zeros(N)

    Sqrt_term   = 0

    pos_now = delay
    pos_del = 1

    rng = Random.MersenneTwister()
    #param["rng"] = rng


    for i_t = 1
	    if mod(i_t,n_tsim/100) == 1  #print percent complete
            @printf("\33[2K\r%d%%",round(Int,100*i_t/n_tsim))
	    end
        pos_write = circular_index(pos_now+1, delay)
        

        sqrt_term   = w*sqrt((1-p)/C*r_dline[pos_del])
        mu_term     = mu_0 + mu_func(i_t*dt_sim, param)
        wA_term     = w*A_dline[pos_delay]
        for i_N = 1:N
            h_new[i_N] = h_old[i_N] + dt_sim/tau * ( -h_old[i_N] + mu_term + wA_term + sqrt_term * randn(rng) )


            lamb_new[i_N] = exp_hazard(h_new[i_N])
            P_fire = 1 - exp(-dt_sim*(A_bin*lamba_new[i_N]+B_bin*lambda_old[i_N]))            

            # reset mechanism
            if(rand() < P_fire)
                h_new[i_N] = 0
            end
        end

        lambda_old  = lambda_new

    end

end


function translate_skew_normal_parameter(h, s, g)
    println("h: ", h)
    println("s: ", s)
    println("g: ", g)
    g_hat = g * 2/(4-pi)
    delta_hat = cbrt(g_hat)/sqrt(1+(cbrt(g_hat))^2)
    println("d: ", delta_hat)
    a   = delta_hat * sqrt(pi/2)/sqrt(1-delta_hat^2*pi/2)
    om  = sqrt(s)/sqrt(1-delta_hat^2) 
    z   = h - om*delta_hat
    return z, om, a
end

function Phi_erf(x)
    return 0.5*(1+erf(x/sqrt(2)))
end

function lambda_skew_normal(alpha, beta, z, om, a)
    return 2*alpha*exp(beta*z+beta^2*om^2/2)*Phi_erf(a*beta*om/sqrt(1+a^2))
end

function debugmessenger(cond, message)
    if(cond)
        println(message)
    end
end

function pdf_skewnormal(x; z=0, w=0.001, a=0)
    return 2/w * 1/sqrt(2*pi)*exp(-(x-z)^2/(2*w^2))*Phi_erf(a*(x-z)/w)
end


mutable struct integral_equation_for_skew_normal_RESULTS
    A::Union{ Nothing, Vector{Float64} }
    m::Union{ Nothing, Vector{Float64} }
end

function integral_equation_for_skew_normal(mu_func, param)
#=
    Integral equations as above but not assuming that the 

=#

    # call parameters
    tmax    = param["tmax"]                                     # max simulation time [s]
    T_mem   = param["T_mem"]                                    # max time where trajectories remember last spike time [s]
    dt_sim  = param["dt_sim"]                                   # time step for simulation [s]
    dt_rec  = param["dt_rec"]                                   # time step for recording [s]
    mu_0    = param["mu_0"]                                     # a constant part of the external drive [mV]
    tau     = param["tau"]                                      # membrane time constant [s]
    w       = param["w"]                                        # connection strength, rescaled [mVs]
    alpha   = param["alpha"]                                    # phi = alpha exp(beta v), [s^(-1)]
    beta    = param["beta"]                                     # phi = alpha exp(beta v), [mV^(-1)]
    C       = param["C"]                                        # fixed in-degree number of synapses [1]
    p       = param["p"]                                        # connection probability [1]
    n_delay = round(Int, param["delay"]/dt_sim)                 # steps of transmission delay (delay in [s])
    A_bin   = param["bin_weight"]
    B_bin   = 1 - A_bin    


    n_tsim = round(Int, tmax/dt_sim)            # number of simulation time steps
    n_T = round(Int, T_mem/dt_sim)              # number of steps where history is recorded
    if(n_delay > n_T)
        println("error, delay is assumed to be shorter than T_mem")
    end
    n_rec   = round(dt_rec/dt_sim)              # number of steps for averaging
    N_rec   = Int(ceil(tmax/dt_rec))            # number of recorded time steps in simulation
    
    # initialize recorded values
    A_rec   = zeros(N_rec)
    m_rec   = zeros(N_rec)
    m_inter = 0

    # initialize calculation of lambda_free
    h_free      = param["h_free_0"]
    s_free      = param["s_free_0"]
    g_free      = param["g_free_0"]
    z, om, a = translate_skew_normal_parameter(h_free, s_free, g_free)
    m_free      = 0
    lamb_free   = lambda_skew_normal(alpha, beta, z, om, a)

    # initialize memory arrays and intermediate
    h_T       = zeros(n_T)
    s_T       = zeros(n_T)
    g_T       = zeros(n_T)
    m_T       = zeros(n_T)                          # normalized fraction m(t, t')  
    lamb_T    = zeros(n_T)
    lamb_inter= 0
    inter1    = 0
    inter2    = 0
    inter3    = 0
    inter4    = 0 
    s         = 0
    h         = 0
    g         = 0
    dh        = 0
    ds        = 0
    dg        = 0
    a         = 0
    om        = 0
    z         = 0
    a_free    = 0
    om_free   = 0
    z_free    = 0

    m_T[2]  = 1                                   # start all neurons as having just recently fired
    A_T     = zeros(n_T)
    N_T     = zeros(n_T)                          # n = A*dt  
    i_rec   = 0  

    # initialize pos for memeory array
    pos_now     = n_T                       # current position in mem arrays
    pos_del     = n_T - n_delay             # position of delay
    pos_write   = 1                         # where to write in the new values

    dm_free_inter = 0

    for i_t = 1:n_tsim
	    if mod(i_t,n_tsim/100) == 1  #print percent complete
            @printf("\33[2K\r%d%%",round(Int,100*i_t/n_tsim))
	    end

        mu_func_now = mu_func(i_t*dt_sim, param)

        pos_write   = circular_index(pos_now+1, n_T)
        i_rec       = ceil(Int, i_t/n_rec)
        N_T[pos_write] = 0
        m_free += m_T[n_T]                  # farthest state gets shifted into the free state
        debugmessenger(poormansdebug, "enter j loop over integral past")
        for j in reverse(1:(n_T-1))
            # start with some abbreviations to get rid of annoying arguments
            h = h_T[j]
            s = s_T[j]            
            g = g_T[j]            

            # translate current variables to parameters for skew-normal:
            z, om, a = translate_skew_normal_parameter(h, s, g)            

            # start with some abbreviations to save computation:
            exp_0 = exp(beta * z + beta^2*om^2/2)
            exp_1 = a*om/sqrt(2*pi)/sqrt(1+a^2) * exp(-a^2*om^2*beta^2/(2*(1+a^2)))
            Phi_0 = Phi_erf(a*beta*om/sqrt(1+a^2))
            mu_now= mu_0 + mu_func_now
            D_now = w^2*(1-p)/(2*tau^2*C)*A_T[pos_del]
            if(s>0)
                ss= sqrt(s)
            else
                ss = 0
            end
            power3 = (ss^3*g+h^3+3*h*s)


            # calculate the increments for the dynamics (some of them will be reused):
            dh = lamb_T[j] * h - 2 * alpha * exp_0 * ((z+beta*om^2)*Phi_0+exp_1) + 1/tau * (mu_now + w * A_T[pos_del] - h)
            ds = lamb_T[j] * (s + h^2) + 2/tau * (mu_now - s - h^2) + 2*D_now - 2 * h * dh - 2*alpha*exp_0*(Phi_0*(om^2+(z+beta*om^2)^2)
                                                                                        + exp_1*(2*(z+beta*om^2) -a^2*om^2*beta/(1+a^2)))
            if(s>0)
                inter1 = lamb_T[j] *power3
                inter2 = 3*(z+beta*om^2)*om^2+(z+beta*om^2)^2
                inter3 = 3*om^2+3*(z+beta*om^2)^2-3*a^2*om^2*beta/(1+a^2)*(z+beta*om^2)+a^4*om^4/(1+a^2)*beta^2-a^2*om^2/(1+a^2)
                inter4 = 3/tau * mu_now * (s+h^2) + 3/tau * power3 + 6*D_now*h-3*h^2*dh-3*s*dh-3*h*ds 
                dg = 1/ss^3 * (inter1 - 2 * alpha * exp_0 * (Phi_0 * inter2 + exp_1 * inter3) + inter4) - 3/2*g/s*ds
            else
                dg = 0
            end

            h_T[j+1] = h_T[j] + dt_sim * dh
            s_T[j+1] = s_T[j] + dt_sim * ds
            g_T[j+1] = g_T[j] + dt_sim * dg
            # EXPERIMENTAL!!!:
            """
            if(g_T[j+1]>1)
                g_T[j+1] = 1
            elseif(g_T[j+1]<-1)
                g_T[j+1] = -1
            end
            """
            z, om, a = translate_skew_normal_parameter(h_T[j+1], s_T[j+1], g_T[j+1])
  
            lamb_inter  = lambda_skew_normal(alpha, beta, z, om, a)

            P_fire      = (1 - exp(-dt_sim*(A_bin*lamb_T[j]+B_bin*lamb_inter)))
            lamb_T[j+1] = lamb_inter
            dm_T = P_fire * m_T[j]
            m_T[j+1] = m_T[j] - dm_T
            N_T[pos_write] += dm_T
        end
            debugmessenger(poormansdebug, "exit j loop over integral past")

        # evolve free solution
        h = h_free
        s = s_free
        g = g_free

        z, om, a = translate_skew_normal_parameter(h, s, g)

        exp_0 = exp(beta * z + beta^2*om^2/2)
        exp_1 = a*om/sqrt(2*pi)/sqrt(1+a^2) * exp(-a^2*om^2*beta^2/(2*(1+a^2)))
        Phi_0 = Phi_erf(a*beta*om/sqrt(1+a^2))
        mu_now= mu_0 + mu_func_now
        D_now = w^2*(1-p)/(2*tau^2*C)*A_T[pos_del]
        if(s>0)
            ss= sqrt(s)
        else
            ss = 0
        end
        power3 = (ss^3*g+h^3+3*h*s)

        dh = lamb_free * h - 2 * exp_0 * ((z+beta*om^2)*Phi_0+exp_1) + 1/tau * (mu_now + w * A_T[pos_del] - h)
        ds = lamb_free * (s + h^2) + 2/tau * (mu_now - s - h^2) + 2*D_now - 2 * h * dh - 2*exp_0*(Phi_0*(om^2+(z+beta*om^2)^2)
                                                                                        + exp_1*(2*(z+beta*om^2) -a^2*om^2*beta/(1+a^2)))
        if(s>0)
            inter1 = lamb_free *power3
            inter2 = 3*(z+beta*om^2)*om^2+(z+beta*om^2)^2
            inter3 = 3*om^2+3*(z+beta*om^2)^2-3*a^2*om^2*beta/(1+a^2)*(z+beta*om^2)+a^4*om^4/(1+a^2)*beta^2-a^2*om^2/(1+a^2)
            inter4 = 3/tau * mu_now * (s+h^2) + 3/tau * power3 + 6*D_now*h-3*h^2*dh-3*s*dh-3*h*ds 
            dg = 1/ss^3 * (inter1 - 2 * exp_0 * (Phi_0 * inter2 + exp_1 * inter3) + inter4)-3/2/s*g*ds
        else
            dg = 0
        end

        h_free = h_free + dt_sim * dh
        s_free = s_free + dt_sim * ds
        g_free = g_free + dt_sim * dg
        # EXPERIMENTAL:
        """
        if(g_free >1)
            g_free = 1
        elseif(g_free<-1)
            g_free = -1
        end
        """        
        z, om, a = translate_skew_normal_parameter(h_free, s_free, g_free)

        lamb_inter  = lambda_skew_normal(alpha, beta, z, om, a)
        P_fire = (1 - exp(-dt_sim*(A_bin*lamb_free+B_bin*lamb_inter)))
        lamb_free = lamb_inter
        dm_free = P_fire * m_free
        m_free -= dm_free
            #A_T[pos_write] += dm_free/dt_sim
        N_T[pos_write] += dm_free
            #m_T[1] = A_T[pos_write] * dt_sim    # for new time step: all neurons that spiked are in the first bin
        m_T[1] = N_T[pos_write]
            #m_free+=dm_free_inter
        A_T[pos_write] = N_T[pos_write]/dt_sim        


        # write down for output
        A_rec[i_rec] += A_T[pos_write]/n_rec

        m_inter = 0
        for j_T = 1:n_T
            m_inter+=m_T[j_T]
        end
        m_inter += m_free
        #m_free = (1-m_inter) # test
        #m_inter=1
        m_rec[i_rec] += m_inter/n_rec

        # test: print out some of the data
        if(i_t==(n_tsim-1))
            println(s_T)
            println(s_free)
            println("mfree: ", m_free)
            println("mtotal = 1 vs ", m_rec[i_rec])
            println("m_T[n_T]: ", m_T[n_T])
            println("dm_free: ", dm_free)
        end

        # update current position for next time step
        pos_now = circular_index(pos_now+1, n_T)
        pos_del = circular_index(pos_del+1, n_T)
    end
    return integral_equation_for_skew_normal_RESULTS(A_rec, m_rec)
end



function moments_of_v_integral_skew_normal(z, om, a)
    #=
    express the moments <v>, <v^2> and <v^3> in terms of z, om, a
    =#
    d_hat = a/sqrt(1+a^2) * sqrt(2/pi)
    v1 = z+d_hat*om
    v2 = om^2 * (1-d_hat^2) + v1^2
    v3 = (4-pi)/2*d_hat^3*om^3+3*v2*v1-2*v1^3

    return v1, v2, v3
end


mutable struct integral_equation_for_skew_normal_zwa_RESULTS
    A::Union{ Nothing, Vector{Float64} }
    m::Union{ Nothing, Vector{Float64} }
end

function integral_equation_for_skew_normal_zwa(mu_func, param)
#=
    Integral equations as above but not now we take z, omega (om) and a as our main parameter. I hope that this fixes
    a problem that the system of equations wonder into complex values

=#

    # call parameters
    tmax    = param["tmax"]                                     # max simulation time [s]
    T_mem   = param["T_mem"]                                    # max time where trajectories remember last spike time [s]
    dt_sim  = param["dt_sim"]                                   # time step for simulation [s]
    dt_rec  = param["dt_rec"]                                   # time step for recording [s]
    mu_0    = param["mu_0"]                                     # a constant part of the external drive [mV]
    tau     = param["tau"]                                      # membrane time constant [s]
    w       = param["w"]                                        # connection strength, rescaled [mVs]
    alpha   = param["alpha"]                                    # phi = alpha exp(beta v), [s^(-1)]
    beta    = param["beta"]                                     # phi = alpha exp(beta v), [mV^(-1)]
    C       = param["C"]                                        # fixed in-degree number of synapses [1]
    p       = param["p"]                                        # connection probability [1]
    n_delay = round(Int, param["delay"]/dt_sim)                 # steps of transmission delay (delay in [s])
    A_bin   = param["bin_weight"]
    B_bin   = 1 - A_bin    


    n_tsim = round(Int, tmax/dt_sim)            # number of simulation time steps
    n_T = round(Int, T_mem/dt_sim)              # number of steps where history is recorded
    if(n_delay > n_T)
        println("error, delay is assumed to be shorter than T_mem")
    end
    n_rec   = round(dt_rec/dt_sim)              # number of steps for averaging
    N_rec   = Int(ceil(tmax/dt_rec))            # number of recorded time steps in simulation
    
    # initialize recorded values
    A_rec   = zeros(N_rec)
    m_rec   = zeros(N_rec)
    m_inter = 0

    # initialize calculation of lambda_free
    z_free      = param["z_free_0"]
    om_free     = param["om_free_0"]
    a_free      = param["a_free_0"]
    m_free      = 0
    lamb_free   = lambda_skew_normal(alpha, beta, z_free, om_free, a_free)

    # initialize memory arrays and intermediate
    z_T       = ones(n_T)*param["z_free_0"]
    om_T      = ones(n_T)*param["om_free_0"]
    a_T       = ones(n_T)*param["a_free_0"]
    m_T       = zeros(n_T)                          # normalized fraction m(t, t')  
    lamb_T    = zeros(n_T)
    lamb_inter= 0
    da        = 0
    dz        = 0
    dom        = 0
    dd        = 0
    a         = 0
    om        = 0
    z         = 0
    f1        = 0
    f2        = 0
    f3        = 0

    m_T[2]  = 1                                   # start all neurons as having just recently fired
    a_T[(n_T-1)] = param["a_free_0"]
    om_T[(n_T-1)] = param["om_free_0"]
    z_T[(n_T-1)] = param["z_free_0"]

    
    A_T     = zeros(n_T)
    N_T     = zeros(n_T)                          # n = A*dt  
    i_rec   = 0  

    # initialize pos for memeory array
    pos_now     = n_T                       # current position in mem arrays
    pos_del     = n_T - n_delay             # position of delay
    pos_write   = 1                         # where to write in the new values

    dm_free_inter = 0

    for i_t = 1:n_tsim
	    if mod(i_t,n_tsim/100) == 1  #print percent complete
            @printf("\33[2K\r%d%%",round(Int,100*i_t/n_tsim))
	    end

        mu_hat  = mu_0 + mu_func(i_t*dt_sim, param)
        D_hat   = w^2*(1-p)/(2*tau^2*C)*A_T[pos_del]

        pos_write   = circular_index(pos_now+1, n_T)
        i_rec       = ceil(Int, i_t/n_rec)
        N_T[pos_write] = 0
        m_free += m_T[n_T]                  # farthest state gets shifted into the free state
        debugmessenger(poormansdebug, "enter j loop over integral past")
        for j in reverse(1:(n_T-1))
            # start with some abbreviations to get rid of annoying arguments
            a    = a_T[j]
            om   = om_T[j]           
            z    = z_T[j]
            lamb = lamb_T[j]
            v1, v2, v3  = moments_of_v_integral_skew_normal(z, om, a)
            d_hat = a/sqrt(1+a^2)*sqrt(2/pi)              

            # start with some abbreviations to save computation:
            exp_0 = exp(beta * z + beta^2*om^2/2)
            exp_1 = a*om/sqrt(2*pi)/sqrt(1+a^2) * exp(-a^2*om^2*beta^2/(2*(1+a^2)))
            Phi_0 = Phi_erf(a*beta*om/sqrt(1+a^2))
            zbo2  = (z+beta*om^2)

            # calculate f1, f2, and f3 as abbreviation for the increments:
            f1 = lamb*v1 + 1/tau * (mu_hat - v1) - 2*alpha*exp_0*(zbo2*Phi_0+exp_1)
            f2 = lamb*v2 + 2/tau * (mu_hat - v2) + 2*D_hat -2*alpha*exp_0*( Phi_0*(om^2+zbo2^2) + exp_1*(2*zbo2-a^2*om^2*beta/(1+a^2)) ) -2*v1*f1
            f3 = lamb*v3 + 3/tau * (mu_hat - v3) + 6*D_hat*v1 - 2*alpha*exp_0*( Phi_0*(3*zbo2*om^2+zbo2^3)+exp_1*(3*om^2+3*zbo2^2-3*a^2*om*2*beta/(1+a^2)*zbo2+a^4*om^4*beta^2/(1+a^2)^2-a^2*om^2/(1+a^2)) )  
                    -3*v1*(f2+2*v1*f1)+(6*v1^2-3*v2)*f1

            # calculate increments z, om, a:
            dd  = f3/(3*om^3*d_hat^2)*2/(4-pi)*(1-d_hat^2)-d_hat^2/(2*om^2)*f2
            da  = (1+a^2)^(3/2)*sqrt(pi/2)*dd
            dom = f2/(2*om) + f3/(3*om^2*d_hat)*2/(4-pi)
            dz  = f1- f3/(3*om^2*d_hat^2)*2/(4-pi)

            # update z, om, a:
            if(i_t == 1)
                println(om)
                println(j)
                println(n_T-1)
            end

            z_T[j+1] = z_T[j] + dt_sim * dz
            a_T[j+1] = a_T[j] + dt_sim * da
            om_T[j+1] = om_T[j] + dt_sim * dom           
  
            if(om_T[j+1]< 0)
                println("here is the time for om:", i_t)
                println(dom)
                println()
            end
            if(a_T[j+1]>=0)
                println("here is the time for a:", i_t)
                println(da)
                println()                
            end

            # update hazarad rates and calculate how the density changes:
            lamb_inter  = lambda_skew_normal(alpha, beta, z_T[j+1], om_T[j+1], a_T[j+1])
            P_fire      = (1 - exp(-dt_sim*(A_bin*lamb_T[j]+B_bin*lamb_inter)))
            lamb_T[j+1] = lamb_inter
            dm_T = P_fire * m_T[j]
            m_T[j+1] = m_T[j] - dm_T
            N_T[pos_write] += dm_T
        end
            debugmessenger(poormansdebug, "exit j loop over integral past")

        # evolve free solution, analogous to the above:

        # start with some abbreviations to get rid of annoying arguments
        a    = a_free
        om   = om_free           
        z    = z_free
        lamb = lamb_free
        v1, v2, v3  = moments_of_v_integral_skew_normal(z, om, a)
        d_hat = a/sqrt(1+a^2)*sqrt(2/pi) 

        # start with some abbreviations to save computation:
        exp_0 = exp(beta * z + beta^2*om^2/2)
        exp_1 = a*om/sqrt(2*pi)/sqrt(1+a^2) * exp(-a^2*om^2*beta^2/(2*(1+a^2)))
        Phi_0 = Phi_erf(a*beta*om/sqrt(1+a^2))
        zbo2  = (z+beta*om^2)

        # calculate f1, f2, and f3 as abbreviation for the increments:
        f1 = lamb*v1 + 1/tau * (mu_hat - v1) - 2*alpha*exp_0*(zbo2*Phi_0+exp_1)
        f2 = lamb*v2 + 2/tau * (mu_hat - v2) + 2*D_hat -2*alpha*exp_0*( Phi_0*(om^2+zbo2^2) + exp_1*(2*zbo2-a^2*om^2*beta/(1+a^2)) ) -2*v1*f1
        f3 = lamb*v3 + 3/tau * (mu_hat - v3) + 6*D_hat*v1 - 2*alpha*exp_0*( Phi_0*(3*zbo2*om^2+zbo2^3)+exp_1*(3*om^2+3*zbo2^2-3*a^2*om*2*beta/(1+a^2)*zbo2+a^4*om^4*beta^2/(1+a^2)^2-a^2*om^2/(1+a^2)) )  
                -3*v1*(f2+2*v1*f1)+(6*v1^2-3*v2)*f1

        # calculate increments z, om, a:
        dd  = f3/(3*om^3*d_hat^2)*2/(4-pi)*(1-d_hat^2)-d_hat^2/(2*om^2)*f2
        da  = (1+a^2)^(3/2)*sqrt(pi/2)*dd
        dom = f2/(2*om) + f3/(3*om^2*d_hat)*2/(4-pi)
        dz  = f1- f3/(3*om^2*d_hat^2)*2/(4-pi)

        # update z, om, a:
        z_free = z_free + dt_sim * dz
        a_free = a_free + dt_sim * da
        om_free = om_free + dt_sim * dom  

        # update hazarad rates and calculate how the density changes:
        lamb_inter      = lambda_skew_normal(alpha, beta, z_free, om_free, a_free)
        P_fire          = (1 - exp(-dt_sim*(A_bin*lamb_free+B_bin*lamb_inter)))
        lamb_free       = lamb_inter
        dm_free         = P_fire * m_free
        m_free         -= dm_free
        N_T[pos_write] += dm_free

        # all that fired comes into the first bin in bin-history
        m_T[1] = N_T[pos_write]

        # actual firing rate and save data
        A_T[pos_write]  = N_T[pos_write]/dt_sim
        A_rec[i_rec]   += A_T[pos_write]/n_rec

        # sanity check: Is the density conserved?
        m_inter = 0
        for j_T = 1:n_T
            m_inter+=m_T[j_T]
        end
        m_inter += m_free
        m_rec[i_rec] += m_inter/n_rec

        # update current position for next time step
        pos_now = circular_index(pos_now+1, n_T)
        pos_del = circular_index(pos_del+1, n_T)
    end
    return integral_equation_for_skew_normal_zwa_RESULTS(A_rec, m_rec)
end

function variance_complete_analytical_approximation(param)
    # also done directly in the graphing tool for the paper but here also to play around: give me the analytical approximation of the variance formula
    mu0 = param["mu0"]
    rm  = param["Phimax"]
    w   = param["w"]
    tau = param["tau"]
    N   = param["N"]
    beta= param["beta"]
    p   = param["p"]
    if(!haskey(param, "sig_ext"))
        param["sig_ext"] = 0
    end
    sig_ext = param["sig_ext"]

    Phi_inv_val = sqrt(2)*erfinv(2*(-mu0/(w*rm))-1)
    Q_val       = exp(-0.5*Phi_inv_val^2)
    sig_val     = -w*mu0*(1-p)/(2*tau*N*p)
    Fh_val      = rm*Q_val/sqrt(2*pi)/sqrt(1/beta^2+sig_val)

    inter1    = (mu0/(2*tau*N) + sig_ext/(-2*w))

    var_r       = inter1 * Fh_val
    var_h       = inter1 /Fh_val

    return var_r, var_h
end

function mean_complete_analytical_approximation(param)
    # mean in completely approximate form, should be also in the plotting tool, but here as a function to play around with:
    mu0 = param["mu0"]
    rm  = param["Phimax"]
    w   = param["w"]
    tau = param["tau"]
    N   = param["N"]
    beta= param["beta"]
    p   = param["p"]
    if(!haskey(param, "sig_ext"))
        param["sig_ext"] = 0
    end
    sig_ext = param["sig_ext"]
    sig_val     = -w*mu0*(1-p)/(2*tau*N*p)
    r_val       = -mu0/w

    Phi_inv_val = sqrt(2)*erfinv(2*(-mu0/(w*rm))-1)
    h_val       =  Phi_inv_val * sqrt(1/beta^2 + sig_val) 

    return h_val, sig_val, r_val
end


function capped_exp_hazard(h, param)
#=
    exponential hazard rate, assume theta =0, cap the hazard rate at a maximal value
=#
    return min(param["alpha"] * exp(param["beta"] * h), param["exp_cap"])
end

function exp_hazard_fit_parameter_to_erf(h, param, fp_mode, func_mode)

    # fp_mode: either give fixed point manually or give the h0 
    if(fp_mode=="manual")
        h0 = h
        p  = param["p"]
        s0 = param["w"]*(1-p)/(2*param["tau"]*param["N"]*p)*(h0 - param["mu0"])
    elseif(fp_mode=="automatic")
        h0, s0, r0, _ = PSD_get_one_fix_point(param, false)
    end

    # calculate value of transfer function and its derivative at h0
    if(func_mode =="erf_single")
        beta = param["beta"]
        c0 = Phi_sigm_erf(h0, param)
        c1 = param["Phimax"]/sqrt(2*pi) *beta * exp(-beta^2 * h0^2/2)
    elseif(func_mode =="erf_flattened")
        c0 = r0
        c1 = delFdelh_sig_erf(h0, s0, param)
    end

    # return alpha_exp and beta_exp
    return c0 * exp(-c1/c0*h0), c1/c0
end

function F_exp_capped(h, s, param; capped_yn = true)
    alpha   = param["alpha"]
    beta    = param["beta"]
    
    if(capped_yn)
        exp_cap = param["exp_cap"]
        inter1 = 1/beta*log(exp_cap/alpha)
        return alpha*exp(beta*h+s*beta^2/2) * Phi_erf((inter1-h-beta*s)/sqrt(s)) + exp_cap * (1 - Phi_erf((inter1-h)/sqrt(s)))
    end
    return alpha*exp(beta*h+s*beta^2/2)
end

function calculate_fixed_point_exp_capped(param, hmin, hmax; capped_yn = true)
    w   = param["w"]
    mu0 = param["mu0"]
    if((abs(w) < 1e-10))
        return mu0
    end
    #Phimax  = param["Phimax"]
    N       = param["N"]
    tau     = param["tau"]
    beta    = param["beta"]
    alpha   = param["alpha"]
    p       = param["p"]
    exp_cap = param["exp_cap"]

    if(capped_yn)
        inter1 = 1/beta*log(exp_cap/alpha)
        #f = x-> alpha*exp(beta*x+(w * (1-p)/(2*tau*N*p) *(x - mu0))*beta^2/2) * 0.5 * (1+erf((inter1-x-beta*(w * (1-p)/(2*tau*N*p) *(x - mu0)))/sqrt(2*(w * (1-p)/(2*tau*N*p) *(x - mu0))))) + exp_cap * (1 - 0.5 * (1+erf((inter1-x)/sqrt(2*(w * (1-p)/(2*tau*N*p) *(x - mu0)))))) - (x-mu0)/w
        f = x -> F_exp_capped(x, w*(1-p)/(2*tau*N*p)*(x-mu0), param, capped_yn = capped_yn) - (x -mu0)/w
    else
        f = x-> alpha*exp(beta*x+(w * (1-p)/(2*tau*N*p) *(x - mu0))*beta^2/2)
    end
    #hmin=-20 # remove this line
    println(hmin)
    #hmax=-1 # remove this line
    #rts = find_zeros(f, (hmin, mu0-0.0001), atol=1e-15, rtol=1e-15)
    rts = find_zeros(f, (hmin, hmax), atol=1e-15)
    return rts
end


function test_exp_capped(h, param)
    w   = param["w"]
    mu0 = param["mu0"]
    if((abs(w) < 1e-10))
        return mu0
    end
    #Phimax  = param["Phimax"]
    N       = param["N"]
    tau     = param["tau"]
    beta    = param["beta"]
    alpha   = param["alpha"]
    p       = param["p"]
    exp_cap = param["exp_cap"]

    if(true)
        inter1 = 1/beta*log(exp_cap/alpha)
        f = x-> alpha*exp(beta*x+(w * (1-p)/(2*tau*N*p) *(x - mu0))*beta^2/2) * 0.5 * (1+erf((inter1-x-beta*(w * (1-p)/(2*tau*N*p) *(x - mu0)))/sqrt(2*(w * (1-p)/(2*tau*N*p) *(x - mu0))))) + exp_cap * (1 - 0.5 * (1+erf((inter1-x)/sqrt(2*(w * (1-p)/(2*tau*N*p) *(x - mu0))))))
    else
        f = x-> alpha*exp(beta*x+(w * (1-p)/(2*tau*N*p) *(x - mu0))*beta^2/2)
    end
    return f(h)
end



function exp_capped_get_one_fix_point(param, capped_yn = true; replacement_h =0, replacement_s =0, replacement_r = 0)
    # replacement_x: if no fixed point s found it is likely the case where h0 \approx \mu_0 otherwise put it to zero, so that I know that there might be a problem
    w           = param["w"]
    mu0         = param["mu0"]
    max_search  = param["max_search_for_fixed_point"]
    p           = param["p"]
    tau         = param["tau"]
    C           = param["C"]
    hfix        = 0
    sfix        = 0
    rfix        = 0
    fix_found   = false

    if(w>0)
        hmin = mu0
        hmax = abs(max_search)
    else
        hmin = -abs(max_search)
        hmax = mu0
    end
    fp = calculate_fixed_point_exp_capped(param, hmin, hmax, capped_yn = capped_yn)
    println(fp)
    # there can be 1 or 3 fixed points: take the fixed point with the smallest firing rate
    len_fp = length(fp)
    if(len_fp == 0)
        return replacement_h, replacement_s, replacement_r, fix_found
    end
    fix_found = true
    for i=1:len_fp
        hi = fp[i]
        si = w * (1-p)/(2*tau*C)*(hi - mu0)
        ri = F_exp_capped(hi, si, param, capped_yn = capped_yn)
        if((i==1) || (hi <= hfix))
            hfix = hi
            sfix = si
            rfix = ri
        end
    end
    return hfix, sfix, rfix, fix_found
end


function linear_hazard(h, param)
   return max(0, param["beta_lin"] * (h - param["theta_lin"])) 
end



function linear_hazard_fit_parameter_to_erf(h, param, fp_mode, func_mode)

    # fp_mode: either give fixed point manually or give the h0 
    if(fp_mode=="manual")
        h0 = h
        p  = param["p"]
        s0 = param["w"]*(1-p)/(2*param["tau"]*param["N"]*p)*(h0 - param["mu0"])
    elseif(fp_mode=="automatic")
        h0, s0, r0, _ = PSD_get_one_fix_point(param, false)
    end

    # calculate value of transfer function and its derivative at h0
    if(func_mode =="erf_single")
        beta = param["beta"]
        c0 = Phi_sigm_erf(h0, param)
        c1 = param["Phimax"]/sqrt(2*pi) *beta * exp(-beta^2 * h0^2/2)
    elseif(func_mode =="erf_flattened")
        c0 = r0
        c1 = delFdelh_sig_erf(h0, s0, param)
    end

    # return beta_linear and theta_lin
    return c1, h0 - c0/c1
end
