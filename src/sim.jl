function saveplot(folder, name, file_ending, savefig_yn;dpi=300)
    if(savefig_yn)
        if(file_ending==".png")
            savefig(folder*name*file_ending, dpi=dpi)
        else
            savefig(folder*name*file_ending)
        end
    end
end

function Pfire(V,k, dt)
    if (V>0)
        return -expm1(-k * V^2 * dt)
    else
        return 0
    end
end

function moving_average(x, n, dt)
    M, N = size(x)
    L=N-(n-1)    # size of output array
    t=dt*(n/2:N-n/2)
    x_av=[sum(x[k,j:(j+n-1)])/n for k in 1:M, j in 1:L]
    return t, x_av
end

function moving_average_fft(x, dt, Twindow)
    M , N =size(x)
    freqs = fftfreq(N,1/dt)
    filter=sinc.(Twindow*freqs)
    xF=fft(x,2)
    for i in 1:M
        xF[i,:] = xF[i,:] .* filter
    end
#    xF = xF .* filter'
    return real.(ifft(xF, 2))
end

function edges_to_center_1d(x)
#=
histogram method gives edges to the left or to the right, I want the center of the bins, which have the length = length(x)-1
input:
    x:          edges of the historgam e.g. hist.edges[1]

output:         
out:            center positions of the bins
=#
    out = zeros(Float32,length(x)-1) 
    for i=2:length(x)
        out[i-1]=(x[i]+x[i-1])/2.
    end
    return out
end

function edges_to_center_nd(x) # needs to be completed!!!
#=
histogram method gives edges to the left or to the right, I want the center of the bins, which have the length = length(x)-1
input:
    x:          edges of the historgam e.g. hist.edges[1]

output:         
out:            center positions of the bins
=#
    out = zeros(Float32,length(x)-1) 
    for i=2:length(x)
        out[i-1]=(x[i]+x[i-1])/2.
    end
    return out
end

function calculate_emphirical_curtosis(x)
#=
calculate the emphirical curtosis of a data set
input:
    x:      data array, must have length greater than 1

output:
    out:    emphirical curtosis
=#
    n       = length(x)
    mean_x  = sum(x)/n
    var_x   = sqrt(sum((x-ones(n)*mean_x).^2)/(n-1))
    return sum(((x-ones(n)*mean_x)./var_x).^4)/n
end

function calculate_emphirical_skewness(x)
#=
calculate the emphirical skew of a data set
input:
    x:      data array, must have length greater than 1

output:
    out:    emphirical skew
=#
    n       = length(x)
    mean_x  = sum(x)/n
    var_x   = sqrt(sum((x-ones(n)*mean_x).^2)/(n-1))
    return sum(((x-ones(n)*mean_x)./var_x).^3)/n
end


#=
function Phi_1(x, Phi0, alpha)
    gmax=25
    xmax=25^(1/alpha)
    if (x>0)
        if (x<xmax)
            return Phi0 * x^alpha
        else
            return Phi0 * gmax
        end
    else
        return 0
    end
end
=#

function Phi_1(x, param)
    gmax=25
    xmax=25^(1/param["alpha"])
    if (x>0)
        if (x<xmax)
            return param["Phi0"] * x^param["alpha"]
        else
            return param["Phi0"] * gmax
        end
    else
        return 0
    end
end


function Phi_2(x, param)
    if (x>0)
        return param["Phi0"] * x^param["alpha"]
    else
        return 0
    end
end

function Phi_2_2_arg(x, y, param)
    return Phi_2(x, param)
end

function Phi_rect_pow(x, param)
    if(x>0)
        unbounded = param["Phi0"]*x^param["alpha"]
    else
        unbounded = 0.
    end
    return min(unbounded, param["Phimax"])
end

function sigm_erf(x)
    return 0.5*(1+erf(x/sqrt(2)))
end

function Phi_sigm_erf(x, param)
    return param["Phimax"]*sigm_erf(param["beta"]*(x-param["theta"]))
end

function F_rect_quad_full(h, s, para_F_rect_quad_full)
    phi0 = para_F_rect_quad_full["Phi0"]
    return phi0/2 * (h^2+s)*(1+erf(h/(sqrt(2*s)))) + phi0*h*sqrt(s)/sqrt(2*pi)*exp(-h^2/(2*s))
end

function F_rect_quad_full_save(h, s, para_F_rect_quad_full)
    #=
    save function F_rect_quad_full for the case s==0 and h==0
    =#
    phi0 = para_F_rect_quad_full["Phi0"]
    if(s==0)
        return Phi_2(x, para_F_rect_quad_full)
    end
    return phi0/2 * (h^2+s)*(1+erf(h/(sqrt(2*s)))) + phi0*h*sqrt(s)/sqrt(2*pi)*exp(-h^2/(2*s))
end


function binom_func(alpha, h, s2, k)
    return binomial(alpha, k)*h^(alpha-k)*(s2)^(k/2) * gamma((k+1)/2)
end

function F_rect_pow_full(h, s, param)
    if(s>0.)
        round_yn= param["round_yn"]
        alpha   = Int(param["alpha"])
        hmax    = param["hmax"]
        s2      = 2*s
        Dh      = hmax - h
        out = 0.
        if(h<0)
            for k=0:alpha
                out+= binom_func(alpha, h, s2, k) * ( gamma_inc((k+1)/2, h^2/s2)[2] - gamma_inc((k+1)/2, Dh^2/s2)[2] )
            end
        else
            if(Dh >= 0.)    
                for k=0:alpha
                    out += binom_func(alpha, h, s2, k) * ( (1+(-1)^k) -1*(-1)^k*gamma_inc((k+1)/2, h^2)[2] - gamma_inc((k+1)/2, Dh^2/s2)[2] )   
                end
            else
                for k=0:alpha
                    out += binom_func(alpha, h, s2, k) * (-1)^k * ( gamma_inc((k+1)/2, Dh^2/s2)[2] - gamma_inc((k+1)/2, h^2/s2)[2] ) 
                end
            end        
        end
        if(round_yn)
            return max( param["Phi0"]/sqrt(pi)*out/2+param["Phimax"]/2.0 * (1.0 - erf(Dh/sqrt(s2))), 1e-10)
        else
            return  param["Phi0"]/sqrt(pi)*out/2+param["Phimax"]/2.0 * (1.0 - erf(Dh/sqrt(s2)))
        end
    else
        if(param["hmax"]>h)
            if(h>0)
                return param["Phi0"]*h^param["alpha"]
            else
                return 0.0
            end
        else
            return param["Phimax"]
        end
    end
end

function gauss_times_phi(x, phi, h, s, param_phi)
    return 1.0/sqrt(2*pi*s)*exp(-(x-h)^2/(2*s))*phi(x, param_Phi)
end

function F_rect_pow_full_numerical(h, s, param)
    if(s>0)
        round_yn= param["round_yn"]
        alpha   = Int(param["alpha"])
        hmax    = param["hmax"]
        rel_tol = param["rel_tol"]
        return quadgk(x -> gauss_times_phi(x, Phi_rect_pow, h, s, param), 0, hmax, rtol=rel_tol)[1] + param["Phimax"]/2*erfc((hmax-h)/sqrt(2*s))
    else
        if(param["hmax"]>h)
            if(h>0)
                return param["Phi0"]*h^param["alpha"]
            else
                return 0.0
            end
        else
            return param["Phimax"]
        end
        
    end

end

function F_sigm_erf(h, s, param)
    beta = param["beta"]    
    return param["Phimax"].*sigm_erf.(beta./ sqrt.(1 .+s.*beta^2) .* (h .-param["theta"]))

end

function F_rect_pow_full_wrong(h, s, param)
    if(s>0.)
        alpha   = Int(param["alpha"])
        hmax    = param["hmax"]
        s2      = 2*s
        out = 0.
        if(h<0)
            print("h<0\r")
            for k=0:alpha     
                out += round(binomial(alpha, k)*h^(alpha-k)*(s2)^(k/2)*(1)*( gamma((k+1)/2, h^2/s2) - gamma((k+1)/2, (hmax-h)^2/s2) ), digits=10)# oder nach unten begrenzen
                #out += binomial(alpha, k)*h^(alpha-k)*(s2)^(k/2)*(1)*( gamma((k+1)/2, h^2/s2) - gamma((k+1)/2, (hmax-h)^2/s2) )
            end
        else    
            for k=0:alpha     
                out += binomial(alpha, k)*h^(alpha-k)*(s2)^(k/2)* 
                        ( gamma_inc((k+1)/2, (hmax-h)^2/s2)[1] + (-1)^(k) * gamma_inc((k+1)/2, (h)^2/s2)[1] ) *gamma((k+1)/2)
            end        
        end
        return  param["Phi0"]/sqrt(pi)*out/2+param["Phimax"]/2.0 * (1.0 - erf((hmax-h)/sqrt(s2)))
    else
        if(param["hmax"]>h)
            if(h>0)
                return param["Phi0"]*h^param["alpha"]
            else
                return 0.0
            end
        else
            return param["Phimax"]
        end
    end
end


function f_h_rect_quad_full_int(h, s, para_F_rect_quad_full_int; noise=0.)
    F   = F_rect_quad_full(h, s, para_F_rect_quad_full_int)
    mu  = para_F_rect_quad_full_int["mu"]
    w   = para_F_rect_quad_full_int["w"]
    N   = para_F_rect_quad_full_int["N"]
    tau = para_F_rect_quad_full_int["tau"]
    dt  = para_F_rect_quad_full_int["dt"]
    return (-h+mu+w*(F + sqrt(F/N)*noise/sqrt(dt)))/tau
end

function f_s_rect_quad_full_int(h, s, para_F_rect_quad_full_int; noise=0.)
    F   = F_rect_quad_full(h, s, para_F_rect_quad_full_int)
    w   = para_F_rect_quad_full_int["w"]   
    tau = para_F_rect_quad_full_int["tau"]
    N   = para_F_rect_quad_full_int["N"]
    C   = para_F_rect_quad_full_int["C"]
    return (-2*s + w^2 * (1-C/N)/(C*tau) * F) / tau
end

function f_h_general_non_fully_connected(h, s, F_general, param; noise=0.)
    F   = F_general(h, s, param)
    if(F<0)
        println("here")
    end
    mu  = param["mu"]
    w   = param["w"]
    N   = param["N"]
    tau = param["tau"]
    dt  = param["dt"]
    return (-h+mu+w*(F + sqrt(F/N)*noise/sqrt(dt)))/tau
end

function f_s_general_non_fully_connected(h, s, F_general, param; noise=0.)
    F   = F_general(h, s, param)
    w   = param["w"]   
    tau = param["tau"]
    N   = param["N"]
    C   = param["C"]
    return (-2*s + w^2 * (1-C/N)/(C*tau) * F) / tau  
end

function f_s_general_naive(h, s, F_general, param; noise=0.)
    return 0.0
end

function f_h_rect_quad_naive(h, s, para_rect_quad_naive; noise=0.)
    #para_rect_quad_naive["alpha"]=2
    F   = Phi_2(h, para_rect_quad_naive)    # you MUST choose alpha = 2
    mu  = para_rect_quad_naive["mu"]
    w   = para_rect_quad_naive["w"]
    N   = para_rect_quad_naive["N"]
    tau = para_rect_quad_naive["tau"]
    dt  = para_rect_quad_naive["dt"]    
    return (-h+mu+w*(F + sqrt(F/N)*noise/sqrt(dt)))/tau
end


function f_s_rect_quad_naive(h, s, para_rect_quad_naive; noise=0.)
    # s0 must be chosen as 0 as well in the Euler Integrator!
    return 0.
end

function F_sigm_erf_shifted(h, s, param)  
    return F_sigm_erf(h, s+param["sig_mu"], param)
end

function Euler_integrator_rect_quad_full_integral(h0, s0, f_h, f_s, F, para_Euler_integrator; supress_noise_yn=false)
    # is also capable of non-rect quad full integrals 
    tmax    = para_Euler_integrator["tmax"]
    dt      = para_Euler_integrator["dt"]
    delta   = para_Euler_integrator["delta"]
    N       = para_Euler_integrator["N"]
    nSteps  = trunc(Int, tmax/dt)
    h       = zeros(nSteps+1)
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
    s[1:delay+1] .= s0
    A[1:delay+1] .= F(h0, s0, para_Euler_integrator)
    r[1:delay+1] .= F(h0, s0, para_Euler_integrator)

    for i=1:(nSteps-delay)
        h[i+1+delay]    = h[i+delay] + f_h(h[i], s[i], para_Euler_integrator, noise=noise[i]) * dt
        s[i+1+delay]    = s[i+delay] + f_s(h[i], s[i], para_Euler_integrator, noise=noise[i]) * dt
        r_inter         = F(h[i+1+delay], s[i+1+delay], para_Euler_integrator)
        r[i+1+delay]    = r_inter
        A[i+1+delay]    = (r_inter + sqrt(r_inter/N)*noise[i+1+delay]/sqrt(dt))
    end
    return h, s, A, r
end

function Euler_integrator_general_transf_full_int(h0, s0, f_h, f_s, F, para_Euler_integrator; supress_noise_yn=false)
#=
    F       : full integral of Gaussian distribution over transfer function
=#
    tmax    = para_Euler_integrator["tmax"]
    dt      = para_Euler_integrator["dt"]
    delta   = para_Euler_integrator["delta"]
    N       = para_Euler_integrator["N"]
    nSteps  = trunc(Int, tmax/dt)
    h       = zeros(nSteps+1)
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
    s[1:delay+1] .= s0
    A[1:delay+1] .= F(h0, s0, para_Euler_integrator)
    r[1:delay+1] .= F(h0, s0, para_Euler_integrator)

    for i=1:(nSteps-delay)
        if(s[i]<0)
            println("here")
        end
        h[i+1+delay]    = h[i+delay] + f_h(h[i], s[i], F, para_Euler_integrator, noise=noise[i]) * dt
        s[i+1+delay]    = s[i+delay] + f_s(h[i], s[i], F, para_Euler_integrator, noise=noise[i]) * dt
        if(s[i+1+delay]<0)
            print("here")
        end
        r_inter         = F(h[i+1+delay], s[i+1+delay], para_Euler_integrator)
        r[i+1+delay]    = r_inter
        A[i+1+delay]    = (r_inter + sqrt(r_inter/N)*noise[i+1+delay]/sqrt(dt))
    end
    return h, s, A, r
end

function f_h_general_non_fully_connected_delay(h, s, r, param; noise=0.)
    mu  = param["mu"]
    w   = param["w"]
    N   = param["N"]
    tau = param["tau"]
    dt  = param["dt"]
    return (-h+mu+w*(r + sqrt(r/N)*noise/sqrt(dt)))/tau
end

function f_s_general_non_fully_connected_delay(h, s, r, param; noise=0.)
    w   = param["w"]   
    tau = param["tau"]
    N   = param["N"]
    C   = param["C"]
    return (-2*s + w^2 * (1-C/N)/(C*tau) * r) / tau  
end

function f_s_general_naive_delay(h, s, r, param; noise=0.)
    return 0.0
end

function Euler_integrator_general_transf_full_int_delay(h0, s0, f_h, f_s, F, para_Euler_integrator; supress_noise_yn=false, give_out_xi=false)
#=
    F       : full integral of Gaussian distribution over transfer function
=#
    tmax    = para_Euler_integrator["tmax"]
    dt      = para_Euler_integrator["dt"]
    delta   = para_Euler_integrator["delta"]
    N       = para_Euler_integrator["N"]
    nSteps  = trunc(Int, tmax/dt)
    h       = zeros(nSteps+1)
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
    s[1:delay+1] .= s0
    A[1:delay+1] .= F(h0, s0, para_Euler_integrator)
    r[1:delay+1] .= F(h0, s0, para_Euler_integrator)

    for i=1:(nSteps-delay)
        if(s[i]<0)
            println("here")
        end
        h[i+1+delay]    = h[i+delay] + f_h(h[i+delay], s[i+delay], r[i], para_Euler_integrator, noise=noise[i]) * dt
        s[i+1+delay]    = s[i+delay] + f_s(h[i+delay], s[i+delay], r[i], para_Euler_integrator, noise=noise[i]) * dt
        if(s[i+1+delay]<0)
            print("here")
        end
        r_inter         = F(h[i+1+delay], s[i+1+delay], para_Euler_integrator)
        r[i+1+delay]    = r_inter
        A[i+1+delay]    = (r_inter + sqrt(r_inter/N)*noise[i+1+delay]/sqrt(dt))
    end
    if(give_out_xi)
        return h, s, A, r, noise/sqrt(dt)
    end
    return h, s, A, r
end

function del_F_del_h(h, s, para_del_F_del_h)
    phi0 = para_del_F_del_h["Phi0"]    
    erf_val = erf.(h./(sqrt.(2 .*s)))
    exp_val = exp.(-h.^2 ./(2 .*s))
    out     = h .* (1 .+erf_val)
    out     += (h.^2+s) ./sqrt.(2*pi .*s).*exp_val
    out     += sqrt.(s./(2*pi)).*exp_val
    out     -= -h.^2/sqrt.(2*pi .*s).*exp_val
    return phi0 .*out
end

function del_F_del_s(h, s, para_del_F_del_s)
    phi0 = para_del_F_del_s["Phi0"]    
    erf_val = erf.(h./(sqrt.(2 .*s)))
    exp_val = exp.(-h.^2 ./(2 .*s))
    out     = (1 .+erf_val)
    out     -= 0.5 .*(h.^2 .+s).*exp_val.*h./sqrt(2*pi).*s.^(-3/2)
    out     += 0.5 .*h ./sqrt.(2*pi .*s).*exp_val
    out     += h.*sqrt.(2*pi.*s)*exp_val.*h.^2 ./(2 .*s^.2)
    return phi0.*out
end

function calculate_analytical_PSD_meso(f, h0, s0, r0, para_ana_PSD; a=1)
#=
Depricated
=#
    tau = para_ana_PSD["tau"]
    N   = para_ana_PSD["N"]
    K1  = (para_ana_PSD["w"])^2/tau * (1/para_ana_PSD["C"] - 1/N)
    K3  = del_F_del_h(h0, s0, para_ana_PSD)
    K4  = del_F_del_s(h0, s0, para_ana_PSD)

    y = K1 .* K3 ./(2 .+ im .* tau .* 2 .* pi .* f .- K1 .* K4)
    z = a ./ (1 .+ im .* tau .* 2 .* pi .* f .- K3 .* a - K4 .* a .* y)

    return r0./N .* (1 .+ abs.(z).^2 .* (K3.^2 .+ K4.^2 .* abs.(y).^2 .+ 2 .* K3 .* K4 .* real.(y)) .+ 2 .* real.(z .* (K3 .+ K4 .* y)))
end



function calculate_PSD(x; dt=1, n=100, full_length=false, t_relax=0.)
#=
calculate the power spectral density of the input array x
input:
    x:              array with equal time steps in between measurements
    dt:             time steps of array x
    n:              subdivide x into n-sized subintervalls for the PSD
    full_length:    take x as one big intervall
    t_relax:        discard the first time t_relax

output:
    f:              frequency of the PSD
    PSD:            power spectral density
=#

    n_relax = trunc(Int, t_relax/dt)
    y       = x[(n_relax+1):end]
    N_max   = length(y)
    if(full_length)
        n   = N_max
    end
    recast_length   = N_max-(N_max%n)
    nr_cols         = trunc(Int, round(recast_length/n))
    y   = reshape(y[1:recast_length], (n, nr_cols))   # in julia: (a, b) = (rows, col), columnwise sorted!
#    yf  = FFTW.fft(y)                                # damnit!
    yf  = FFTW.fft(y, (1,))
    df  = 1.0/(n*dt)
    #f   = df .* ((1:trunc(Int, n/2)).-1) .+ df                          # these two lines must be changed
    f   = df .* (1:(trunc(Int, n/2)-1))
    #psd = mean((abs.(yf)).^2 ./((n-1)./dt), dims=2)[1:trunc(Int, n/2)]  # these two lines must be changed
    psd = mean((abs.(yf)).^2 ./((n-1)./dt), dims=2)[2:trunc(Int, n/2)]  # these two lines must be changed
    return f, psd
    # better be f[2:end], psd[2:end]  ???
end


function calculate_cross_spectrum(x, y; dt=1, n=100, full_length=false, t_relax=0.)
#=
calculate the power spectral density of the input array x
input:
    x:              array with equal time steps in between measurements
    dt:             time steps of array x
    n:              subdivide x into n-sized subintervalls for the PSD
    full_length:    take x as one big intervall
    t_relax:        discard the first time t_relax

output:
    f:              frequency of the PSD
    PSD:            power spectral density
=#

    n_relax = trunc(Int, t_relax/dt)
    x       = x[(n_relax+1):end]
    y       = y[(n_relax+1):end]
    N_max   = length(x)
    if(full_length)
        n   = N_max
    end
    recast_length   = N_max-(N_max%n)
    nr_cols         = trunc(Int, round(recast_length/n))
    x   = reshape(x[1:recast_length], (n, nr_cols))   # in julia: (a, b) = (rows, col), columnwise sorted!
    y   = reshape(y[1:recast_length], (n, nr_cols))
#    yf  = FFTW.fft(y)                                # damnit!
    xf  = FFTW.fft(x, (1,))
    yf  = FFTW.fft(y, (1,))
    df  = 1.0/(n*dt)
    #f   = df .* ((1:trunc(Int, n/2)).-1) .+ df                          # these two lines must be changed
    f   = df .* (1:(trunc(Int, n/2)-1))
    #psd = mean((abs.(yf)).^2 ./((n-1)./dt), dims=2)[1:trunc(Int, n/2)]  # these two lines must be changed
    cpsd = mean(conj(yf) .* xf ./((n-1)./dt), dims=2)[2:trunc(Int, n/2)]  # these two lines must be changed
    return f, cpsd
    # better be f[2:end], psd[2:end]  ???
end


function delFdelh_sig_erf(h, s, param)
    beta = param["beta"]
    #return param["Phimax"]/sqrt(2*pi)*exp(-0.5*beta^2*(h-param["theta"])^2/(1+beta^2*s))*beta/(1+beta^2*s)
    return param["Phimax"]/sqrt(2*pi)*exp(-0.5*beta^2*(h-param["theta"])^2/(1+beta^2*s))*beta/(1+beta^2*s)^(1/2)
end

function delFdels_sig_erf(h, s, param)
    beta = param["beta"]
    theta= param["theta"]
    return -0.5*param["Phimax"]/sqrt(2*pi)*exp(-0.5*beta^2*(h-theta)^2/(1+beta^2*s))*beta^3*(h-theta)*(1+beta^2*s)^(-3/2)
end

function calculate_PSD_analytically(f, delFdelh, delFdels, atilde, h0, param)
    #=
    atilde is the kernel for int ds A(s)a(t-s). We have in the standard case a = con_ii delta(t), therefore atilde = con_ii  
    =#
    w   = param["w"]
    tau = param["tau"]
    p   = param["p"]
    C   = param["C"]
    mu0 = param["mu0"]
    N   = param["N"]
    delay = param["delay"]
    om  = 2 .* pi .* f
    r0  = (h0-mu0)/w            # implement special case w=0
    s0  = w*(1-p)/(tau*C*2)*(h0-mu0)
    Kh  = delFdelh(h0, s0, param)
    Ks  = delFdels(h0, s0, param)
    K1  = w^2*(1-p)/(tau*C)

    atilde = atilde .* exp.(-1im .* om .* delay)

    y   = K1.*Kh ./ (2 .+ 1im .* om .* tau .- K1*Ks)
    zhat= atilde ./ (1 .+ 1im .* om .* tau .- Kh .* atilde .- Ks .* atilde .* y)

    return r0/N * ( 1 .+ 2 .* real.(zhat .* (Kh .+ Ks .* y)) .+ (abs.(Kh .+ Ks .* y)).^2 .* (abs.(zhat)).^2  )
end

function calculate_PSD_analytically_naive(f, delFdelh, delFdels, atilde, h0, param)
    #=
    atilde is the kernel for int ds A(s)a(t-s). We have in the standard case a = con_ii delta(t), therefore atilde = con_ii  
    =#
    w   = param["w"]
    tau = param["tau"]
    p   = param["p"]
    C   = param["C"]
    mu0 = param["mu0"]
    N   = param["N"]
    delay = param["delay"]
    om  = 2 .* pi .* f
    r0  = (h0-mu0)/w            # implement special case w=0
    s0  = 0#w*(1-p)/(tau*C)*(h0-mu0)
    Kh  = delFdelh(h0, s0, param)
    Ks  = delFdels(h0, s0, param)
    K1  = w^2*(1-p)/(tau*C)

    atilde = atilde .* exp.(-1im .* om .* delay)

    return r0/N * ( Kh^2*w ./(abs.(1 .- 1im .* om .* tau .-(w*Kh))).^2 .+ 1 .+ (2*Kh*w) .* real.(1 ./(1 .- 1im .* om .* tau .-(w*Kh))))
end

function calculate_PSDh_Risken(f, delFdelh, delFdels, atilde, h0, param)
    #=
    atilde is the kernel for int ds A(s)a(t-s). We have in the standard case a = con_ii delta(t), therefore atilde = con_ii  
    =#
    w   = param["w"]
    tau = param["tau"]
    p   = param["p"]
    C   = param["C"]
    mu0 = param["mu0"]
    N   = param["N"]
    delay = param["delay"]
    om  = 2 .* pi .* f
    r0  = (h0-mu0)/w            # implement special case w=0
    s0  = w*(1-p)/(tau*C*2)*(h0-mu0)
    Kh  = delFdelh(h0, s0, param)
    Ks  = delFdels(h0, s0, param)
    K1  = w^2*(1-p)/(tau*C)

    atilde = atilde .* exp.(-1im .* om .* delay)

    det_plus    = 1/tau^2*(1-Kh*w)*(2-K1*Ks) .+ 1im .* om ./tau .* (3-Kh*w-K1*Ks) .- K1*Kh*Ks*w/tau^2
    det_minus   = 1/tau^2*(1-Kh*w)*(2-K1*Ks) .- 1im .* om ./tau .* (3-Kh*w-K1*Ks) .- K1*Kh*Ks*w/tau^2

    return 2 .* ( (1/tau*(2-K1*Ks) .+ 1im .* om).^2 .+ om.^2 ) ./(det_plus .* det_minus) .* (r0/N*w^2/tau^2)
end

function calculate_PSD_A_Risken(f, delFdelh, delFdels, atilde, h0, param)
    #=
    atilde is the kernel for int ds A(s)a(t-s). We have in the standard case a = con_ii delta(t), therefore atilde = con_ii  
    =#
    w   = param["w"]
    tau = param["tau"]
    p   = param["p"]
    C   = param["C"]
    mu0 = param["mu0"]
    N   = param["N"]
    delay = param["delay"]
    om  = 2 .* pi .* f
    r0  = (h0-mu0)/w            # implement special case w=0
    s0  = w*(1-p)/(tau*C*2)*(h0-mu0)#newlinebugfix : *2
    Kh  = delFdelh(h0, s0, param)
    Ks  = delFdels(h0, s0, param)
    K1  = w^2*(1-p)/(tau*C)

    #atilde = atilde .* exp.(-1im .* om .* delay)

    det_plus    = 1/tau^2*(1-Kh*w)*(2-K1*Ks) .+ 1im .* om ./tau .* (3-Kh*w-K1*Ks) .- K1*Kh*Ks*w/tau^2 .-om.^2
    det_minus   = 1/tau^2*(1-Kh*w)*(2-K1*Ks) .- 1im .* om ./tau .* (3-Kh*w-K1*Ks) .- K1*Kh*Ks*w/tau^2 .-om.^2

    #Shh = (2*r0/N*w^2/tau^2) .* ((2-K1*Ks)/tau .+ 1im .* om) .* ((2-K1*Ks)/tau .- 1im .* om) ./(det_minus .* det_plus)
    #Sss = (2*r0/N*w^2/tau^2) .* K1^2 * Kh^2/tau^2 ./(det_minus .* det_plus)
    #Ahs = (2*r0/N*w^2/tau^2) .* (2*Kh^2*Ks*K1)/tau .* real.( ((2-K1*Ks)/tau .+ 1im .* om) ./(det_minus .* det_plus) )
    #Ahe = (4*r0/N*w/tau*Kh) .* real.( ((2-K1*Ks)/tau .+ 1im .* om) ./det_plus )
    #Ase = (4*r0/N*w/tau*Ks*K1*Kh)/tau .* real.(1 ./ det_plus)

    Shh = (2*r0/N*w^2/tau^2) .* ((2-K1*Ks)/tau .+ 1im .* om) .* ((2-K1*Ks)/tau .- 1im .* om) ./(det_minus .* det_plus) ./w^2
    Sss = (2*r0/N*w^2/tau^2) .* K1^2 * Kh^2/tau^2 ./(det_minus .* det_plus) ./w^2
    Ahs = (2*r0/N*w^2/tau^2) .* (2*Kh^2*Ks*K1)/tau .* real.( ((2-K1*Ks)/tau .+ 1im .* om) ./(det_minus .* det_plus) ) ./w^2
    Ahe = (4*r0/N*w/tau*Kh) .* real.( ((2-K1*Ks)/tau .+ 1im .* om) ./det_plus ) ./w
    Ase = (4*r0/N*w/tau*Ks*K1*Kh)/tau .* real.(1 ./ det_plus) ./w
    
    return (2*r0/N .+ Shh .+ Sss .+ Ahs .+ Ahe .+ Ase)./2
end

function calculate_PSDh_compare(f, delFdelh, delFdels, atilde, h0, param)
    #=
    atilde is the kernel for int ds A(s)a(t-s). We have in the standard case a = con_ii delta(t), therefore atilde = con_ii  
    =#
    w   = param["w"]
    tau = param["tau"]
    p   = param["p"]
    C   = param["C"]
    mu0 = param["mu0"]
    N   = param["N"]
    delay = param["delay"]
    om  = 2 .* pi .* f
    r0  = (h0-mu0)/w            # implement special case w=0
    #s0  = w*(1-p)/(tau*C)*(h0-mu0)
    #new line bug fix:
    s0 = w*(1-p)/(2*tau*C)*(h0-mu0)
    Kh  = delFdelh(h0, s0, param)
    Ks  = delFdels(h0, s0, param)
    K1  = w^2*(1-p)/(tau*C)

    atilde = atilde .* exp.(-1im .* om .* delay)

    y   = K1.*Kh ./ (2 .+ 1im .* om .* tau .- K1*Ks)
    zhat= atilde ./ (1 .+ 1im .* om .* tau .- Kh .* atilde .- Ks .* atilde .* y)

    return r0/N * (abs.(zhat)).^2
end


function test_power_1(f, delFdelh, delFdels, h0, param)
    w   = param["w"]
    tau = param["tau"]
    p   = param["p"]
    C   = param["C"]
    mu0 = param["mu0"]
    N   = param["N"]
    om  = 2 .* pi .* f   
    r0  = (h0-mu0)/w
    s0  = w*(1-p)/(2*tau*C)*(h0-mu0)
    Kh  = delFdelh(h0, s0, param)
    Ks  = delFdels(h0, s0, param)
    K1  = w^2*(1-p)/(tau*C)

    y   = (K1*Kh) ./ (2 .+ 1im .* om .* tau .- (K1*Ks))
    zhat= w ./(1 .+ 1im .* om .* tau .- (w*Kh) .- (w*Ks) .* y)
    htil= sqrt(r0/N) .* zhat                                    # I omit the fourier transform of WGN
    stil= sqrt(r0/N) .* zhat .* y                               # I omit the fourier transform of WGN
    Atil= Kh .* htil .+ Ks .* stil .+ (1*sqrt(r0/N))            # I omit the fourier transform of WGN

    return (abs.(Atil)).^2
end


function test_power_2(f, delFdelh, delFdels, h0, param)
    w   = param["w"]
    tau = param["tau"]
    p   = param["p"]
    C   = param["C"]
    mu0 = param["mu0"]
    N   = param["N"]
    om  = 2 .* pi .* f   
    r0  = (h0-mu0)/w
    s0  = w*(1-p)/(2*tau*C)*(h0-mu0)
    Kh  = delFdelh(h0, s0, param)
    Ks  = delFdels(h0, s0, param)
    K1  = w^2*(1-p)/(tau*C)

    det_plus    = 1im .* om./tau .* (3-K1*Ks-w*Kh) .- om.^2 .+ 1 ./tau^2 .* (2-K1*Ks).*(1-w*Kh) .- 1 ./tau^2 * w*Ks*K1*Kh
    htil= 1 ./ det_plus .* ((2-K1*Ks)/tau .+ 1im .* om) .* w/tau .* sqrt(r0/N)  # I omit the fourier transform of WGN
    stil= 1 ./ det_plus .* (K1*Kh/tau) .* w/tau .* sqrt(r0/N)                   # I omit the fourier transform of WGN
    Atil= Kh .* htil .+ Ks .* stil .+ (1*sqrt(r0/N))                            # I omit the fourier transform of WGN

    return (abs.(Atil)).^2
end

function test_power_3(f, delFdelh, delFdels, h0, param)
    w   = param["w"]
    tau = param["tau"]
    p   = param["p"]
    C   = param["C"]
    mu0 = param["mu0"]
    N   = param["N"]
    om  = 2 .* pi .* f   
    r0  = (h0-mu0)/w
    s0  = 0.
    Kh  = delFdelh(h0, s0, param)
    Ks  = delFdels(h0, s0, param)

    htil= (w*sqrt(r0/N)) ./ (1 .+ 1im .* om .* tau .- w*Kh) # I omit the fourier transform of WGN
    Atil= Kh .* htil .+ (1*sqrt(r0/N))                      # I omit the fourier transform of WGN

    return (abs.(Atil)).^2
end


function FI_curve_difference(h, F, param_FI_curve; naive_model=false)
    w   = param_FI_curve["w"]
    mu0 = param_FI_curve["mu0"]
    tau = param_FI_curve["tau"]
    p   = param_FI_curve["p"]
    N   = param_FI_curve["N"]
    if(naive_model)
        s0 = 0
    else
        s0  = w * (1 - p)/(2*tau*N*p)*(h-mu0)
    end
    return F(h, s0, param_FI_curve) - (h-mu0)/w
end

function find_fixed_points_FI_curve(F, param_FI_curve; naive_model=false)
    #=
        This function gives a wrong answer, the function FI_curve_difference is definitely fine, but two different packages
        only give cryptic error messages when 
    =#
    hmin = param_FI_curve["hmin"]
    hmax = param_FI_curve["hmax"]
    if(abs(param_FI_curve["w"]) > 1e-10)
        f = x-> FI_curve_difference(x, F, param_FI_curve, naive_model)
        ff= x-> f.(x)
        #root_arr = roots(x -> FI_curve_difference(x, F, param_FI_curve, naive_model), hmin..hmax)
        root_arr = find_zero(ff, (hmin, hmax), Roots.Order1())
    else
        root_arr = param_FI_curve["mu0"]
    end
    return root_arr
end

function calculate_limit_w_minus_firing_rate_sigmoidal(param)
    p = param["p"]
    if((1-p<1e-10) || (p==1.0))
        return 0
    end
    Phimax = param["Phimax"]
    N = param["N"]
    tau= param["tau"]
    A = (1-p)/(2*tau*N*p)
    f = x-> Phimax*sigm_erf(-sqrt(abs(x))/(sqrt(A)))-x
    rts = find_zero(f, (0, Phimax))
    return rts
end

#erf2(x::Interval{T}) where T = Interval(prevfloat(erf(x.lo)), nextfloat(erf(x.hi)))
#function sigm_erf2(x)
#    return 0.5*(1+erf2(x/sqrt(2)))
#end

function calculate_fixed_point_sigmoidal_erf(param, hmin, hmax; naive_model = false)
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
    if(!haskey(param, "sparse_limit_yn"))   # old versions of this program do not have this parameter, 
                                            # the parameter would be false
        param["sparse_limit_yn"] = false
    end
    if(param["sparse_limit_yn"])
        if(!haskey(param, "C"))
            C = N * p    
        else
            C = param["C"]
        end
        A = w/(2*tau*C)
    else
        A = w*(1-p)/(2*tau*N*p)
        println(A)
    end
    
    if(naive_model)
        f = x -> Phimax*sigm_erf(beta*(x-theta))-(x-mu0)/w
    else
        #f = x-> Phimax*sigm_erf(beta*(x-theta)/(sqrt(abs(1+beta^2*A*(x-mu0)))))-(x-mu0)/w
        f = x-> Phimax*sigm_erf(beta*(x-theta)/( sqrt(1+beta^2*A*(x-mu0)) ) )-(x-mu0)/w
    end
    #rts = find_zeros(f, (hmin, hmax), atol=1e-7, rtol=1e-7)
    rts = find_zeros(f, (hmin, mu0-0.0001), atol=1e-15, rtol=1e-15)
    #rts = roots(f, hmin..hmax)
    return rts
end

function calculate_fixed_point_sigmoidal_erf_in_sparse_limit(param, hmin, hmax; naive_model = false)
    #=
        additional function in peer review 2025-03-17, we take the limit $N\to\infty$, $p\to 0$ and C = const
    =#

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
    C       = param["C"]
    A = w*(1-p)/(2*tau*N*p)
    A = w/(2*tau*C)
    
    if(naive_model)
        f = x -> Phimax*sigm_erf(beta*(x-theta))-(x-mu0)/w
    else
        #f = x-> Phimax*sigm_erf(beta*(x-theta)/(sqrt(abs(1+beta^2*A*(x-mu0)))))-(x-mu0)/w
        f = x-> Phimax*sigm_erf(beta*(x-theta)/(sqrt((1+beta^2*A*(x-mu0)))))-(x-mu0)/w
    end
    #rts = find_zeros(f, (hmin, hmax), atol=1e-7, rtol=1e-7)
    rts = find_zeros(f, (hmin, mu0), atol=1e-15, rtol=1e-15)
    #rts = roots(f, hmin..hmax)
    return rts
end

function calculate_fixed_point_sigmoidal_erf_MFsparselim_hypothetical(param, hmin, hmax; naive_model = false)
    #=
        MFsparselim, but take hypothetical C = Np
    =#

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
    #C       = param["C"]
    #A = w*(1-p)/(2*tau*N*p)
    A = w/(2*tau*N*p)
    
    #f = x-> Phimax*sigm_erf(beta*(x-theta)/(sqrt(abs(1+beta^2*A*(x-mu0)))))-(x-mu0)/w
    f = x-> Phimax*sigm_erf(beta*(x-theta)/(sqrt((1+beta^2*A*(x-mu0)))))-(x-mu0)/w

    #rts = find_zeros(f, (hmin, hmax), atol=1e-7, rtol=1e-7)
    rts = find_zeros(f, (hmin, mu0), atol=1e-15, rtol=1e-15)
    #rts = roots(f, hmin..hmax)
    return rts
end




function calculate_Cov_mat_sigmoidal_erf(param; naive_model=false)
    hmin    = param["hmin"]
    hmax    = param["hmax"]
    w       = param["w"]
    N       = param["N"]
    tau     = param["tau"]
    p       = param["p"]
    mu0     = param["mu0"]
    D       = param["D"]
    h_fix   = minimum( calculate_fixed_point_sigmoidal_erf(param, hmin, hmax, naive_model=naive_model) )
    if(naive_model)
        s_fix = 0.
    else
        s_fix = w * (1-p)/(2*tau*N*p) * (h_fix - mu0)
    end
    r0  = F_sigm_erf(h_fix, s_fix, param)
    Kh  = delFdelh_sig_erf(h_fix, s_fix, param) #validate!
    Ks  = delFdels_sig_erf(h_fix, s_fix, param) #validate!
    K1  = w^2*(1-p)/(tau*N*p)
    gam = [(1 - Kh*w)/tau (w*Ks)/tau; -(K1 * Kh)/tau (2-K1*Ks)/tau]
    V   = eigvecs(gam)
    W   = inv(V)
    Lam = eigvals(gam)

    Cov11 = (r0/N+D)/tau^2*( V[1,1]^2*W[1,1]^2/(2*Lam[1])   + 2*V[1,1]*V[1,2]*W[1,1]*W[2,1]/(Lam[1]+Lam[2]) + V[1,2]^2*W[2,1]^2/(2*Lam[2]) )
    Cov22 = (r0/N+D)/tau^2*( V[2,1]^2*W[1,1]^2/(2*Lam[1]) + 2*V[2,1]*V[2,2]*W[1,1]*W[2,1]/(Lam[1]+Lam[2]) + V[2,2]^2*W[2,1]^2/(2*Lam[2]) )
    Cov12 = (r0/N+D)/tau^2*( V[1,1]*V[2,1]*W[1,1]^2/(2*Lam[1]) + (W[1,1]*W[2,1])*(V[1,1]*V[2,2]+V[1,2]*V[2,1])/(Lam[1]+Lam[2]) + V[1,2]*V[2,1]*W[2,1]^2/(2*Lam[2]) )

    Var_r = Kh^2 * Cov11 + Ks^2 * Cov22 + 2*Kh*Ks*Cov12

    println([Cov11 Cov12; Cov12 Cov22])

    return [Cov11 Cov12; Cov12 Cov22], Var_r

end

function stability_map_sig_erf(w, mu, param; max_search_for_fixed_point = 1000)
    #=
        Assumes that there is either: monostable up/down or bistable or 1 vs. 3 fixed_points, w and mu are arrays
    =#
    len_mu      = length(mu)
    len_w       = length(w)    
    out_grid    = zeros((len_w, len_mu)) # mapping: 1: monostable-up, 2: bistable, 3: monostable-down, 4: bifurcation, 5: no_fixedpoint found
    naive_model = param["naive_model"]
    p           = param["p"]
    N           = param["N"]
    tau         = param["tau"]
    Phimax_2    = param["Phimax"]/2
    param_fixed_points = copy(param)

    for i in 1:len_w
        param_fixed_points["w"] = w[i]
        for j in 1:len_mu
            param_fixed_points["mu0"] = mu[j]
            if(w[i]>0)
                hmin = mu[j]
                hmax = abs(max_search_for_fixed_point)
            else
                hmin = -abs(max_search_for_fixed_point)
                hmax = mu[j]
            end
            fixed_points = calculate_fixed_point_sigmoidal_erf(param_fixed_points, hmin, hmax, naive_model = naive_model)
            if(length(fixed_points)==3)
                out_grid[i, j] = 2
            elseif(length(fixed_points)==1)
                fixed_point_val = fixed_points[1]
                if(naive_model)
                    s = 0
                else
                    s = w[i]*(1 - p)/(2*tau*N*p)*(fixed_point_val - mu[j])
                end
                if(F_sigm_erf(fixed_point_val, s, param) > Phimax_2)
                    out_grid[i, j] = 1
                else
                    out_grid[i, j] = 3
                end
            elseif(length(fixed_points)==2)
                    out_grid[i, j] = 4
            elseif(length(fixed_points)==0)
                    out_grid[i, j] = 5
            end
        end
    end
    return out_grid
end

function plot_fixed_point_map_sig_erf(w, mu, map_grid)
    for i in 1:length(w)
        for j = 1:length(mu)
            if(map_grid[i, j]==1)
                plot(w[i], mu[j], "o", color="red")
            elseif(map_grid[i, j]==2)
                plot(w[i], mu[j], "o", color="green")
            elseif(map_grid[i, j]==3)
                plot(w[i], mu[j], "o", color="blue")
            elseif(map_grid[i, j]==4)
                plot(w[i], mu[j], "o", color="black")
            elseif(map_grid[i, j]==5)
                plot(w[i], mu[j], "o", color="orange")
            end
        end
    end
    show()
end


function onset_of_oscillation_Deletememaybe(om, tau, d)
#=
    y1 = 3 .*sin.(om.*d).-om.*tau.*cos.(om.*d)
    y2 = 3 .*cos.(om.*d).+om.*tau.*sin.(om.*d)
    x1 = ((2 .-om.^2 .*tau.^2).*y1.+3 .*om.*tau.*y2)./(om.*tau.*sin.(om.*d).*y1.+om.*tau.*cos.(om.*d).*y2)
    x2 = (2 .-om.^2 .*tau.^2)./y2 .- om.*tau.*sin.(om.*d)./y2.*x1
    #err_out = -1im .*om.*tau .-1 .+x1.*exp.(-1im .*om.*d).+x1.*x2.*exp.((-2) .* (1im) .*om.*d)./(1im .*om.*tau.+2 .-x2.*exp.(-1im .*om.*d))
    #err_out = -om.^2 .*tau.^2 .-x2 .*om.*tau.*sin.(om.*d).-3 .* x2 .* cos.(om.*d) .+ 2 .- x1 .* sin.(om .*d) .*om .* tau
    #err_out = 3 .* om .* tau .-x2 .* om .* tau .* cos.(om .* d) .+ 3 .* x2 .* sin.(om .*d) .- x1 .* cos.(om .*d) .* om .* tau
    inter = (cos.(om .* d) - 1im .*sin.(om .* d))
    err_out = om.^2 .* tau.^2 .- 2 .* 1im .* om .*tau .+ x2 .* 1im .* om .* tau .* inter .- 1im .* om .* tau .- 2 .+ x2 .* inter .+ x1 .* inter .* 1im .* om .*tau .+ x2 .* inter .* 2 .- x1 .* x2 .* exp.(-1im .* om .*d) .* exp.(-1im.*om.*d) .+ x1 .* x2 .* exp.((-2) .* 1im .* om .* d) 
    #inter = (1im .* om .* tau .+ 2 .- x2 .* exp.(-1im .* om .* d))
    #err_out = -(1im .*om .* tau) .* inter .- inter .+x1 .*exp.(-1im.*om.*d) .* inter .+ x1 .*x2 .*exp.((-2) .* 1im .* om .* d)
=#
    y1 = cos.(om.*d) .+
    return x1, x2, err_out
end


function onset_of_oscillation_wrong(om, tau, d)
    """
    This program falsely assumes tau to be independent of x1, x2, this is just here for legacy
    """
    y1  = cos(om*d)+om*tau*sin(om*d)
    y2  = om*tau*cos(om*d)-sin(om*d)
    y11 = 2*cos(om*d)+om*tau*sin(om*d)
    y21 = 2*sin(om*d)-cos(om*d)*om*tau
    x1  = ((2-om^2*tau^2)*y2-3*om*tau*y1)/(y2*y11+y1*y21)
    x2  = (2-om^2*tau^2)/y1 - x1 * y11/y1
    err_out = abs(-1im*om*tau-1+x1*exp(-1im*om*d) + x1*x2*exp(-2*1im*om*d)/(1im*om*tau+2-x2*exp(-1im*om*d)))
    return x1, x2, err_out
end

function check_eigenvalue_calculation_linear_stability_sig_erf(nu, om, K1, Kh, Ks, param)
    lamb = nu + 1im * om
    w    = param["w"]
    tau  = param["tau"]
    d    = param["d"]
    return abs(-tau*lamb-1+w*Kh*exp(-lamb*d)+(w*Ks*Kh*K1*exp(-2*lamb*d))/(lamb*tau+2-K1*Ks*exp(-lamb*d)))
end


function calculate_eigenvalue_array_linear_stability_sig_erf(x0_arr, x1_arr, param; max_search_for_fixed_point =1000. , naive_model_yn =false)
    r0  = 0.
    h0  = 0
    s0  = 0
    #nu0 = 0.
    mu0 = param["mu0"]
    w   = param["w"]
    tau = param["tau"]
    d   = param["d"]
    p   = param["p"]
    N   = param["N"]
    C   = N*p
    h0_was_calculated = false
    # step 1: calculate h0, sigma_h0^2, for multiple fixed points take the value with the lowest firing rate
    if(w>0)
        hmin = mu0
        hmax = abs(max_search_for_fixed_point)
    else
        hmin = -abs(max_search_for_fixed_point)
        hmax = mu0
    end
    fixed_points = calculate_fixed_point_sigmoidal_erf(param, hmin, hmax, naive_model = naive_model_yn)
    len_fp = length(fixed_points)
    if(len_fp>0)
        list_eigen_val_nu = zeros(length(x0_arr) * length(x1_arr)) #return array of possible Eigenvalues, multiple values being equal is allowed
        list_eigen_val_om = zeros(length(list_eigen_val_nu))
        list_eigen_abserr = zeros(length(list_eigen_val_nu))

        # calculate the fixed point with the lowest firing rate
        for i=1:len_fp
           h_inter = fixed_points[i]
            if(naive_model_yn)
                s_inter = 0
            else
                s_inter = w * (1-p)/(2*tau*C)*(h_inter - mu0)
            end
            r_inter = F_sigm_erf(h_inter, s_inter, param) 
            if((i==1) || (r_inter <= r0))
                h0 = h_inter
                s0 = s_inter
                r0 = r_inter
                h0_was_calculated = true
            end 
        end
        if(!h0_was_calculated)
            print("warning: h0 could not be calculated!")
        end
        # go through all initial values for the calculation of the Eigenvalues and calculate Eigenvalues
        k  = 0 # counter for the final list, because I am leazy
        Kh = delFdelh_sig_erf(h0, s0, param)
        Ks = delFdels_sig_erf(h0, s0, param)
        K1 = w^2*(1-p)/(tau*C)
        for i = 1:length(x0_arr)
            for j=1:length(x1_arr)
                k+=1
                list_eigen_val_nu[k], list_eigen_val_om[k] = opt_py.solve_Eigenvalues_of_linearized_stability_sig_erf(w, 
                                                        tau, d, Kh, Ks, K1, x0_arr[i], x1_arr[j], naive_model=naive_model_yn)
                list_eigen_abserr[k] = check_eigenvalue_calculation_linear_stability_sig_erf(list_eigen_val_nu[k], 
                                                                            list_eigen_val_om[k], K1, Kh, Ks, param)
            end
        end
    else
        list_eigen_val_nu = []
        list_eigen_val_om = []
        list_eigen_abserr = []
    end
    return list_eigen_val_nu, list_eigen_val_om, list_eigen_abserr
end

function onset_of_oscillation_map_sig_erf_w_d(d_arr, w_arr, x0_arr, x1_arr, param;
                                            abs_tol=1e-10, max_search_for_fixed_point=1000, naive_model_yn = false) # to be completed!!! naive model not included yet
    nu_map = zeros((length(d_arr), length(w_arr)))
    om_map = zeros((length(d_arr), length(w_arr)))
    for (i, d_i) = enumerate(d_arr)
        param["d"] = d_i
        for (j, w_j) = enumerate(w_arr)
            param["w"] = w_j
            inter_list_nu, inter_list_om, inter_list_err = calculate_eigenvalue_array_linear_stability_sig_erf(
                                                                x0_arr, x1_arr, param, max_search_for_fixed_point =max_search_for_fixed_point,
                                                                naive_model_yn =naive_model_yn)
            ind = inter_list_err .< abs_tol
            if(length(inter_list_nu[ind])>0)
                index_of_max = argmax(inter_list_err[ind])
                nu_map[i, j] = inter_list_nu[ind][index_of_max]
                om_map[i, j] = abs(inter_list_om[ind][index_of_max])    # per convention I just note the positive value from the complex pair
            else
                nu_map[i, j] = NaN
                om_map[i, j] = NaN
            end
        end
    end
    return nu_map, om_map
end

function onset_of_oscillation_phase_amplitude_sig_erf_ver_1(param; naive_model_yn =false, kmax=10, max_search_for_fixed_point=1000)
    r0  = 0.
    h0  = 0
    s0  = 0
    #nu0 = 0.
    mu0 = param["mu0"]
    w   = param["w"]
    tau = param["tau"]
    d   = param["d"]
    p   = param["p"]
    N   = param["N"]
    C   = N*p
    h0_was_calculated = false
    omega_list = zeros(kmax)
    # step 1: calculate h0, sigma_h0^2, for multiple fixed points take the value with the lowest firing rate
    if(w>0)
        hmin = mu0
        hmax = abs(max_search_for_fixed_point)
    else
        hmin = -abs(max_search_for_fixed_point)
        hmax = mu0
    end
    fixed_points = calculate_fixed_point_sigmoidal_erf(param, hmin, hmax, naive_model = naive_model_yn)
    len_fp = length(fixed_points)
    for i=1:len_fp
       h_inter = fixed_points[i]
        if(naive_model_yn)
            s_inter = 0
        else
            s_inter = w * (1-p)/(2*tau*C)*(h_inter - mu0)
        end
        r_inter = F_sigm_erf(h_inter, s_inter, param) 
        if((i==1) || (r_inter <= r0))
            h0 = h_inter
            s0 = s_inter
            r0 = r_inter
            h0_was_calculated = true
        end 
    end
    Kh = delFdelh_sig_erf(h0, s0, param)
    if(naive_model_yn)
        Ks=0
    else
        Ks = delFdels_sig_erf(h0, s0, param)
    end
    B = (1-p)/(tau*C)
    Fhtil = w*Kh
    Fstil = w^2*B*Ks
    #println((1-Fhtil)^2)

    if(naive_model_yn)
        for k=(1-1):(kmax-1)
            #println(k)
            f = x-> -atan(x*tau)+angle(Fhtil)-x*d+2*pi*k
            try
                omega_list[k+1] = find_zero(f, 0)
            catch e
                omega_list[k+1]= NaN
            end
        end
    else
        for k=(1-1):(kmax-1)
            f = x -> atan(x*tau)-atan(x*tau/2)-x*d-atan((x*tau+Fhtil*sin(x*d))/(1-Fhtil*cos(x*d)))+2*pi*k+angle(Fstil)
            try
                omega_list[k+1] = find_zero(f, 0)
            catch e
                omega_list[k+1] = NaN
            end
            #println(find_zeros(f,(-1000000, 1000000)))
        end
    end
    #return omega_list
    element_greater_1=0
    if(naive_model_yn)
        S = x-> Fhtil^2-x^2*tau^2   
    else
        S = x-> (1+x^2*tau^2)*Fstil^2/((1-Fhtil*cos(x*d))^2 + (x*tau+Fhtil*sin(x*d))^2)*1/(4+x^2*tau^2)
    end
    for (k, om) =enumerate(omega_list)
        if( S(om)>= 1 )
            element_greater_1 = 1
        end
    end
    return element_greater_1
end

function onset_of_oscillation_phase_amplitude_sig_erf_ver_2(param; naive_model_yn =false, kmax=10, max_search_for_fixed_point=1000, version=2)
    r0  = 0.
    h0  = 0
    s0  = 0
    #nu0 = 0.
    mu0 = param["mu0"]
    w   = param["w"]
    tau = param["tau"]
    d   = param["d"]
    p   = param["p"]
    N   = param["N"]
    C   = N*p
    h0_was_calculated = false
    omega_list = zeros(kmax)
    # step 1: calculate h0, sigma_h0^2, for multiple fixed points take the value with the lowest firing rate
    if(w>0)
        hmin = mu0
        hmax = abs(max_search_for_fixed_point)
    else
        hmin = -abs(max_search_for_fixed_point)
        hmax = mu0
    end
    fixed_points = calculate_fixed_point_sigmoidal_erf(param, hmin, hmax, naive_model = naive_model_yn)
    len_fp = length(fixed_points)
    for i=1:len_fp
       h_inter = fixed_points[i]
        if(naive_model_yn)
            s_inter = 0
        else
            s_inter = w * (1-p)/(2*tau*C)*(h_inter - mu0)
        end
        r_inter = F_sigm_erf(h_inter, s_inter, param) 
        if((i==1) || (r_inter <= r0))
            h0 = h_inter
            s0 = s_inter
            r0 = r_inter
            h0_was_calculated = true
        end 
    end
    Kh = delFdelh_sig_erf(h0, s0, param)
    if(naive_model_yn)
        Ks=0
    else
        Ks = delFdels_sig_erf(h0, s0, param)
    end
    B = (1-p)/(tau*C)
    Fhtil = w*Kh
    Fstil = w^2*B*Ks
    #println((1-Fhtil)^2)

    if(naive_model_yn)
        for k=(1-1):(kmax-1)
            #println(k)
            f = x-> -atan(x*tau)+angle(Fhtil)-x*d+2*pi*k
            try
                omega_list[k+1] = find_zero(f, 0)
            catch e
                omega_list[k+1]= NaN
                println("here")
            end
        end
    else
        for k=(1-1):(kmax-1)
            #f = x -> atan(x*tau)-atan(x*tau/2)-x*d-atan((x*tau+Fhtil*sin(x*d))/(1-Fhtil*cos(x*d)))+2*pi*k+angle(Fstil)
            #f = x-> -atan(x*tau)+atan(x*tau/2)+angle(Fhtil)-x*d-atan((x*tau+Fstil*sin(x*d))/(2-Fstil*cos(x*d)))+2*pi*k
            if(version==2)
                f = x-> -atan(x*tau)+atan(x*tau/2)+angle(Fhtil)-x*d-atan(x*tau+Fstil*sin(x*d), 2-Fstil*cos(x*d)) +2*pi*k
            elseif(version==1)
                f = x -> atan(x*tau)-atan(x*tau/2)-x*d-atan((x*tau+Fhtil*sin(x*d)), (1-Fhtil*cos(x*d)))+2*pi*k+angle(Fstil)
            end
            try
                omega_list[k+1] = find_zero(f, 0)
            catch e
                omega_list[k+1] = NaN
            end
            #println(find_zeros(f,(-1000000, 1000000)))
        end
    end
    #return omega_list
    element_greater_1=0
    if(naive_model_yn)
        S = x-> Fhtil^2-x^2*tau^2   
    else
        if(version==1)
            S = x-> (1+x^2*tau^2)*Fstil^2/((1-Fhtil*cos(x*d))^2 + (x*tau+Fhtil*sin(x*d))^2)*1/(4+x^2*tau^2)
        elseif(version==2)
            S = x-> (4+x^2*tau^2)/(1+x^2*tau^2)*Fhtil^2/((2-cos(x*d)*Fstil)^2+(x*d+Fstil*sin(x*d))^2)
        end
    end
    for (k, om) =enumerate(omega_list)
        if( S(om)>= 1 )
            element_greater_1 = 1
        end
    end
    return element_greater_1
end

function onset_of_oscillation_map_sig_erf_w_d(d_arr, w_arr, param; max_search_for_fixed_point=1000, naive_model_yn = false, version=2,
                                                    by_simulation = false, supress_noise_yn=true)
    var_condition = (-1) .* ones((length(d_arr), length(w_arr)))
    mean_condition= (-1) .* ones((length(d_arr), length(w_arr)))
    progress_counter = 0
    max_progress = length(d_arr) * length(w_arr)
    for (i, d_i) = enumerate(d_arr)
        param["d"]=d_i
        param["delta"] = d_i
        for (j, w_j) = enumerate(w_arr)
            param["w"]=w_j
            #println(w_j, "\t", d_i)
            progress_counter+=1
            println(progress_counter, "\t of ", max_progress)
            if(!by_simulation)
                var_condition[i, j] = onset_of_oscillation_phase_amplitude_sig_erf_ver_2(param; naive_model_yn =naive_model_yn,
                                                                    kmax=kmax, max_search_for_fixed_point=max_search_for_fixed_point, version=version)# This I might want to update
            else
                n_relax = round(Int, param["T_relax"]/param["dt"])
                if(naive_model_yn)
                    h0 = 0.1
                    s0 = 0.
                    hn, sn, An, rn = Euler_integrator_general_transf_full_int_delay(h0, s0, f_h_general_non_fully_connected_delay,
                                        f_s_general_naive_delay, F_sigm_erf, param, supress_noise_yn=supress_noise_yn)
                    var_condition[i, j] = var(rn[n_relax:end])
                    mean_condition[i, j]= mean(rn[n_relax:end])
                else
                    h0 = 0.1
                    s0 = 0.1
                    h, s, A, r = Euler_integrator_general_transf_full_int_delay(h0, s0, f_h_general_non_fully_connected_delay,
                                        f_s_general_non_fully_connected_delay, F_sigm_erf, param, supress_noise_yn=supress_noise_yn)
                    var_condition[i, j] = var(r[n_relax:end])
                    mean_condition[i, j]= mean(r[n_relax:end])
                end
            end
        end
    end
    return var_condition, mean_condition
end


function onset_of_oscillation_map_sig_erf_w_d_also_micro(d_arr, w_arr, param, param_phi; max_search_for_fixed_point=1000, meso_micro_naive_model = "micro",
                                                    by_simulation=false, supress_noise_yn=false)
    # has key code structure .... IMPLEMENT!!!! This is not super important as long as you do not use param afterwards
    var_condition = (-1) .* ones((length(d_arr), length(w_arr)))
    mean_condition= (-1) .* ones((length(d_arr), length(w_arr)))
    progress_counter = 0
    max_progress = length(d_arr) * length(w_arr)
    for (i, d_i) = enumerate(d_arr)
        param["d"]=d_i
        param["delta"] = d_i
        param["delay"] = d_i
        for (j, w_j) = enumerate(w_arr)
            param["w"]=w_j
            #println(w_j, "\t", d_i)
            progress_counter+=1
            println(progress_counter, "\t of ", max_progress)
            println()
            if(!by_simulation)
                #var_condition[i, j] = onset_of_oscillation_phase_amplitude_sig_erf_ver_2(param; naive_model_yn =naive_model_yn,
                #                                                    kmax=kmax, max_search_for_fixed_point=max_search_for_fixed_point, version=version)# This I might want to update
            else
                n_relax = round(Int, param["T_relax"]/param["dt"])
                if(meso_micro_naive_model=="naive")
                    h0 = 0.1
                    s0 = 0.
                    hn, sn, An, rn = Euler_integrator_general_transf_full_int_delay(h0, s0, f_h_general_non_fully_connected_delay,
                                        f_s_general_naive_delay, F_sigm_erf, param, supress_noise_yn=supress_noise_yn)
                    var_condition[i, j] = var(rn[n_relax:end])
                    mean_condition[i, j]= mean(rn[n_relax:end])
                elseif(meso_micro_naive_model=="meso")
                    h0 = 0.1
                    s0 = 0.1
                    h, s, A, r = Euler_integrator_general_transf_full_int_delay(h0, s0, f_h_general_non_fully_connected_delay,
                                        f_s_general_non_fully_connected_delay, F_sigm_erf, param, supress_noise_yn=supress_noise_yn)
                    var_condition[i, j] = var(r[n_relax:end])
                    mean_condition[i, j]= mean(r[n_relax:end])
                else
                    param["con_ii"] = w_j
                    _,_,_,_,rI,_,_,_,_,_= sim_lnp(param["tmax"], param["dt"], 0, param["Ni"], 1, param["dt_record"], Phi_sigm_erf, param_phi, param, 0, ini_v = param["ini_h_dis"])
                    var_condition[i, j] =var(rI[n_relax:end])
                    mean_condition[i, j]=mean(rI[n_relax:end])
                end
            end
        end
    end
    return var_condition, mean_condition
end


function onset_of_oscillation_check_condition(param, Fhtil, Fstil, om_n)
    # calculate possible values of omega
    tau     = param["tau"]
    d       = param["d"]
    om_max  = param["omega_max_search"]
    rel_tol = param["rel_tol"]

    f_om = x-> (4+x^2*tau^2)/(1+x^2*tau^2) * Fhtil^2/((2-Fstil*cos(x*d))^2 + (x*tau+Fstil*sin(x*d))^2) -1
    om = zeros(length(om_n))
    if(false) #work in progress
    for j=1:length(om_n)
        try
            om[j] = find_zero(f_om, om_n[j], atol=1e-1)
        catch
            om[j] = NaN
        end
    end
    end
    if(true)
    for j=1:length(om_n)
        if(f_om(om_n[j]) >= 1)
            om[j] = om_n[j]
        else
            om[j] = NaN
        end
    end 
    end
    om = om[.!isnan.(om)]
    #om = find_zeros(f_om, (0, om_max), rtol=1e-1)
    len_om = length(om)
    out_condition = -1

    if(len_om == 0)
        out_condition = 0
    else
        # go through list of possible om: Can we fullfill phase condition?
        for i=1:len_om
            if(length(om[i]) > 1)
                println("here")
                println(om[i])
            end
            kval = ( atan(om[i]*tau)-atan(om[i]*tau/2)+om[i]*d-angle(Fhtil)+atan(om[i]*tau+Fstil*sin(om[i]*d), 2-Fstil*cos(om[i]*d)) )/(2*pi)
            distance_to_int = abs(round(Int, kval)-kval)
            if(distance_to_int <= rel_tol)
                out_condition = 1
                break
            else
                out_condition = 2 # change later to 0
            end
        end
    end
    return out_condition
end

function onset_of_oscillation_get_one_fix_point(param, naive_model_yn)
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
    fp = calculate_fixed_point_sigmoidal_erf(param, hmin, hmax, naive_model = naive_model_yn)
    
    # there can be 1 or 3 fixed points: take the fixed point with the smallest firing rate
    len_fp = length(fp)
    if(len_fp == 0)
        return 0, 0, fix_found
    end
    fix_found = true
    for i=1:len_fp
        hi = fp[i]
        if(naive_model_yn)
            si = 0
        else
            si = w * (1-p)/(2*tau*C)*(hi - mu0)
        end
        ri = F_sigm_erf(hi, si, param)
        #if((i==1)|| (ri <= r0))
         if((i==1) || (ri <= rfix))
            hfix = hi
            sfix = si
            rfix = ri
        end
    end
    return hfix, sfix, fix_found
end

function onset_of_oscillation_map_sig_erf_w_d_semianalytic(d_arr, w_arr, param, om_n; naive_model_yn=false)
    map_condition    = (-1) .* ones((length(d_arr), length(w_arr)))
    progress_counter = 0
    max_progress     = length(d_arr) * length(w_arr)
    B                = (1-param["p"])/(param["tau"]*param["C"])
    fix_found        = false

    for (i, d_i) = enumerate(d_arr)
        param["d"]=d_i
        param["delta"] = d_i
        for (j, w_j) = enumerate(w_arr)
            param["w"]=w_j
            progress_counter+=1
            println(progress_counter, "\t of ", max_progress)

            # calculate the fixed point of the system, if there are multiple
            hfix, sfix, fix_found = onset_of_oscillation_get_one_fix_point(param, naive_model_yn)

            if(fix_found)
                Kh = delFdelh_sig_erf(hfix, sfix, param)
                if(naive_model_yn)
                    Ks = 0
                else
                    Ks = delFdels_sig_erf(hfix, sfix, param)
                end
                Fhtil = w_j*Kh
                Fstil = w_j^2*B*Ks
                #println(Fhtil, "\t", Fstil)

                map_condition[i, j] = onset_of_oscillation_check_condition(param, Fhtil, Fstil, om_n)
            else
                map_condition[i, j] = -2 # maybe change value
            end
        end
    end
    return map_condition
end

function onset_of_oscillation_naive_get_d_and_omega(param, Fhtil, k; experimental_om_tau_yn=false)
"""
...
"""
    om  = 0
    tau = param["tau"]
    d   = 0
    if(Fhtil^2 > 1)
        om = 1/tau * sqrt(Fhtil^2 - 1) 
    else
        om = NaN
    end
    if(om > 1e-15)
        if(experimental_om_tau_yn)
            d = 1/om * (-pi/2+angle(Fhtil) +2*pi*k)
        else
            d = 1/om * (-atan(om * tau) + angle(Fhtil) + 2*pi*k)
        end
    else
        d = NaN
    end
    return om, d
end


function onset_of_oscillation_special_case_naive(w_arr, param, k; experimental_mode=0, experimental_p=0, experimental_om_tau_yn=false)
"""
in the naive case I can directly get d_arr = func(w_arr)    
"""
    # out array
    d_arr       = (-1) .* ones(length(w_arr))
    om_arr      = zeros(length(w_arr))
    fix_found   = false
    B           = (1-param["p"])/(param["tau"]*param["C"])

    # go through all w in w_arr
    for (i, w_i) = enumerate(w_arr)
        param["w"] = w_i

        # give fixed point with smallest firing rate
        if(experimental_mode==0)
            hfix, sfix, fix_found = onset_of_oscillation_get_one_fix_point(param, true)
        else
            p = param["p"]
            param["p"]=experimental_p
            hfix, sfix, fix_found = onset_of_oscillation_get_one_fix_point(param, false)
            param["p"]=p
        end

        if(fix_found)
            Kh = delFdelh_sig_erf(hfix, sfix, param)
            Ks = 0
            Fhtil = w_i*Kh
            Fstil = 0
            #println(Fhtil, "\t", Fstil)
            om_arr[i], d_arr[i] = onset_of_oscillation_naive_get_d_and_omega(param, Fhtil, k, experimental_om_tau_yn=experimental_om_tau_yn)           
        else
            om_arr[i], d_arr[i] = NaN, NaN
        end
    end
    d_arr       = d_arr[.!isnan.(om_arr)]
    w_arr_out   = w_arr[.!isnan.(om_arr)]
    om_arr      = om_arr[.!isnan.(om_arr)]
    return om_arr, d_arr, w_arr_out
end

function onset_of_oscillation_calculate_S_omega_phase_cond(Fhtil, Fstil, k, param)
    tau = param["tau"]
    d   = param["delay"]
    phase_condition_fullfilled = true
    om = 0
    # note how the atan(a,b) is defined!
    f_phase = x -> -atan(x*tau)+atan(x*tau/2)+angle(Fhtil)-x*d-atan(x*tau+Fstil*sin(x*d), 2-Fstil*cos(x*d)) +2*pi*k

    try
         om = find_zero(f_phase, 500)   
        #om = find_zero(f_phase, 0.1)
    catch
        om = NaN#-1#0
        phase_condition_fullfilled = false
        #println(k)
    end

    # control: how large is the error the root finder does?
    phase_err = abs(f_phase(om))

    if(phase_condition_fullfilled)
        #S_om = (4+om^2*tau^2)/(1+om^2*tau^2) * Fhtil^2/((2-cos(om*tau)*Fstil)^2+(om*tau+Fstil*sin(om*tau))^2)
        S_om = (4+om^2*tau^2)/(1+om^2*tau^2) * Fhtil^2/((2-cos(om*d)*Fstil)^2+(om*tau+Fstil*sin(om*d))^2)
    else
        S_om = NaN
    end
    return S_om, om, phase_condition_fullfilled, phase_err
end


function onset_of_oscillation_d_slices(d_arr, w_arr, param, k, naive_model_yn)
    map_condition    = (-1) .* ones(length(w_arr), length(d_arr))
    map_S            = (-1) .* ones(length(w_arr), length(d_arr))
    map_phase        = (-1) .* ones(length(w_arr), length(d_arr))
    map_phase_err    = (-1) .* ones(length(w_arr), length(d_arr))
    map_om           = (-1) .* ones(length(w_arr), length(d_arr))
    map_r0           = (-1) .* ones(length(w_arr), length(d_arr))
    progress_counter = 0
    max_progress     = length(d_arr) * length(w_arr)
    fix_found        = false
    fix_found_old    = false
    B                = (1-param["p"])/(param["tau"]*param["C"])
    w_list           = []
    d_list           = []
    om_list          = []

    S_om_old         = 0
    phase_con_old    = false

    # go through all w in w_arr
    for (j, d_j) = enumerate(d_arr)
        param["delay"] = d_j
        param["delta"] = d_j
        for (i, w_i) = enumerate(w_arr)
            param["w"] = w_i

            # give fixed point with smallest firing rate
            hfix, sfix, fix_found = onset_of_oscillation_get_one_fix_point(param, naive_model_yn)
            if(fix_found)
                map_r0[i, j] = F_sigm_erf(hfix, sfix, param)
                Kh = delFdelh_sig_erf(hfix, sfix, param)
                if(naive_model_yn)
                    Ks = 0
                else
                    Ks = delFdels_sig_erf(hfix, sfix, param)
                end
                Fhtil = w_i*Kh
                Fstil = w_i^2*B*Ks
                #println(Fhtil, "\t", Fstil)

                S_om, om_k, phase_condition_fullfilled, phase_err = onset_of_oscillation_calculate_S_omega_phase_cond(Fhtil, Fstil, k, param)
                map_S[i, j] = S_om
                map_phase[i, j] = Int(phase_condition_fullfilled)
                map_phase_err[i, j] = phase_err
                map_om[i, j] = om_k

                if((S_om > 1) && (phase_condition_fullfilled))
                    map_condition[i, j] = 1
                else
                    map_condition[i, j] = 0
                end

                if(!phase_condition_fullfilled)
                    map_condition[i, j]=-3
                end

                if( (i > 1) && ((S_om - 1) * (S_om_old -1) <= 0) && (phase_condition_fullfilled) ) # not finished!
                    append!(w_list, w_i)
                    append!(d_list, d_j)
                    append!(om_list, om_k)
                else
                    #map_condition[i, j] = 0
                end
            else
                map_condition[i, j] = -2 # maybe change value
            end
            S_om_old = S_om
            phase_con_old = phase_condition_fullfilled
        end
    end
    return map_condition, w_list, d_list, om_list, map_S, map_phase, map_om, map_phase_err, map_r0
end


function onset_of_oscillation_vary_any_parameter(a_arr, b_arr, a_string, b_sring, param, k, naive_model_yn)
#=
same as above, but now I can vary any paramter I like
=#
    map_condition    = (-1) .* ones(length(a_arr), length(b_arr))
    map_S            = (-1) .* ones(length(a_arr), length(b_arr))
    map_phase        = (-1) .* ones(length(a_arr), length(b_arr))
    map_phase_err    = (-1) .* ones(length(a_arr), length(b_arr))
    map_om           = (-1) .* ones(length(a_arr), length(b_arr))
    map_r0           = (-1) .* ones(length(a_arr), length(b_arr))
    map_Fhtil        = (-1) .* ones(length(a_arr), length(b_arr))
    map_Fstil        = (-1) .* ones(length(a_arr), length(b_arr))
    progress_counter = 0
    max_progress     = length(b_arr) * length(a_arr)
    fix_found        = false
    fix_found_old    = false
    a_list           = []
    b_list           = []
    om_list          = []

    S_om_old         = 0
    S_om             = 0
    phase_con_old    = false

    # go through all parameters in a and b
    for (j, b_j) = enumerate(b_arr)
        param[b_string] = b_j
        for (i, a_i) = enumerate(a_arr)
            param[a_string] = a_i
            param["delta"]  = param["delay"] # makes sure that both synonyms are covered
            param["d"]      = param["delay"]
            B               = (1-param["p"])/(param["tau"]*param["C"])
            

            # give fixed point with smallest firing rate
            hfix, sfix, fix_found = onset_of_oscillation_get_one_fix_point(param, naive_model_yn)
            if(fix_found)
                map_r0[i, j] = F_sigm_erf(hfix, sfix, param)
                Kh = delFdelh_sig_erf(hfix, sfix, param)
                if(naive_model_yn)
                    Ks = 0
                else
                    Ks = delFdels_sig_erf(hfix, sfix, param)
                end
                Fhtil = param["w"]*Kh
                Fstil = (param["w"])^2*B*Ks
                map_Fhtil[i, j] = Fhtil
                map_Fstil[i, j] = Fstil
                #println(Fhtil, "\t", Fstil)

                S_om, om_k, phase_condition_fullfilled, phase_err = onset_of_oscillation_calculate_S_omega_phase_cond(Fhtil, Fstil, k, param)
                map_S[i, j] = S_om
                map_phase[i, j] = Int(phase_condition_fullfilled)
                map_phase_err[i, j] = phase_err
                map_om[i, j] = om_k

                if((S_om > 1) && (phase_condition_fullfilled))
                    map_condition[i, j] = 1
                else
                    map_condition[i, j] = 0
                end

                if(!phase_condition_fullfilled)
                    map_condition[i, j]=-3
                end

                if( (i > 1) && ((S_om - 1) * (S_om_old -1) <= 0) && (phase_condition_fullfilled) ) # not finished!
                    append!(a_list, a_i)
                    append!(b_list, b_j)
                    append!(om_list, om_k)
                else
                    #map_condition[i, j] = 0
                end
            else
                map_Fhtil[i, j] = NaN
                map_condition[i, j] = -2 # maybe change value
                S_om = S_om_old
                phase_condition_fullfilled = phase_con_old
            end
            S_om_old = S_om
            phase_con_old = phase_condition_fullfilled
        end
    end
    return map_condition, a_list, b_list, om_list, map_S, map_phase, map_om, map_phase_err, map_r0, map_Fhtil, map_Fstil
end


function newton_calculate_initial_guess(d, Fhtil, Fstil, param)
#=
calculate inital guess for newton method for the calculation of eigenvalues for the linear 
stability analysis for the onset of oscillation with delay, for d = 0 analytical value,
for $d\neq 0$ simply use initial guess = 0
=#
    if(d<1e-10)
        A = (Fhtil-1)/2
        B = sqrt(complex(A^2 + 2 + Fhtil - Fhtil*Fstil))
        return [1/param["tau"] * (A + B), 1/param["tau"] * (A - B)]
    else
        return [0, 0]
    end
end

function only_positive_Im_part(x)
    y = copy(x)                 # why do I need to use copy here
    for i=1:length(y)
        if(imag(y[i])< 0)
            y[i] = conj(y[i])
        end
    end
    return y
end

function reduce_eigenvalue_cloud_to_n_single_points(eigenlist_unsort, dist_unique, n)
#=
give list of eigenvalues, most of them are the same, give me the first n unique eigenvalues, aka 
those with a distance larger than dist_unique to the nearest (nearest determined only by real part)
=#
    out = []

    # only consider positive Imaginary values
    eigenliste = only_positive_Im_part(eigenlist_unsort)

    # sort the list via the real part
    idx = sortperm(real.(eigenliste), rev=true)

    # the value with the larges Real part is in the output
    out = append!(out, eigenliste[idx][1])    
    #out[1] = eigenlist[idx][1]
    counter_n = 1

    diff_list = abs.(diff(eigenliste[idx]))

    #print(diff_list)
    
    # go through all values of the sorted eigenvalues list, only take those with a large enough distance
    for i=1:(length(eigenliste)-1)

        # we only take the first distinct n eigenvalues, if there are so many       
        if(counter_n == n)
            break
        end

        # now compare if the difference is large enough
        if(diff_list[i] > dist_unique)
            counter_n += 1
            #out[counter_n] = eigenlist[idx][i+1]
            out = append!(out, eigenliste[idx][i+1])
        end
    end
    return out
end

function test_write(x)
    a = x
    for i = 1:length(x)
        a[i] = 1
    end
    return a
end

function construct_grid_for_init_guess(init_guess, param)
    dRe = param["dRe"]
    dIm = param["dIm"]
    Re_plusmax = param["Re_plusmax"]
    Im_plusmax = param["Im_plusmax"]

    max_point_Re = maximum(real.(init_guess))
    min_point_Re = minimum(real.(init_guess))
    max_point_Im = maximum(imag.(init_guess))
    min_point_Im = minimum(imag.(init_guess))

    return (min_point_Re-Re_plusmax):dRe:(max_point_Re+Re_plusmax+dRe), (min_point_Im-Im_plusmax):dIm:(max_point_Im+Im_plusmax+dIm)
end

function linear_stability_go_through_slices_of_fixed_w_with_newton(d_arr, naive_model_yn, param, n)
#=
calculate "pseudo-eigenvalues" = solution of characteristic equation for linear stability analysis.
Do so for w fixed. Then go through several delays d,
observe the behaviour of n eigenvalues (with largest Real part) or less if there are less
=#
    # some often used parameters
    w = param["w"]
    tau = param["tau"]

    #empty array for every d:
    eigenlist = [[] for i=1:length(d_arr)]

    # calculation of fixed points, if multiple fixed points are given we use the smallest one
    # simply because we use w < 0 must of the time ==> only one fix point anyway
    hfix, sfix, fix_found = onset_of_oscillation_get_one_fix_point(param, naive_model_yn)

    if(fix_found)
        # calculate some abbreviations
        B = (1-param["p"])/(param["tau"]*param["C"])
        Kh = delFdelh_sig_erf(hfix, sfix, param)
        if(naive_model_yn)
            Ks = 0
        else
            Ks = delFdels_sig_erf(hfix, sfix, param)
        end
        Fhtil = w*Kh
        Fstil = w^2*B*Ks

        # calculate the initial guess, if \(d\neq 0\) use \(d=0\)
        init_guess = newton_calculate_initial_guess(d_arr[1], Fhtil, Fstil, param)
        init_guess = reduce_eigenvalue_cloud_to_n_single_points(init_guess, param["dist_unique"], n)
        eigenlist[1] = init_guess

        for i=2:length(d_arr)

            # construct Re_grid, Im_grid for the initial guess
            Re_grid, Im_grid = construct_grid_for_init_guess(eigenlist[i-1], param)
            
            # calculate new eigenlist
            eigenlist[i] = reduce_eigenvalue_cloud_to_n_single_points(
                    opt_py.newton_multiple_initial_for_linear_stability_sig_erf(Re_grid, Im_grid, Fhtil, Fstil, d_arr[i], tau),
                        param["dist_unique"], n)
        end
    else
        # if no fixed point is found a stability ananlysis makes no sense, return empty array
        return eigenlist, fix_found
    end
    return eigenlist, fix_found
end

function onset_of_oscillation_meso_approx(w_arr, param; experimental_mode=[0,0,0,0])
# using Polynomials for an approximate d = func(w)
    d_arr   = -1 * ones((length(w_arr), 4))
    d_arr_1 = -1 * ones(length(w_arr))
    B       = (1-param["p"])/(param["tau"]*param["C"])
    len_rts = -1 * ones(length(w_arr))

    for i =1:length(w_arr)
        param["w"] = w_arr[i]
        
            hfix, sfix, fix_found = onset_of_oscillation_get_one_fix_point(param, false)
        if(experimental_mode[1]==1)
            p = param["p"]
            param["p"] = 1
            hfix_exp, sfix_exp, fix_found_exp = onset_of_oscillation_get_one_fix_point(param, true)
            param["p"] = p
        end
        if(fix_found)
            if(experimental_mode[2]==0)
                Kh  = delFdelh_sig_erf(hfix, sfix, param)
            else
                p = param["p"]
                param["p"] = 1
                Kh  = delFdelh_sig_erf(hfix_exp, 0, param)
                param["p"] = p  
            end
            if(experimental_mode[3]==0)
                Ks  = delFdels_sig_erf(hfix, sfix, param)
            else
                Ks  = 0
            end
            Fhtil   = param["w"]*Kh
            Fstil   = (param["w"])^2*B*Ks
            if(experimental_mode[4]==0)
                pol     = Polynomials.Polynomial([4*(1-Fhtil^2)+Fstil^2, 2*Fstil, 5+Fstil^2-Fhtil^2, 2*Fstil, 1]);
            else
                pol     = Polynomials.Polynomial([1, 2*Fstil, 5+Fstil^2-Fhtil^2]);
            end
            rts     = Polynomials.roots(pol)
            cond    = ((abs.(imag.(rts)) .< 1e-10) .& (real.(rts) .>= 0 ) )
            len_rts[i] = sum(cond)
            if(len_rts[i] == 1)
                d_arr_1[i] = rts[cond][1]
            else
                d_arr_1[i] = NaN
            end
        end
    end
    return pi*param["tau"] ./(2 .* d_arr_1)
end

function cont_average(new_value, old_average, counter_trials)
    return  (new_value .+ (counter_trials .- 1) .* old_average)./counter_trials
end

function cont_variance(new_value, old_average, old_variance, new_average, counter_trials)
    return (abs.(new_value).^2 .+ (counter_trials .- 1) .* (old_variance .+ abs.(old_average).^2) )./counter_trials .- abs.(new_average).^2
end

function recursive_mean_var(new_value, old_mean, old_var, counter_trials)
    new_mean    = cont_average(new_value, old_mean, counter_trials)
    new_var     = cont_variance(new_value, old_mean, old_var, new_mean, counter_trials)
    return new_mean, new_var
end

function linearized_variance(param, naive_model_yn)
    #=
    Linearize the Langevin equation, the covariance matrix can be calculated as the solution of a Lyapunov equation, here the solution is directly written
    =#
    w   = param["w"]
    tau = param["tau"]
    N   = param["N"]
    C   = param["C"]
    p   = param["p"]
    hfix, sfix, r0, fix_found = PSD_get_one_fix_point(param, naive_model_yn)
    #println(fix_found) # comment this line out
    Fh  = delFdelh_sig_erf(hfix, sfix, param)
    if(!naive_model_yn)
        Fs = delFdels_sig_erf(hfix, sfix, param)
    else
        Fs = 0
    end
    beta    = w/tau
    alpha   = w^2/(tau^2*C)*(1-p)
    Delta   = 4*Fh^2*beta^2*tau^2+6*Fh*Fs*alpha*beta*tau^2-16*Fh*beta*tau+2*Fs^2*alpha^2*tau^2-10*Fs*alpha*tau+12
    shh     = -w^2*r0/N * (2*Fh*beta-Fs^2*alpha^2*tau+5*Fs*alpha-6/tau)/Delta
    shs     = -w^2*r0/N * (Fh*Fs*alpha^2*tau-2*Fh*alpha)/Delta
    sss     = w^2*r0/N * Fh^2*alpha^2*tau/Delta
    srr     = Fh^2*shh + Fs^2*sss + 2*Fh*Fs*shs

    return shh, shs, sss, srr   
end

function linearized_variance_check_alternative_form(param, naive_model_yn)
    #=
    Linearize the Langevin equation, the covariance matrix can be calculated as the solution of a Lyapunov equation, here the solution is directly written
    same as above, just in a slightly different form, I just want to confirm that this form is also correct
    =#
    w   = param["w"]
    tau = param["tau"]
    N   = param["N"]
    C   = param["C"]
    p   = param["p"]
    hfix, sfix, r0, fix_found = PSD_get_one_fix_point(param, naive_model_yn)
    #println(fix_found) # comment this line out
    Fh  = delFdelh_sig_erf(hfix, sfix, param)
    if(!naive_model_yn)
        Fs = delFdels_sig_erf(hfix, sfix, param)
    else
        Fs = 0
    end
    beta    = w/tau
    alpha   = w^2/(tau^2*C)*(1-p)
    Gam11   = beta*Fh-1/tau
    Gam12   = beta*Fs
    Gam21   = alpha*Fh
    Gam22   = alpha*Fs-2/tau
    Delta   = 4*Gam11*Gam22*(Gam11+Gam22)-4*Gam12*Gam21*Gam11-4*Gam21*Gam12*Gam22
    pref    = -w^2/tau^2*r0/N/Delta
    shh     = pref * (2*(Gam11+Gam22)*Gam22-2*Gam21*Gam12)
    shs     = -pref * (2*Gam22*Gam21)
    sss     = pref * 2 * Gam21^2
    srr     = Fh^2*shh + Fs^2*sss + 2*Fh*Fs*shs

    return shh, shs, sss, srr   
end

function Smm_0_function(om, param_mu)
    return 0
end

function linearized_PSD(f, param, Smm, param_mu, naive_model_yn)
    #=
    Linearize the Langevin equation, PSD can be calculated by the Fourier Transform NOT CORRECT YET
    =#
    w   = param["w"]
    tau = param["tau"]
    N   = param["N"]
    C   = param["C"]
    p   = param["p"]
    d   = param["delta"]
    hfix, sfix, r0, fix_found = PSD_get_one_fix_point(param, naive_model_yn, replacement_h=param["mu0"])
    Fh  = delFdelh_sig_erf(hfix, sfix, param)
    if(!naive_model_yn)
        Fs = delFdels_sig_erf(hfix, sfix, param)
    else
        Fs = 0
    end
    om      = 2*pi*f
    beta    = w/tau
    alpha   = w^2/(tau^2*C)*(1-p)
    exp_term= exp(-1im*om*d)

    determ  = -om^2 + 3*1im * om/tau + 2/tau^2 + exp_term * (-1im*om*(alpha*Fs+beta*Fh) -beta*Fh*2/tau -alpha*Fs/tau)
    D11     = (1im*om + 2/tau -alpha*Fs*exp_term)/determ
    #D21     = beta*Fh*exp_term/determ                      # ??? THIS NEEDS TO BE CHECKED!!!
    D21     = alpha*Fh*exp_term/determ
    #D22     = (1im*om + 1/tau -beta*Fh*exp_term)/determ    # is unncessary to calculate

    pre_fac = 1/tau^2 * (w^2*r0/N + Smm(om, param_mu))

    Shh    = pre_fac*(abs(D11))^2
    Shs     = pre_fac*D11*conj(D21)
    #Shs     = real(Shs_x)-1im*imag(Shs_x)
    Sss     = pre_fac*(abs(D21))^2
    # rest needs to be checked!!!
    Srr     = Fh^2*Shh+Fs^2*Sss+2*Fh*Fs*real(Shs)
    SAA     = Srr+r0/N+2*w*r0/N/tau*real(Fh*D11+Fs*D21)

    return SAA, Srr, Shh, Sss, Shs, 2*w*r0/N/tau*real(Fh*D11+Fs*D21), w/tau*sqrt(r0/N)*D11, w/tau*sqrt(r0/N)*D21#determ
end
        
function sim_lnp(T, dt, Ne, Ni, Nrecord, dt_rec, Phi, param_Phi, param_lnp, seed_quenched;fixed_in_degree=true, ini_v=0, record_mean_h_i=false)
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
    else
        rng = Random.MersenneTwister(seed_quenched)
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
    v = ini_v * ones(Ncells)
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

	    for ci = 1:Ncells
            xe[ci] = xe[ci] * Etaus[ci] + delaylineE[ci, ende] / dt * (1 - Etaus[ci])
            xi[ci] = xi[ci] * Etaus[ci] + delaylineI[ci, ende] / dt * (1 - Etaus[ci])
	        # xe[ci] += -dt*xe[ci]/taus + delaylineE[ci, ende]  #mV
	        # xi[ci] += -dt*xi[ci]/taus + delaylineI[ci, ende]
                
	        synInputE = xe[ci]
            synInputI = xi[ci]
            synInput = synInputE + synInputI
                
	        v[ci] += dtau[ci]*(mu[ci]-v[ci] + synInput)

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
    nbins   = 25
    hist    = fit(Histogram, v, nbins=nbins)
    hist    = normalize(hist, mode=:pdf)
    kurt    = calculate_emphirical_curtosis(v)
    skew    = calculate_emphirical_skewness(v)
    if(record_mean_h_i)        
        mean_h_i /=Nbin
    end

    @printf("\r")
    if(record_mean_h_i)
        return ns, v_record, spikes, rpopE_record, rpopI_record, hist, kurt, skew, actpopE_record, actpopI_record, mean_h_i
    end    
    return ns, v_record, spikes, rpopE_record, rpopI_record, hist, kurt, skew, actpopE_record, actpopI_record
end



function sim_lnp_annealed(T, dt, Ne, Ni, Nrecord, dt_rec, Phi, param_Phi, param_lnp, seed_quenched)
#=
Simulate microscopic model of spiking neurons, in general an IE-network, quenched noise is replaced with a drawing of neural
connections for every spike
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
    else
        rng = Random.MersenneTwister(seed_quenched)
    end
#    println("setting up parameters")
    #Ne = 0
    #Ni = 1000
    Ncells  = Ne + Ni
    #alpha=2.
    
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
    
    delay = param_lnp["delay"]
    #n_delay = round(Int, delay/dt)
    n_delay = ceil(Int, delay/dt)
        
    Nbin = round(Int, dt_rec/dt) #nin size for recording voltage and currents
    
    #membrane time constant
    tau = zeros(Ncells)
    dtau = zeros(Ncells)
    
    tau[1:Ne] .= taue
    tau[(1+Ne):Ncells] .= taui
    dtau[1:Ne] .= dt/taue
    dtau[(1+Ne):Ncells] .= dt/taui

    weights = zeros(Ncells,Ncells)
    
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

    Jmatrix                             = zeros(Ncells, Ncells)
    Jmatrix[1:Ne, 1:Ne]                 = jee .* ones(Ne, Ne)
    Jmatrix[1:Ne, 1+Ne:Ncells]          = jie .* ones(Ne, Ni)
    Jmatrix[1+Ne:Ncells, 1:Ne]          = jei .* ones(Ni, Ne)
    Jmatrix[1+Ne:Ncells, 1+Ne:Ncells]   = jii .* ones(Ni, Ni)
    
    ns = zeros(Int,Ncells)
    
    forwardInputsE = zeros(Ncells)
    forwardInputsI = zeros(Ncells)

    delaylineE = zeros(Ncells, n_delay)
    delaylineI = zeros(Ncells, n_delay)
    start = 1

    
    xe = zeros(Ncells)                                      # synaptic variable for excitatory input
    xi = zeros(Ncells)                                      # synaptic variable for inhibitory input

    v = 0 * ones(Ncells)                                    # voltage of neurons
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
            @printf("\r%d%% poprateE=%g poprateI=%g, abgelaufene Zeit=%g, rate=%g, Mu%g",round(Int,100*ti/Nsteps), poprateE, poprateI, ti*dt, sum(ns)/(Ni*ti*dt), mu0)
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
        
        mu = mu0                                            # just renaming

        poprateE = 0
        poprateI = 0

	    for ci = 1:Ncells
            xe[ci] = xe[ci] * Etaus[ci] + delaylineE[ci, ende] / dt * (1 - Etaus[ci])
            xi[ci] = xi[ci] * Etaus[ci] + delaylineI[ci, ende] / dt * (1 - Etaus[ci])
	        # xe[ci] += -dt*xe[ci]/taus + delaylineE[ci, ende]  #mV
	        # xi[ci] += -dt*xi[ci]/taus + delaylineI[ci, ende]
                
	        synInputE = xe[ci]
            synInputI = xi[ci]
            synInput = synInputE + synInputI
                
	        v[ci] += dtau[ci]*(mu-v[ci] + synInput)

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
                    w = 0.      
                    if(ci <= Ne)
                        if(rand() < p_e)
                            w = Jmatrix[ci, j]
                        end
                    else
                        if(rand() < p_i)
                            w = Jmatrix[ci, j]
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
    nbins   = 25
    hist    = fit(Histogram, v, nbins=nbins)
    hist    = normalize(hist, mode=:pdf)
    kurt    = calculate_emphirical_curtosis(v)
    skew    = calculate_emphirical_skewness(v)

    @printf("\r")

    
    return ns, v_record, spikes, rpopE_record, rpopI_record, hist, kurt, skew, actpopE_record, actpopI_record
end
