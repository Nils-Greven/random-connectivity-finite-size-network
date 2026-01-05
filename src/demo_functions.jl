function PSD_get_one_fix_point(param, naive_model_yn; replacement_h =0, replacement_s =0, replacement_r = 0)
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

    if(!haskey(param, "sparse_limit_yn"))
        param["sparse_limit_yn"] = false
    end

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
        return replacement_h, replacement_s, replacement_r, fix_found
    end
    fix_found = true
    for i=1:len_fp
        hi = fp[i]
        if(naive_model_yn)
            si = 0
        else
            if(param["sparse_limit_yn"])
                si = w/(2*tau*C)*(hi-mu0)
            else
                si = w * (1-p)/(2*tau*C)*(hi - mu0)
            end
        end
        ri = F_sigm_erf(hi, si, param)
        #if((i==1) || (ri <= rfix))
        if((i==1) || (hi <= hfix))
            hfix = hi
            sfix = si
            rfix = ri
        end
    end
    return hfix, sfix, rfix, fix_found
end
