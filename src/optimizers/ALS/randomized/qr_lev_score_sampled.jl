function compute_als(
    alg::QRPivProjected,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    shuffle_pivots = true,
    trunc_tol = 0.01,
    normal = true,
    kwargs...
)
    pivots = Vector{Vector{Int}}()
    ref_pivs = Vector{Vector{Int}}()
    projectors = Vector{ITensor}()
    targets = Vector{ITensor}()
    qr_factors = Vector{AbstractArray}()
    effective_ranks = Vector{Int}()
    piv_id = nothing
    for (i, n) in zip(inds(target), 1:length(cp))
        
        Ris = uniqueinds(target, i)
        m = dim(i)
        Tmat = reshape(array(target, (i, Ris...)), (m, dim(Ris)))
        q, r, p = qr(Tmat, ColumnNorm())
        dr = diag(r)
        r = nothing
        GC.gc()
        #meff = sum(abs.(diag(r)) .> trunc_tol)
        meff = sum(abs.(dr) ./ maximum(abs.(dr)) .> trunc_tol)
        push!(effective_ranks, meff)
        
        #q,r,p = lu(Tmat', RowMaximum(), allowsingular=true)
        
        ### QR based inital guess strategy.
        idx = Index(m, "rank")
        qt = had_contract(itensor(copy(q), Index(m),idx), itensor(dr, idx), idx)
        q = array(qt)
        push!(qr_factors, q)

        push!(ref_pivs, deepcopy(p))
        ## Potentially, we should look at r and start sampling when the 
        ## value on the diagonal falls below some threshold (equivalent to running a truncated CP-QR)
        p1 = p[1:meff]
        ## We skip the rest of the pivots in p because we assume we took all the important directions already
        p_rest = p[meff+1:end]
        p2 = shuffle_pivots ? p_rest[randperm(length(p_rest))] : p_rest
        p = vcat(p1, p2)

        # p = randperm(dim(Ris))
        push!(pivots, p)

        dRis = dim(Ris)
        int_end = stop(alg)
        int_end = length(int_end) == 1 ? int_end[1] : int_end[n]
        int_end = iszero(int_end) ? dRis : int_end
        int_end = dRis < int_end ? dRis : int_end

        int_start = start(alg)
        int_start = length(int_start) == 1 ? int_start[1] : int_start[n]
        @assert int_start > 0 && int_start ≤ int_end

        ndim = int_end - int_start + 1
        piv_id = Index(ndim, "pivot")

        push!(projectors, itensor(tensor(Diag(p[int_start:int_end]), (Ris..., piv_id))))
        TP = fused_flatten_sample(target, n, projectors[n])
        
        push!(targets, TP)
    end
    extra_args[:ref_projectors] = ref_pivs
    extra_args[:projects] = pivots
    extra_args[:projects_tensors] = projectors
    extra_args[:target_transform] = targets
    extra_args[:qr_factors] = qr_factors
    extra_args[:effective_ranks] = effective_ranks
    extra_args[:normal] = normal
    
    return ALS(ITensor(inds(target)), alg, extra_args, check)
end

function compute_als(
    alg::SEQRCSPivProjected,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    shuffle_pivots = true,
    trunc_tol = 0.01,
    normal = true,
    injective = false,
    kwargs...
)
    lst = random_modes(alg)
    lst = isnothing(lst) ? [] : lst
    rank_sk = rank_vect(alg)
    ref_pivs = Vector{Vector{Int}}()
    pivots = Vector{Vector{Int}}()
    projectors = Vector{ITensor}()
    targets = Vector{ITensor}()
    qr_factors = Vector{AbstractArray}()
    effective_ranks = Vector{Int}()
    piv_id = nothing
    for (i, n) in zip(inds(target), 1:length(cp))
        Ris = uniqueinds(target, i)

        dRis = dim(Ris)
        int_end = stop(alg)
        int_end = length(int_end) == 1 ? int_end[1] : int_end[n]
        int_end = iszero(int_end) ? dRis : int_end
        int_end = dRis < int_end ? dRis : int_end

        int_start = start(alg)
        int_start = length(int_start) == 1 ? int_start[1] : int_start[n]
        @assert int_start > 0 && int_start ≤ int_end

        q = nothing
        r = nothing
        m = dim(i)
        if n in lst
            ## TODO there is still a bug in this line below
            k_sk = isnothing(rank_sk) ? int_end : rank_sk[n]
            l=Int(round(3 * m * log(m)))
            # l=Int(round(3 * m )) 
            s=Int(round(log(m)))
            q,r,p = SEQRCS(target,n,i,l,s,k_sk; compute_r = false, use_omega=false,injective = injective)
            # p = vcat(p[1:m], p[m+1:end][randperm(end-m)])
        else
            Tmat = reshape(array(target, (i, Ris...)), (dim(i), dim(Ris)))
            q, r, p = qr(Tmat, ColumnNorm())
        end
        dr = diag(r)
        r = nothing
        GC.gc()

        push!(ref_pivs, deepcopy(p))

        # meff = sum(abs.(diag(r)) ./ maximum(abs.(diag(r))) .> trunc_tol)
        meff = sum(abs.(dr) ./ maximum(abs.(dr)) .> trunc_tol)
        push!(effective_ranks, meff)
        p1 = p[1:meff]
        ## We skip the rest of the pivots in p because we assume we took all the important directions already
        p_rest = p[meff+1:end]
        p2 = shuffle_pivots ? p_rest[randperm(length(p_rest))] : p_rest
        p = vcat(p1, p2)
        push!(pivots, p)

        ### QR based inital guess strategy.
        idx = Index(m, "rank")
        qt = had_contract(itensor(copy(q), Index(m),idx), itensor(dr, idx), idx)
        q = array(qt)
        push!(qr_factors, q)

        ndim = int_end - int_start + 1
        piv_id = Index(ndim, "pivot")

        push!(projectors, itensor(tensor(Diag(p[int_start:int_end]), (Ris..., piv_id))))
        TP = fused_flatten_sample(target, n, projectors[n])
        
    push!(targets, TP)
    end
    extra_args[:ref_projectors] = ref_pivs
    extra_args[:projects] = pivots
    extra_args[:projects_tensors] = projectors
    extra_args[:target_transform] = targets
    extra_args[:qr_factors] = qr_factors
    extra_args[:effective_ranks] = effective_ranks
    extra_args[:normal] = normal
    
    return ALS(ITensor(inds(target)), alg, extra_args, check)
end

function compute_als(
    alg::KSEQRCSPivProjected,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    shuffle_pivots = true,
    trunc_tol = 0.01,
    normal = true,
    injective = false,
    guess_num_levs=nothing,
    prelim_sample_size=nothing,
    prelim_niter=10,
    kwargs...
)
    updated_cpd=nothing
    prelim_sample_size = isnothing(updated_cpd) ? (10 * dim(cp_rank(cp))) : (5 * guess_num_levs)
    if isnothing(guess_num_levs)
        updated_cpd = ITensorCPD.als_optimize(target, cp; alg=ITensorCPD.LevScoreSampled(prelim_sample_size),
        check=ITensorCPD.NoCheck(prelim_niter), normal=true, stop_resample=0,verbose=false)
    else
        dummy_cpd = random_CPD(target, guess_num_levs)
        updated_cpd = ITensorCPD.als_optimize(target, dummy_cpd; alg=ITensorCPD.LevScoreSampled(prelim_sample_size),
        check=ITensorCPD.NoCheck(prelim_niter), normal=true, stop_resample=0, verbose=false)
    end
    lst = random_modes(alg)
    lst = isnothing(lst) ? [] : lst
    cprank = cp_rank(updated_cpd)
    rank_sk = rank_vect(alg)
    ref_pivs = Vector{Vector{Int}}()
    pivots = Vector{Vector{Int}}()
    projectors = Vector{ITensor}()
    targets = Vector{ITensor}()
    qr_factors = Vector{AbstractArray}()
    effective_ranks = Vector{Int}()
    piv_id = nothing
    for (i, n) in zip(inds(target), 1:length(cp))
        Ris = uniqueinds(target, i)

        dRis = dim(Ris)
        int_end = stop(alg)
        int_end = length(int_end) == 1 ? int_end[1] : int_end[n]
        int_end = iszero(int_end) ? dRis : int_end
        int_end = dRis < int_end ? dRis : int_end

        int_start = start(alg)
        int_start = length(int_start) == 1 ? int_start[1] : int_start[n]
        @assert int_start > 0 && int_start ≤ int_end

        q = nothing
        r = nothing
        ## I still use dim(i) because we are looking for the projection on dimension i. If we choose R its 
        ## very expensive. This works well so we need to figure out what size to make this variable.
        m = dim(i)
        if n in lst
            ## TODO there is still a bug in this line below
            k_sk = isnothing(rank_sk) ? int_end : rank_sk[n]
            l=Int(round(3 * m * log(m)))
            # l=Int(round(3 * m )) 
            s=Int(round(log(m)))
            q,r,p = SEQRCS(updated_cpd.factors[1:end .!= n], cprank,l,s,k_sk; compute_r = false, use_omega=false,injective = injective)
            # p = vcat(p[1:m], p[m+1:end][randperm(end-m)])
        else
            ## TODO there should be a QR algorithm for structured tensors like the QR.
            krp = itensor(array(ITensorCPD.had_contract(updated_cpd.factors[1:end .!= n], cprank), (Ris..., cprank)), Ris..., cprank)
            Tmat = reshape(array(krp, (cprank, Ris...)), (dim(cprank), dim(Ris)))
            q, r, p = qr(Tmat, ColumnNorm())
        end
        dr = diag(r)
        r = nothing
        GC.gc()

        push!(ref_pivs, deepcopy(p))

        # meff = sum(abs.(diag(r)) ./ maximum(abs.(diag(r))) .> trunc_tol)
        meff = sum(abs.(dr) ./ maximum(abs.(dr)) .> trunc_tol)
        push!(effective_ranks, meff)
        p1 = p[1:meff]
        ## We skip the rest of the pivots in p because we assume we took all the important directions already
        p_rest = p[meff+1:end]
        p2 = shuffle_pivots ? p_rest[randperm(length(p_rest))] : p_rest
        p = vcat(p1, p2)
        push!(pivots, p)

        ### QR based inital guess strategy.
        # idx = Index(m, "rank")
        # qt = had_contract(itensor(copy(q), Index(m),idx), itensor(dr, idx), idx)
        # q = array(qt)
        # push!(qr_factors, q)

        ndim = int_end - int_start + 1
        piv_id = Index(ndim, "pivot")

        push!(projectors, itensor(tensor(Diag(p[int_start:int_end]), (Ris..., piv_id))))
        TP = fused_flatten_sample(target, n, projectors[n])
        
        push!(targets, TP)
    end
    extra_args[:ref_projectors] = ref_pivs
    extra_args[:projects] = pivots
    extra_args[:projects_tensors] = projectors
    extra_args[:target_transform] = targets
    extra_args[:qr_factors] = qr_factors
    extra_args[:effective_ranks] = effective_ranks
    extra_args[:normal] = normal
    # extra_args[:krp_views] = Vector{Vector{arraytype}}()
    
    return ALS(ITensor(inds(target)), alg, extra_args, check)
end