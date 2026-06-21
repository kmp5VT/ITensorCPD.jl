function compute_als(
    alg::LevScoreSampled,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    normal=false,
    stop_resample=-1,
    cache_sampled_targets=true,
    variance_truncation = true,
    kwargs...
)
    ## For each factor matrix compute its weights
    extra_args[:factor_weights] = [compute_leverage_score_probabilitiy(cp[i], ind(cp, i); use_variance = variance_truncation) for i in 1:length(cp)]
    projects_tensors = Vector{ITensor}()
    cache_sampled_targets = (stop_resample == -1 ? false : cache_sampled_targets)
    for fact in 1:length(cp)
        ## grab the tensor indices for all other factors but fact
        Ris = inds(cp)[1:end .!= fact]

        ## sample the factor matrix space one taking nsamps independent samples from each
        ## tensor
        nsamps = nsamples(alg)
        nsamps = length(nsamps) == 1 ? nsamps[1] : nsamps[fact]
        sampled_cols = sample_factor_matrices(nsamps, fact, extra_args[:factor_weights])

        ## We store the tensor-wise indices because storing the column position or the vectorized step position (α) 
        ## because this value is limited by dim(Ris) which can overflow int64
        nind = Index(length(Ris))
        piv_ind = Index(nsamps, "selector_$(fact)")

        ## make the canonical pivot tensor. This list of pivots will be overwritten each ALS iteration
        push!(projects_tensors, itensor(tensor(Dense(sampled_cols), (piv_ind, nind))))
    end
    extra_args[:projects_tensors] = projects_tensors
    extra_args[:normal] = normal
    extra_args[:stop_resample] = stop_resample
    extra_args[:sampled_targets] = Vector{ITensor}(undef, ndims(target))
    extra_args[:cache_sampled_targets] = cache_sampled_targets
    extra_args[:variance_truncation] = variance_truncation
    return ALS(target, alg, extra_args, check)
end

function compute_als(
    alg::BlockLevScoreSampled,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    normal=false,
    stop_resample=-1,
    cache_sampled_targets=true,
    variance_truncation = true,
    kwargs...
)
    ## For each factor matrix compute its weights
    extra_args[:factor_weights] = [compute_leverage_score_probabilitiy(cp[i], ind(cp, i);  use_variance = variance_truncation) for i in 1:length(cp)]
    projects_tensors = Vector{ITensor}()
    cache_sampled_targets = (stop_resample == -1 ? false : cache_sampled_targets)
    for fact in 1:length(cp)
        ## grab the tensor indices for all other factors but fact
        Ris = inds(cp)[1:end .!= fact]

        ## sample the factor matrix space one taking nsamps independent samples from each
        ## tensor
        nsamps = nsamples(alg)
        nsamps = length(nsamps) == 1 ? nsamps[1] : nsamps[fact]
        block_size = blocks(alg)
        block_size = length(block_size) == 1 ? block_size[1] : block_size[fact]
        
        sampled_cols = block_sample_factor_matrices(nsamps, extra_args[:factor_weights] , block_size, fact)

        ## We store the tensor-wise indices because storing the column position or the vectorized step position (α) 
        ## because this value is limited by dim(Ris) which can overflow int64
        nind = Index(length(Ris))
        piv_ind = Index(nsamps, "selector_$(fact)")

        ## make the canonical pivot tensor. This list of pivots will be overwritten each ALS iteration
        push!(projects_tensors, itensor(Int, sampled_cols, (piv_ind, nind)))
    end
    ## Notice the pivot tensor is actually a low rank tensor it stores the diagonal pivot values
    ## in α form (rows of the matricized tensor) and the indices which are captured in the pivot.
    ## The order of indices are (indices which connect to the pivot, pivot_index).
    extra_args[:projects_tensors] = projects_tensors
    extra_args[:normal] = normal
    extra_args[:stop_resample] = stop_resample
    extra_args[:cache_sampled_targets] = cache_sampled_targets
    extra_args[:variance_truncation] = variance_truncation
    return ALS(target, alg, extra_args, check)
end
