function compute_als(
    alg::LevScoreSampled,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    normal=false,
    stop_resample=-1,
    cache_sampled_targets=true,
    kwargs...
)
    ## For each factor matrix compute its weights
    extra_args[:factor_weights] = [compute_leverage_score_probabilitiy(cp[i], ind(cp, i)) for i in 1:length(cp)]
    projects_tensors = Vector{ITensor}()
    cache_sampled_targets = (stop_resample == -1 ? false : cache_sampled_targets)
    for fact in 1:length(cp)
        ## grab the tensor indices for all other factors but fact
        Ris = inds(cp)[1:end .!= fact]
        dRis = dims(Ris)

        ## sample the factor matrix space one taking nsamps independent samples from each
        ## tensor
        nsamps = nsamples(alg)
        nsamps = length(nsamps) == 1 ? nsamps[1] : nsamps[fact]
        sampled_cols = sample_factor_matrices(nsamps, fact, extra_args[:factor_weights])

        ## We store the rows of the flattened tensor (i.e. α) because it can be converted to factor values
        ## or values in the vectorized tensor
        sampled_tensor_cols = multi_coords_to_column(dRis, sampled_cols)
        piv_ind = Index(length(sampled_tensor_cols), "selector_$(fact)")

        ## make the canonical pivot tensor. This list of pivots will be overwritten each ALS iteration
        push!(projects_tensors, itensor(tensor(Diag(sampled_tensor_cols), (Ris..., piv_ind))))
    end
    extra_args[:projects_tensors] = projects_tensors
    extra_args[:normal] = normal
    extra_args[:stop_resample] = stop_resample
    extra_args[:sampled_targets] = Vector{ITensor}(undef, ndims(target))
    extra_args[:cache_sampled_targets] = cache_sampled_targets
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
    kwargs...
)
    ## For each factor matrix compute its weights
    extra_args[:factor_weights] = [compute_leverage_score_probabilitiy(cp[i], ind(cp, i)) for i in 1:length(cp)]
    projects_tensors = Vector{ITensor}()
    for fact in 1:length(cp)
        ## grab the tensor indices for all other factors but fact
        Ris = inds(cp)[1:end .!= fact]
        dRis = dims(Ris)

        ## sample the factor matrix space one taking nsamps independent samples from each
        ## tensor
        nsamps = nsamples(alg)
        nsamps = length(nsamps) == 1 ? nsamps[1] : nsamps[fact]
        block_size = blocks(alg)
        block_size = length(block_size) == 1 ? block_size[1] : block_size[fact]
        
        sampled_cols = block_sample_factor_matrices(nsamps, extra_args[:factor_weights] , block_size, fact)

        ## We store the rows of the flattened tensor (i.e. α) because it can be converted to factor values
        ## or values in the vectorized tensor
        sampled_tensor_cols = multi_coords_to_column(dRis, sampled_cols)
        piv_ind = Index(length(sampled_tensor_cols), "selector_$(fact)")

        ## make the canonical pivot tensor. This list of pivots will be overwritten each ALS iteration
        push!(projects_tensors, itensor(tensor(Diag(sampled_tensor_cols), (Ris..., piv_ind))))
    end
    ## Notice the pivot tensor is actually a low rank tensor it stores the diagonal pivot values
    ## in α form (rows of the matricized tensor) and the indices which are captured in the pivot.
    ## The order of indices are (indices which connect to the pivot, pivot_index).
    extra_args[:projects_tensors] = projects_tensors
    extra_args[:normal] = normal
    extra_args[:stop_resample] = stop_resample
    return ALS(target, alg, extra_args, check)
end
