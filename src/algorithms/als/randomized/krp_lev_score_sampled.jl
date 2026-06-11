using ITensors: Index
using ITensors.NDTensors: data
using ITensors.NDTensors.Expose: expose


### With this solver we are going to compute sampling projectors for LS decomposition
### based on the leverage score of the factor matrices. Then we are going to solve a
### sampled least squares problem 
struct LevScoreSampled <: ProjectionAlgorithm
    NSamples::Tuple
end

    # What happens when sampling is 0?
    LevScoreSampled() = LevScoreSampled((1,))
    LevScoreSampled(n::Int) = LevScoreSampled((n,))

    nsamples(alg::LevScoreSampled) = alg.NSamples

    ## We are going to construct a matrix of sampled indices of the tensor
    function project_krp(::LevScoreSampled, als, factors, cp, rank::Index, fact::Int)
        nsamps = nsamples(als.mttkrp_alg)
        nsamps = length(nsamps) == 1 ? nsamps[1] : nsamps[fact]

        resample = als.additional_items[:stop_resample]
        resample = resample < 0 || resample > iter(als)
        if resample
            sampled_cols = sample_factor_matrices(nsamps, fact, als.additional_items[:factor_weights])
            ## Write new samples to pivot tensor
            
            array(als.additional_items[:projects_tensors][fact]) .= sampled_cols
        else
            sampled_cols = array(als.additional_items[:projects_tensors][fact])
        end
        
        return pivot_hadamard(factors, rank, sampled_cols, inds(als.additional_items[:projects_tensors][fact])[1])
    end

    function matricize_tensor(::LevScoreSampled, als, factors, cp, rank::Index, fact::Int)
        ## I need to turn this into an ITensor and then pass it to the computed algorithm.
        return matricize_tensor(als.mttkrp_alg, Val(als.additional_items[:cache_sampled_targets]), als, factors, cp, rank, fact)
    end

    function matricize_tensor(::LevScoreSampled, ::Val{false}, als, factors, cp, rank::Index, fact::Int)
            return fused_flatten_sample(als.target, fact, als.additional_items[:projects_tensors][fact])
    end

    function matricize_tensor(::LevScoreSampled, ::Val{true}, als, factors, cp, rank::Index, fact::Int)
        if als.check.iter ≤  als.additional_items[:stop_resample]
            als.additional_items[:sampled_targets][fact] = fused_flatten_sample(als.target, fact, als.additional_items[:projects_tensors][fact])
        end

        return @inbounds als.additional_items[:sampled_targets][fact]
    end

    function post_solve(::LevScoreSampled, als, factors, λ, cp, rank::Index, fact::Integer) 
        ## update the factor weights.
        @inbounds als.additional_items[:factor_weights][fact] = compute_leverage_score_probabilitiy(factors[fact], ind(cp, fact); use_variance = als.additional_items[:variance_truncation])
    end

### With this solver we are going to compute sampling projectors for LS decomposition
### based on the leverage score of the factor matrices. Then we are going to solve a
### sampled least squares problem. To make the sampling process more efficient this algorithm
### gathers samples in blocks
struct BlockLevScoreSampled<: ProjectionAlgorithm 
    NSamples::Tuple
    Blocks::Tuple
end

    BlockLevScoreSampled() = BlockLevScoreSampled((0,), (1,))
    BlockLevScoreSampled(n::Int) = BlockLevScoreSampled((n,), (1,))
    BlockLevScoreSampled(n::Int, m::Int) = BlockLevScoreSampled((n,), (m,))
    BlockLevScoreSampled(n::Tuple) = BlockLevScoreSampled{n, (1,)}()
    BlockLevScoreSampled(n::Int, m::Tuple) = BlockLevScoreSampled((n,), m)
    BlockLevScoreSampled(n::Tuple, m::Int) = BlockLevScoreSampled(n, (m,))

    nsamples(alg::BlockLevScoreSampled) = alg.NSamples
    blocks(alg::BlockLevScoreSampled) = alg.Blocks

    ## We are going to construct a matrix of sampled indices of the tensor
    function project_krp(::BlockLevScoreSampled, als, factors, cp, rank::Index, fact::Int)
        nsamps = nsamples(als.mttkrp_alg)
        nsamps = length(nsamps) == 1 ? nsamps[1] : nsamps[fact]
        block_size = blocks(als.mttkrp_alg)
        block_size = length(block_size) == 1 ? block_size[1] : block_size[fact]

        resample = als.additional_items[:stop_resample]
        resample = resample < 0 || resample > iter(als)
        if resample
            sampled_cols = block_sample_factor_matrices(nsamps, als.additional_items[:factor_weights], block_size, fact)
            ## Write new samples to pivot tensor
            array(als.additional_items[:projects_tensors][fact]) .= sampled_cols
        else
            sampled_cols = array(als.additional_items[:projects_tensors][fact])
        end
        
        return pivot_hadamard(factors, rank, sampled_cols, inds(als.additional_items[:projects_tensors][fact])[1])
    end

    function matricize_tensor(::BlockLevScoreSampled, als, factors, cp, rank::Index, fact::Int)
        ## I need to turn this into an ITensor and then pass it to the computed algorithm.
        return fused_flatten_sample(als.target, fact, als.additional_items[:projects_tensors][fact])
    end


    function post_solve(::BlockLevScoreSampled, als, factors, λ, cp, rank::Index, fact::Integer) 
    ## update the factor weights.
        als.additional_items[:factor_weights][fact] = compute_leverage_score_probabilitiy(factors[fact], ind(cp, fact); use_variance = als.additional_items[:variance_truncation])
    end

