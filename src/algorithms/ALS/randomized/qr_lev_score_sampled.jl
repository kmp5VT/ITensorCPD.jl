using ITensors: Index
using ITensors.NDTensors: data
using ITensors.NDTensors.Expose: expose

### With this solver we are trying to solve the modified least squares problem
### || T(a,b,c) P(b,c,l) - A(a,m) (B(b,m) ⊙ C(c,m)) P(b,c,l) ||² (and equivalent for all other factor matrices)
### In order to solve this equation we need the following to be true P(b,c l) P(b',c', l) ≈ I(b,c,b',c')
### One easy way to do this is to make P a pivot matrix from a QR or LU. We will form P by taking the pivoted QR
### of T and choose a set certain number of pivots in each row.
struct QRPivProjected <: ProjectionAlgorithm 
    Start::Union{<:Tuple, <:Int}
    End::Union{<:Tuple, <:Int}
end

    ## TODO modify to use ranges 
    QRPivProjected() = QRPivProjected(1,0)
    QRPivProjected(n::Int) = QRPivProjected(1,n)
    QRPivProjected(n::Tuple) = QRPivProjected(Tuple(Int.(ones(length(n)))),n)

    ## TODO this fails for new_start as a tuple 
    copy_alg(alg::QRPivProjected, new_start = 0, new_end = 0) = 
    QRPivProjected((iszero(new_start) ? alg.Start : new_start), (iszero(new_end) ? alg.End : new_end))

### This solver is nearly identical to the one above. The major difference is that the 
### QR method is replaced with a custom algorithm for randomized pivoted QR.
### The SEQRCS was developed by Israa Fakih and Laura Grigori (DOI: )
### The randomized method is only included for specified modes
struct SEQRCSPivProjected <: ProjectionAlgorithm
    Start::Union{<:Tuple, <:Int}
    End::Union{<:Tuple, <:Int}
    random_modes
    rank_vect
    
    function SEQRCSPivProjected(n, m, rrmodes=nothing, rank_vect=nothing) 
        rrmodes = isnothing(rrmodes) ? nothing : Tuple(rrmodes)
        rank_vect = isnothing(rank_vect) ? nothing : rank_vect isa Dict ? rank_vect : Dict(rrmodes .=> Tuple(rank_vect))
        new(n, m, rrmodes, rank_vect)
    end
end

    ## TODO modify to use ranges 
    SEQRCSPivProjected() = SEQRCSPivProjected(1, 0, nothing, nothing)
    SEQRCSPivProjected(n::Int) = SEQRCSPivProjected(1, n, nothing, nothing)
    SEQRCSPivProjected(n::Tuple) = SEQRCSPivProjected(Tuple(ones(Int, length(n))), n, nothing, nothing)

    random_modes(alg::SEQRCSPivProjected) = alg.random_modes
    rank_vect(alg::SEQRCSPivProjected) = alg.rank_vect

    copy_alg(alg::SEQRCSPivProjected, new_start = 0, new_end = 0) = 
    SEQRCSPivProjected((iszero.(new_start) ? alg.Start : new_start), (iszero.(new_end) ? alg.End : new_end), alg.random_modes, alg.rank_vect)


### This algorithm is similar to the SE-QRCS algorithms above but tries to fix the problem of when 
### There are a large number of leverage scores in the problem. This method leverages the convergence of the
### leverage scores of the KRP to find a single sample from the distribution of the KRP (instead of the target tensor).
### The method first computes a small uniform random sampling of each LS subproblem and solves the ALS one time
### This puts the leverage scores of the factors "close" to the leverage scores of T. We then call the SE-QRCS method
### on the KRP (in a matrix free way) to order the positions of the leverage scores in the KRP and sample from this 
### distribution once. This allows the number of known large leverage score positions to scale with R and not the dimension
### of the target tensor.
struct KSEQRCSPivProjected <: ProjectionAlgorithm
    Start::Union{<:Tuple, <:Int}
    End::Union{<:Tuple, <:Int}
    random_modes
    rank_vect
    
    function KSEQRCSPivProjected(n, m, rrmodes=nothing, rank_vect=nothing) 
        rrmodes = isnothing(rrmodes) ? nothing : Tuple(rrmodes)
        rank_vect = isnothing(rank_vect) ? nothing : rank_vect isa Dict ? rank_vect : Dict(rrmodes .=> Tuple(rank_vect))
        new(n, m, rrmodes, rank_vect)
    end
end

    ## TODO modify to use ranges 
    KSEQRCSPivProjected() = KSEQRCSPivProjected(1, 0, nothing, nothing)
    KSEQRCSPivProjected(n::Int) = KSEQRCSPivProjected(1, n, nothing, nothing)
    KSEQRCSPivProjected(n::Tuple) = KSEQRCSPivProjected(Tuple(ones(Int, length(n))), n, nothing, nothing)

    random_modes(alg::KSEQRCSPivProjected) = alg.random_modes
    rank_vect(alg::KSEQRCSPivProjected) = alg.rank_vect

    copy_alg(alg::KSEQRCSPivProjected, new_start = 0, new_end = 0) = 
    KSEQRCSPivProjected((iszero.(new_start) ? alg.Start : new_start), (iszero.(new_end) ? alg.End : new_end), alg.random_modes, alg.rank_vect)

## This is a union class so that the operations work on both pivot based solver algorithms
const PivotBasedSolvers = Union{QRPivProjected, SEQRCSPivProjected, KSEQRCSPivProjected}

    start(alg::PivotBasedSolvers) = alg.Start
    stop(alg::PivotBasedSolvers) = alg.End

    ## This does an out of place copy of the als with a change in the samples. This way you don't
    ## need to recompute the QR. This is a "dumb" algorithm because it resamples the full
    ## target tensor so a future algorithm should just modify the target to reduce the amount of work.
    ## reshuffle redoes the sampling of the pivots beyond the rank of the matrix.
    function update_samples(target, als, new_num_end; reshuffle = false, new_num_start = 0)
        @assert(als.mttkrp_alg isa PivotBasedSolvers)
        
        ## Make an updated alg with correct new range
        updated_alg = copy_alg(als.mttkrp_alg, new_num_start, new_num_end)
        
        pivots = deepcopy(als.additional_items[:projects])
        ref_pivs = deepcopy(als.additional_items[:ref_projectors])
        projectors = deepcopy(als.additional_items[:projects_tensors])
        targets = similar(als.additional_items[:target_transform])
        effective_ranks = als.additional_items[:effective_ranks]
        for (p,pos, projector_tensor, meff, m) in zip(ref_pivs,1:length(pivots), projectors, effective_ranks, dims(als.target))
            ## This is reshuffling the indices
            Ris = inds(target)[1:end .!= pos]
            if reshuffle
                p1 = p[1:meff]
                p_rest = p[meff+1:end]
                p2 = p_rest[randperm(length(p_rest))]
                pshuff = vcat(p1, p2)
                
                pshuff = column_to_multi_coords(pshuff, dims(Ris))
                pivots[pos] = pshuff
            end

            dRis = dim(Ris)
            int_end = stop(updated_alg)
            int_end = length(int_end) == 1 ? int_end[1] : int_end[pos]
            int_end = iszero(int_end) ? dRis : int_end
            int_end = dRis < int_end ? dRis : int_end

            int_start = start(updated_alg)
            int_start = length(int_start) == 1 ? int_start[1] : int_start[pos]
            @assert int_start > 0 && int_start ≤ int_end

            ndim = int_end - int_start + 1
            piv_id = Index(ndim, "pivot")
            nind = Index(length(Ris))

            projectors[pos] = itensor(Int, pivots[pos][int_start:int_end, :], (piv_id, nind))

            targets[pos] = fused_flatten_sample(target, pos, projectors[pos])
        end

        extra_args = Dict(
        :mttkrp_contract_sequences => als.additional_items[:mttkrp_contract_sequences],
        :ref_projectors => ref_pivs,
        :projects => pivots,
        :projects_tensors => projectors,
        :target_transform => targets,
        :qr_factors => als.additional_items[:qr_factors],
        :effective_ranks => als.additional_items[:effective_ranks],
        :normal => als.additional_items[:normal],
        )
        return ALS(als.target, updated_alg, extra_args, als.check)
    end

    function project_krp(::PivotBasedSolvers, als, factors, cp, rank::Index, fact::Int)
        ## This computes the exact grammian of the normal equations.
        # part_grammian = factors .* dag.(prime.(factors; tags = tags(rank)))
        # p = part_grammian[1]
        # for g in part_grammian[2:end]
        #     hadamard_product!(p,p,g)
        # end
        # return p
        return ITensorCPD.pivot_hadamard(factors, rank, als.additional_items[:projects_tensors][fact])
    end

    function matricize_tensor(::PivotBasedSolvers, als, factors, cp, rank::Index, fact::Int)
        ## This computes the projected MTTKRP
        # return als.additional_items[:target_transform][fact] *  ITensorCPD.pivot_hadamard(dag.(factors[1:end .!= fact]), rank, als.additional_items[:projects_tensors][fact])
        return @inbounds als.additional_items[:target_transform][fact]
    end

    function post_solve(::PivotBasedSolvers, als, factors, λ, cp, rank::Index, fact::Integer) end

