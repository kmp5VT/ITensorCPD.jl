using ITensors: Index
using ITensors.NDTensors: data

## These are solvers which take advantage to the canonical CP-ALS normal equations
abstract type MttkrpAlgorithm end

    ## Default convergence checking function for Mttkrp based algorithms
    function check_converge(::MttkrpAlgorithm, 
        converge::ConvergeAlg, als,
        mtkrp, factors, λ, 
        verbose=false)::Bool
        # potentially save the MTTKRP for the loss function

        save_mttkrp(converge, mtkrp)

        return check_converge(converge, factors, λ,  als.additional_items[:part_grammian]; verbose)
    end

    ## Default algorithm to compute KRP for MTTKRP algorithsm.
    ## Actually computes the grammian.
    function compute_krp(::MttkrpAlgorithm,
        als, factors, cp, rank::Index, fact::Int)
        ## compute the grammian which requires the hadamard product
        grammian = similar(als.additional_items[:part_grammian][1])
        fill!(grammian, one(eltype(cp)))
        num_factors = length(cp)
        for i = 1:num_factors
            if i == fact
                continue
            end
            grammian = hadamard_product(grammian, als.additional_items[:part_grammian][i])
        end
        return grammian
    end

    ## Default algorithm uses the pivoted QR to solve LS problem.
    function solve_ls_problem(::MttkrpAlgorithm,krp, mtkrp, rank)
        ## potentially better to first inverse the grammian then contract
        ## qr(A, Val(true))
        solution = qr(array(dag(krp)), ColumnNorm()) \ transpose(array(mtkrp))
        i = ind(mtkrp, 1)
        return itensor(copy(transpose(solution)), i,rank)
    end

    ## This version assumes we have the exact target and can form the tensor
    ## This forms the khatri-rao product for a single value of r and immediately
    ## contracts it with the target tensor. This is relatively expensive because the KRP will be
    ## order $d - 1$ where d is the number of modes in the target tensor.
    ## This process could be distributed.
    struct KRP <: MttkrpAlgorithm end

        function matricize_tensor(::KRP, als, factors, cp, rank::Index, fact::Int)

            factor_portion = @view factors[1:end .!= fact]
            sequence = ITensors.default_sequence()
            krp = had_contract(dag.(factor_portion), rank; sequence)

            m = als.target * krp
            return m
        end

        function post_solve(::KRP, als, factors, λ, cp, rank::Index, fact::Integer)
            als.additional_items[:part_grammian][fact] =
                factors[fact] * dag(prime(factors[fact]; tags = tags(rank)))
        end

    ## This code skips computing the khatri-rao product by incrementally 
    ## contracting the factor matrices into the tensor for each value of r
    ## This process could be distributed.
    struct direct <: MttkrpAlgorithm end

        function matricize_tensor(::direct, als, factors, cp, rank::Index, fact::Int)
            factor_portion = @view factors[1:end .!= fact]
            if isnothing(als.additional_items[:mttkrp_contract_sequences][fact])
                als.additional_items[:mttkrp_contract_sequences][fact] =
                    optimal_had_contraction_sequence([als.target, dag.(factor_portion)...], rank)
            end
            m = had_contract(
                [als.target, dag.(factor_portion)...],
                rank;
                sequence = als.additional_items[:mttkrp_contract_sequences][fact],
            )
            return m
        end

        function post_solve(::direct, als, factors, λ, cp, rank::Index, fact::Integer) 
            als.additional_items[:part_grammian][fact] .=
                factors[fact] * dag(prime(factors[fact]; tags = tags(rank)))
        end

    ### This solver is slightly silly. It takes a higher-order tensor T forms the SVD of every unfolding
    ### for example T(a,b,c) => T(a,bc) => U(a,r) S(r,rp) V(rp,bc). We use this decomposition to represent T as
    ### T(a,b'c') V(r,b'c') V(r,bc). Then we solve f(A) = || T(a,b'c') V(r,b'c') V(r,bc) - A(a,m) (B(b,m) ⊙ C(c,m)) ||².
    ### We repeat this process for every factor matrix. This gains us a smaller target tensor to store,
    ### i.e. (U(a,r) S(r,rp)) for each mode but solving the least squares problem is no less expensive.
    struct TargetDecomp <: MttkrpAlgorithm end

        function matricize_tensor(::TargetDecomp, als, factors, cp, rank::Index, fact::Int)
            factor_portion = @view factors[1:end .!= fact]
            m = had_contract(
                [
                    als.additional_items[:target_transform][fact],
                    als.additional_items[:target_decomps][fact],
                    dag.(factor_portion)...,
                ],
                rank;
            )

            return m
        end

        function post_solve(::TargetDecomp, als, factors, λ, cp, rank::Index, fact::Integer)
            als.additional_items[:part_grammian][fact] =
                factors[fact] * dag(prime(factors[fact]; tags = tags(rank)))
        end


    ################
    ## This solver is based on ITensorNetwork
    ## It allows one to take a completely connected 
    ## ITensorNetwork and decomposes it into a CPD
    ## You can't currently take a network which is and
    ## outer product of two networks

    struct network_solver <: MttkrpAlgorithm end

        function matricize_tensor(::network_solver, als, factors, cp, rank::Index, fact::Int)
            m = similar(factors[fact])

            target_index = ind(cp, fact)
            target_vert = als.additional_items[:ext_ind_to_vertex][target_index]
            p = copy(als.target[target_vert])
            for x in uniqueinds(als.target, target_vert)
                if x == target_index
                    continue
                end
                factor_ind = als.additional_items[:ext_ind_to_factor][x]
                p = had_contract(dag.(factors[factor_ind]), p, rank)
            end

            ## Next I need to figure out which partial hadamard_product to skip
            env_list = ITensorNetwork([
                p,
                (als.additional_items[:partial_mtkrp])[1:end .!= als.additional_items[:factor_to_part_cont][fact]]...,
            ])
            sequence = als.additional_items[:mttkrp_contract_sequences][fact]
            sequence =
                isnothing(sequence) ? optimal_had_contraction_sequence(env_list, rank) : sequence
            p = had_contract(env_list, rank; sequence)
            als.additional_items[:mttkrp_contract_sequences][fact] = sequence
            return p
        end

        function post_solve(::network_solver, als, factors, λ, cp, rank::Index, fact::Integer)
            als.additional_items[:part_grammian][fact] =
                factors[fact] * dag(prime(factors[fact]; tags = tags(rank)))

            ## Once done with all factor which connect to it, then go through uniqueinds and contract in the 
            ## associated new factors
            partial_ind = als.additional_items[:factor_to_part_cont][fact]
            if fact == length(factors) ||
            als.additional_items[:factor_to_part_cont][fact+1] != partial_ind
                ## go through factors
                partial_vertex = als.additional_items[:ext_ind_to_vertex][ind(cp, fact)]
                p = als.target[partial_vertex]
                for uniq in uniqueinds(als.target, partial_vertex)
                    p = had_contract(
                        p,
                        dag(factors[als.additional_items[:ext_ind_to_factor][uniq]]),
                        rank,
                    )
                end
                als.additional_items[:partial_mtkrp][partial_ind] = p
            end
        end

    ## TODO
    ## This next code is going to take the CPD
    ## of a network that has one CPD rank.



    ## These are algorithms which do not use the normal equations and minimizes the loss function 
    ## f(A,B,C) = | PT - [[A, B, C]] P |² using the least squares problems
    ## PT = A* [(B ⊙ C) P]. 
abstract type ProjectionAlgorithm end

    ## Default algorithm to compute the KRP: This computes a projected/sampled KRP
    function compute_krp(::ProjectionAlgorithm, als, factors, cp, rank, fact)
        factor_portion = @view factors[1:end .!= fact]
        return project_krp(als.mttkrp_alg, als, factor_portion, cp, rank, fact)
    end

    ## Default algorithm to compute the convergence, this doesn't do
    ## Anything except it can compute the fit if verbose =true (this is for
    ## testing purposes only) ## TODO make a new convergence tester
    function check_converge(::ProjectionAlgorithm, 
        converge::ConvergeAlg, als,
        mtkrp, factors, λ, 
        verbose=false)::Bool
        # potentially save the MTTKRP for the loss function

        # save_mttkrp(converge, mtkrp)
        cprank = ind(λ, 1)
        if als.check isa FitCheck
            if als.check.iter == 0
                println("Warning: FitCheck is not enabled for $(als.mttkrp_alg) will run $(als.check.max_counter) iterations.")
            end
            als.check.iter += 1
            if verbose
                inner_prod = real((had_contract([als.target, dag.(factors)...], cprank) * dag(λ))[])
                partial_gram = [fact * dag(prime(fact; tags=tags(cprank))) for fact in factors];
                fact_square = ITensorCPD.norm_factors(partial_gram, λ)
                normResidual =
                    sqrt(abs(als.check.ref_norm * als.check.ref_norm + fact_square - 2 * abs(inner_prod)))
                elt = typeof(inner_prod)
                println("$(dim(cprank))\t$(als.check.iter)\t$(one(elt) - normResidual / norm(als.check.ref_norm))")
            end
            if als.check.iter == als.check.max_counter
                als.check.iter = 0
            end
            return false
        end

        return check_converge(converge, factors, λ,  []; verbose)
    end

    ## Default algorithm uses the pivoted QR to solve LS problem.
    function solve_ls_problem(::ProjectionAlgorithm, projected_KRP, project_target, rank)
        # direction = qr(array(projected_KRP * prime(projected_KRP, tags=tags(rank))), ColumnNorm()) \ transpose(array(project_target * projected_KRP))
        direction = qr(array(projected_KRP), ColumnNorm()) \ transpose(array(project_target))
        i = ind(project_target, 1)
        return itensor(copy(transpose(direction)), i,rank)
    end

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

            sampled_cols = sample_factor_matrices(nsamps, fact, als.additional_items[:factor_weights])
            ## Write new samples to pivot tensor
            dRis = dims(inds(cp)[1:end .!= fact])
            data(als.additional_items[:pivot_tensors][fact]) .= multi_coords_to_column(dRis, sampled_cols)
            
            return pivot_hadamard(factors, rank, sampled_cols, inds(als.additional_items[:pivot_tensors][fact])[end])
        end

        function matricize_tensor(::LevScoreSampled, als, factors, cp, rank::Index, fact::Int)
            ## I need to turn this into an ITensor and then pass it to the computed algorithm.
            return fused_flatten_sample(als.target, fact, als.additional_items[:pivot_tensors][fact])
        end


        function post_solve(::LevScoreSampled, als, factors, λ, cp, rank::Index, fact::Integer) 
            ## update the factor weights.
            als.additional_items[:factor_weights][fact] = compute_leverage_score_probabilitiy(factors[fact], ind(cp, fact))
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

        nsamples(alg::BlockLevScoreSampled) = alg.NSamples
        blocks(alg::BlockLevScoreSampled) = alg.Blocks

        ## We are going to construct a matrix of sampled indices of the tensor
        function project_krp(::BlockLevScoreSampled, als, factors, cp, rank::Index, fact::Int)
            nsamps = nsamples(als.mttkrp_alg)
            nsamps = length(nsamps) == 1 ? nsamps[1] : nsamps[fact]
            block_size = blocks(als.mttkrp_alg)
            block_size = length(block_size) == 1 ? block_size[1] : block_size[fact]

            sampled_cols = block_sample_factor_matrices(nsamps, als.additional_items[:factor_weights], block_size, fact)
            ## Write new samples to pivot tensor
            dRis = dims(inds(cp)[1:end .!= fact])
            data(als.additional_items[:pivot_tensors][fact]) .= multi_coords_to_column(dRis, sampled_cols)
            
            return pivot_hadamard(factors, rank, sampled_cols, inds(als.additional_items[:pivot_tensors][fact])[end])
        end

        function matricize_tensor(::BlockLevScoreSampled, als, factors, cp, rank::Index, fact::Int)
            ## I need to turn this into an ITensor and then pass it to the computed algorithm.
            return fused_flatten_sample(als.target, fact, als.additional_items[:pivot_tensors][fact])
        end


        function post_solve(::BlockLevScoreSampled, als, factors, λ, cp, rank::Index, fact::Integer) 
        ## update the factor weights.
            als.additional_items[:factor_weights][fact] = compute_leverage_score_probabilitiy(factors[fact], ind(cp, fact))
        end

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
            rank_vect = isnothing(rank_vect) ? nothing : Dict(rrmodes .=> Tuple(rank_vect))
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

    ## This is a union class so that the operations work on both pivot based solver algorithms
    const PivotBasedSolvers = Union{QRPivProjected, SEQRCSPivProjected}

        start(alg::PivotBasedSolvers) = alg.Start
        stop(alg::PivotBasedSolvers) = alg.End

        ## This does an out of place copy of the als with a change in the samples. This way you don't
        ## need to recompute the QR. This is a "dumb" algorithm because it resamples the full
        ## target tensor so a future algorithm should just modify the target to reduce the amount of work.
        ## reshuffle redoes the sampling of the pivots beyond the rank of the matrix.
        function update_samples(als, new_num_end; reshuffle = false, new_num_start = 0)
            @assert(als.mttkrp_alg isa PivotBasedSolvers)
            
            ## Make an updated alg with correct new range
            updated_alg = copy_alg(als.mttkrp_alg, new_num_start, new_num_end)
            
            pivots = deepcopy(als.additional_items[:projects])
            projectors = deepcopy(als.additional_items[:projects_tensors])
            targets = deepcopy(als.additional_items[:target_transform])
            effective_ranks = als.additional_items[:effective_ranks]
            for (p,pos, projector_tensor, meff) in zip(pivots,1:length(pivots), projectors, effective_ranks)
                ## This is reshuffling the indices
                if reshuffle
                    p1 = p[1:meff]
                    p_rest = p[meff+1:end]
                    p2 = p_rest[randperm(length(p_rest))]
                    p = vcat(p1, p2)
                    
                    pivots[pos] = p
                end

                Ris = inds(projector_tensor)[1:end-1]

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

                projectors[pos] =  itensor(tensor(Diag(p[int_start:int_end]), (Ris..., piv_id)))

                targets[pos] = fused_flatten_sample(als.target, pos, projectors[pos])
            end

            extra_args = Dict(
            :projects => pivots,
            :projects_tensors => projectors,
            :target_transform => targets,
            :qr_factors => als.additional_items[:qr_factors],
            :effective_ranks => als.additional_items[:effective_ranks]
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
            return als.additional_items[:target_transform][fact]
        end


        function post_solve(::PivotBasedSolvers, als, factors, λ, cp, rank::Index, fact::Integer) end

    ## This solver is for computing Tomega = A(B ododt C)omega 
    ## The c1 and C2x vectors are constant vectors for determing the 
    ## Sketching dimension and sparsity parameter.
    struct SketchProjected <: ProjectionAlgorithm
        C1_vect
        C2_vect
    end
        SketchProjected()=SketchProjected(nothing,nothing)

        C1_vect(alg::SketchProjected) = alg.C1_vect
        C2_vect(alg::SketchProjected) = alg.C2_vect


        function project_krp(::SketchProjected, als, factors, cp, rank::Index, fact::Int)
            return ITensorCPD.omega_hadamard(factors, rank, als.additional_items[:sketch_matrices][fact])
        end

        function matricize_tensor(:: SketchProjected, als, factors, cp, rank::Index, fact::Int)
            return als.additional_items[:target_transform][fact]
        end


        function post_solve(::SketchProjected, als, factors, λ, cp, rank::Index, fact::Integer) end



    ## This solver does not form the normal equations. 
    ## We simply compute the khatri rao product and directly compute Ax=B for each least squres problem.
    struct InvKRP <: ProjectionAlgorithm end

        function project_krp(::InvKRP, als, factors, cp, rank::Index, fact::Int)
            return had_contract(factors, rank)
        end
        function matricize_tensor(::InvKRP, als, factors, cp, rank::Index, fact::Int)
            return als.target
        end

        function solve_ls_problem(::InvKRP, projected_KRP, projected_target, rank)
            U, S, V = svd(dag(projected_KRP), rank; use_absolute_cutoff = true, cutoff = 0)
            return prime(projected_target; tags = tags(rank)) * V * (1 ./ S) * U
        end

        function post_solve(::InvKRP, als, factors, λ, cp, rank::Index, fact::Integer) end
