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
    function solve_ls_problem(::MttkrpAlgorithm, _, krp, mtkrp, rank)
        ## potentially better to first inverse the grammian then contract
        ## qr(A, Val(true))
        #solution = array(dag(krp)) \ transpose(array(mtkrp))
        solution = ldiv_solve!!(expose(array(dag(krp))), expose(transpose(array(mtkrp))); factorizeA = true)
        i = ind(mtkrp, 1)
        return itensor(copy(transpose(solution)), i,rank)
    end