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
        ## This is a angle check function for testing. check_angle can be changed in ITensorCPD
        if check_angle
            appx = ITensorCPD.reconstruct(factors, λ);
            dist = round(dot(als.target, appx) / (norm(als.target) * norm(appx)),digits=5)
            angle = acos(dist)
            println("Angle is : $(angle)")
        end
        if als.check isa FitCheck
            if als.check.iter == 0
                println("Warning: FitCheck is not enabled for $(als.mttkrp_alg) will run $(als.check.max_counter) iterations.")
                if verbose
                    println("Warning: Sampled fit will be provided")
                end
            end
            als.check.iter += 1
            if verbose
                cpd = CPD{ITensor}(factors, λ)
                krpproj = had_contract(factors[1], λ, cprank) * compute_krp(als.mttkrp_alg, als, factors, cpd, cprank, 1)
                tproj =  matricize_tensor(als.mttkrp_alg, als, factors, cpd, cprank, 1)

                cpfit = one(real(eltype(krpproj))) - norm(tproj - krpproj) / norm(tproj)

                println("$(dim(cprank))\t$(als.check.iter)\t$(cpfit)")
            end
            if als.check.iter > als.check.max_counter
                als.check.iter = 0
            end
            return false
        end

        return check_converge(converge, factors, λ,  []; verbose)
    end

    ## Default algorithm uses the pivoted QR to solve LS problem.
    function solve_ls_problem(::ProjectionAlgorithm, als, projected_KRP, project_target, rank)
        # direction = qr(array(projected_KRP * prime(projected_KRP, tags=tags(rank))), ColumnNorm()) \ transpose(array(project_target * projected_KRP))
        #direction = qr(array(projected_KRP), ColumnNorm()) \ transpose(array(project_target))
        direction = nothing
        if als.additional_items[:normal]
            direction = ldiv_solve!!(expose(array(dag(projected_KRP * prime(dag(projected_KRP); tags=tags(rank))))), expose(transpose(array(project_target * prime(dag(projected_KRP); tags=tags(rank)))));factorizeA=true) 
        else
            direction = ldiv_solve!!(expose(array(projected_KRP)), expose(transpose(array(project_target)));factorizeA=true)
        end
        i = ind(project_target, 1)
        return itensor(copy(transpose(direction)), i,rank)
    end