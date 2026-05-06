using ITensors: Index
using ITensors.NDTensors: data
using ITensors.NDTensors.Expose: expose

### This solver does not form the normal equations. 
### We simply compute the khatri rao product and directly compute Ax=B for each least squres problem.
struct InvKRP <: ProjectionAlgorithm end

    function project_krp(::InvKRP, als, factors, cp, rank::Index, fact::Int)
        return had_contract(factors, rank)
    end
    function matricize_tensor(::InvKRP, als, factors, cp, rank::Index, fact::Int)
        return als.target
    end

    function solve_ls_problem(::InvKRP, _, projected_KRP, projected_target, rank)
        U, S, V = svd(dag(projected_KRP), rank; use_absolute_cutoff = true, cutoff = 0)
        return prime(projected_target; tags = tags(rank)) * V * (1 ./ S) * U
    end

    function post_solve(::InvKRP, als, factors, λ, cp, rank::Index, fact::Integer) end