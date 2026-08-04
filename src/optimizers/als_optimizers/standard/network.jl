function compute_als(
    target::ITensorNetwork,
    cp::CPD{<:ITensorNetwork};
    alg = nothing,
    check = nothing,
    maxiter = nothing,
    kwargs...
)
    check = isnothing(check) ? NoCheck(isnothing(maxiter) ? 100 : maxiter) : check
    alg = isnothing(alg) ? network_solver() : alg
    verts = vertices(target)
    elt = eltype(target[first(verts)])
    cpRank = cp_rank(cp)

    partial_mtkrp = typeof(similar(cp.factors))()
    external_ind_to_vertex = Dict()
    extern_ind_to_factor = Dict()
    factor_number_to_partial_cont_number = Dict()

    # What we need to do here is walk through the CP and find where
    # Each factor connects to in the given target newtork.
    # This will assume all external legs are connected.
    factor_number = 1
    partial_cont_number = 1
    for v in verts
        partial = target[v]
        for uniq in uniqueinds(target, v)
            external_ind_to_vertex[uniq] = v
            factor_pos = findfirst(x -> x == uniq, inds(cp))
            factor = dag(cp.factors[factor_pos])
            partial = had_contract(partial, factor, cpRank)
            extern_ind_to_factor[uniq] = factor_pos
            factor_number_to_partial_cont_number[factor_pos] = partial_cont_number
            factor_number += 1
        end
        push!(partial_mtkrp, partial)
        partial_cont_number += 1
    end

    mttkrp_contract_sequences = Vector{Union{Any,Nothing}}()
    for _ in inds(cp)
        push!(mttkrp_contract_sequences, nothing)
    end

    als = ALS(
        target,
        alg,
        Dict(
            :part_grammian => cp.factors .* dag.(prime.(cp.factors; tags = tags(cpRank))),
            :partial_mtkrp => partial_mtkrp,
            :ext_ind_to_vertex => external_ind_to_vertex,
            :ext_ind_to_factor => extern_ind_to_factor,
            :factor_to_part_cont => factor_number_to_partial_cont_number,
            :mttkrp_contract_sequences => mttkrp_contract_sequences,
        ),
        check,
    )
    return als
end