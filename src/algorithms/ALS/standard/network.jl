using ITensors: Index
using ITensors.NDTensors: data
using ITensors.NDTensors.Expose: expose

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
        ## This contracts all the indices not of interest on the target vertice
        for x in uniqueinds(als.target, target_vert)
            if x == target_index
                continue
            end
            factor_ind = als.additional_items[:ext_ind_to_factor][x]
            p = had_contract(p, dag.(factors[factor_ind]), rank)
        end

        ## Next I need to figure out which partial hadamard_product to skip
        env_list =[
            p,
            (als.additional_items[:partial_mtkrp])[1:end .!= als.additional_items[:factor_to_part_cont][fact]]...,
        ]
        
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