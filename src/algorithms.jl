using ITensors: Index
using ITensors.NDTensors: data

abstract type MttkrpAlgorithm end
struct KRP <: MttkrpAlgorithm end

## This version assumes we have the exact target and can form the tensor
## This forms the khatri-rao product for a single value of r and immediately
## contracts it with the target tensor. This is relatively expensive because the KRP will be
## order $d - 1$ where d is the number of modes in the target tensor.
## This process could be distributed.
function mttkrp(::KRP, als, factors, cp, rank::Index, fact::Int)

    factor_portion = factors[1:end .!= fact]
    sequence = ITensors.default_sequence()
    krp = had_contract(dag.(factor_portion), rank; sequence)

    m = krp * als.target
    return m
end

function post_solve(::KRP, als, factors, λ, cp, rank::Index, fact::Integer) end

struct direct <: MttkrpAlgorithm end

## This code skips computing the khatri-rao product by incrementally 
## contracting the factor matrices into the tensor for each value of r
## This process could be distributed.
function mttkrp(::direct, als, factors, cp, rank::Index, fact::Int)
    factor_portion = factors[1:end .!= fact]
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

function post_solve(::direct, als, factors, λ, cp, rank::Index, fact::Integer) end

################
## This solver is based on ITensorNetwork
## It allows one to take a completely connected 
## ITensorNetwork and decomposes it into a CPD
## You can't currently take a network which is and
## outer product of two networks

struct network_solver <: MttkrpAlgorithm end

function mttkrp(::network_solver, als, factors, cp, rank::Index, fact::Int)
    m = similar(factors[fact])

    target_index = ind(factors[fact], 2)
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
    ## Once done with all factor which connect to it, then go through uniqueinds and contract in the 
    ## associated new factors
    partial_ind = als.additional_items[:factor_to_part_cont][fact]
    if fact == length(factors) ||
       als.additional_items[:factor_to_part_cont][fact+1] != partial_ind
        ## go through factors
        partial_vertex = als.additional_items[:ext_ind_to_vertex][ind(factors[fact], 2)]
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
