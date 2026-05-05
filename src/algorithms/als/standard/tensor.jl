using ITensors: Index
using ITensors.NDTensors: data
using ITensors.NDTensors.Expose: expose

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


