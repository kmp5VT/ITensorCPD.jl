using ITensors: Index
using ITensors.NDTensors: data

abstract type MttkrpAlgorithm end
abstract type ProjectionAlgorithm end

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

    m = als.target * krp
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

### This solver is slightly silly. It takes a higher-order tensor T forms the SVD of every unfolding
### for example T(a,b,c) => T(a,bc) => U(a,r) S(r,rp) V(rp,bc). We use this decomposition to represent T as
### T(a,b'c') V(r,b'c') V(r,bc). Then we solve f(A) = || T(a,b'c') V(r,b'c') V(r,bc) - A(a,m) (B(b,m) ⊙ C(c,m)) ||².
### We repeat this process for every factor matrix. This gains us a smaller target tensor to store,
### i.e. (U(a,r) S(r,rp)) for each mode but solving the least squares problem is no less expensive.

struct TargetDecomp <: MttkrpAlgorithm end

function mttkrp(::TargetDecomp, als, factors, cp, rank::Index, fact::Int)
    factor_portion = factors[1:end .!= fact]
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

function post_solve(::TargetDecomp, als, factors, λ, cp, rank::Index, fact::Integer) end


### With this solver we are going to compute sampling projectors for LS decomposition
### based on the leverage score of the factor matrices. Then we are going to solve a
### sampled least squares problem 

struct LevScoreSampled{NSamples} <: ProjectionAlgorithm end

LevScoreSampled() = LevScoreSampled{(0,)}()
LevScoreSampled(n::Int) = LevScoreSampled{(n,)}()
LevScoreSampled(n::Tuple) = LevScoreSampled{n}()

nsamples(::LevScoreSampled{N}) where {N} = N

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

function project_target(::LevScoreSampled, als, factors, cp, rank::Index, fact::Int, krp)
    ## I need to turn this into an ITensor and then pass it to the computed algorithm.
    return fused_flatten_sample(als.target, fact, als.additional_items[:pivot_tensors][fact])
end

function solve_ls_problem(::LevScoreSampled, projected_KRP, project_target, rank)
    # direction = qr(array(projected_KRP * prime(projected_KRP, tags=tags(rank))), ColumnNorm()) \ transpose(array(project_target * projected_KRP))
    direction = qr(array(projected_KRP), ColumnNorm()) \ transpose(array(project_target))
    i = ind(project_target, 1)
    return itensor(copy(transpose(direction)), i,rank)
end

function post_solve(::LevScoreSampled, als, factors, λ, cp, rank::Index, fact::Integer) 
## update the factor weights.
    als.additional_items[:factor_weights][fact] = compute_leverage_score_probabilitiy(factors[fact], ind(cp, fact))
end

### With this solver we are going to compute sampling projectors for LS decomposition
### based on the leverage score of the factor matrices. Then we are going to solve a
### sampled least squares problem. To make the sampling process more efficient this algorithm
### gathers samples in blocks

struct BlockLevScoreSampled{NSamples, Blocks} <: ProjectionAlgorithm end

BlockLevScoreSampled() = BlockLevScoreSampled{(0,), (1,)}()
BlockLevScoreSampled(n::Int) = BlockLevScoreSampled{(n,), (1,)}()
BlockLevScoreSampled(n::Tuple) = BlockLevScoreSampled{n, (1,)}()

BlockLevScoreSampled(n::Int, m::Int) = BlockLevScoreSampled{(n,), (m,)}()
BlockLevScoreSampled(n::Tuple, m::Tuple) = BlockLevScoreSampled{n, m}()

nsamples(::BlockLevScoreSampled{N}) where {N} = N
blocks(::BlockLevScoreSampled{N,M}) where {N, M} = M

## We are going to construct a matrix of sampled indices of the tensor
function project_krp(::BlockLevScoreSampled, als, factors, cp, rank::Index, fact::Int)
    nsamps = nsamples(als.mttkrp_alg)
    nsamps = length(nsamps) == 1 ? nsamps[1] : nsamps[fact]
    block_size = blocks(als.mttkrp_alg)
    block_size = length(block_size) == 1 ? block_size[1] : block_size[fact]

    size_fast = fact == 1 ? dim(ind(cp, 2)) : dim(ind(cp, 1))
    sampled_cols = block_sample_factor_matrices(nsamps, als.additional_items[:factor_weights], block_size, fact, size_fast)
    ## Write new samples to pivot tensor
    dRis = dims(inds(cp)[1:end .!= fact])
    data(als.additional_items[:pivot_tensors][fact]) .= multi_coords_to_column(dRis, sampled_cols)
    
    return pivot_hadamard(factors, rank, sampled_cols, inds(als.additional_items[:pivot_tensors][fact])[end])
end

function project_target(::BlockLevScoreSampled, als, factors, cp, rank::Index, fact::Int, krp)
    ## I need to turn this into an ITensor and then pass it to the computed algorithm.
    return fused_flatten_sample(als.target, fact, als.additional_items[:pivot_tensors][fact])
end

function solve_ls_problem(::BlockLevScoreSampled, projected_KRP, project_target, rank)
    direction = qr(array(projected_KRP), ColumnNorm()) \ transpose(array(project_target))
    i = ind(project_target, 1)
    return itensor(copy(transpose(direction)), i,rank)
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
struct QRPivProjected{Start,End} <: ProjectionAlgorithm end

## TODO modify to use ranges 
QRPivProjected() = QRPivProjected{(1,),(0,)}()
QRPivProjected(n::Int) = QRPivProjected{(1,),(n,)}()
QRPivProjected(n::Int, m::Int) = QRPivProjected{(n,),(m,)}()
QRPivProjected(n::Tuple) = QRPivProjected{Tuple(Int.(ones(length(n)))),n}()
QRPivProjected(n::Tuple, m::Tuple) = QRPivProjected{n,m}()

start(::QRPivProjected{N}) where {N} = N
stop(::QRPivProjected{N,M}) where {N,M} = M

function project_krp(::QRPivProjected, als, factors, cp, rank::Index, fact::Int)
    krp_piv = ITensorCPD.pivot_hadamard(factors, rank, als.additional_items[:projects_tensors][fact])
    return krp_piv
end

function project_target(::QRPivProjected, als, factors, cp, rank::Index, fact::Int, krp)
    return als.additional_items[:target_transform][fact]
end

function solve_ls_problem(::QRPivProjected, projected_KRP, project_target, rank)
    direction = qr(array(projected_KRP), ColumnNorm()) \ transpose(array(project_target))
    i = ind(project_target, 1)
    return itensor(copy(transpose(direction)), i,rank)
end

function post_solve(::QRPivProjected, als, factors, λ, cp, rank::Index, fact::Integer) end


## This solver does not form the normal equations. 
## We simply compute the khatri rao product and directly compute Ax=B for each least squres problem.
struct InvKRP <: ProjectionAlgorithm end

function project_krp(::InvKRP, als, factors, cp, rank::Index, fact::Int)
    return had_contract(factors, rank)
end
function project_target(::InvKRP, als, factors, cp, rank::Index, fact::Int, krp)
    return als.target
end

function solve_ls_problem(::InvKRP, projected_KRP, projected_target, rank)
    U, S, V = svd(dag(projected_KRP), rank; use_absolute_cutoff = true, cutoff = 0)
    return (U * (prime(projected_target; tags = tags(rank)) * V * (1 ./ S)))
end

function post_solve(::InvKRP, als, factors, λ, cp, rank::Index, fact::Integer) end

################
## This solver is based on ITensorNetwork
## It allows one to take a completely connected 
## ITensorNetwork and decomposes it into a CPD
## You can't currently take a network which is and
## outer product of two networks

struct network_solver <: MttkrpAlgorithm end

function mttkrp(::network_solver, als, factors, cp, rank::Index, fact::Int)
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
