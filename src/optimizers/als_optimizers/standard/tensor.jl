## Default constructor algorithms for normal equation based solvers (MttkrpAlgorithm).
## This needs no extra information and passes the function `optimize` as the optimizer algorithm.
function compute_als(
    alg::MttkrpAlgorithm,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    kwargs...
)
    cprank = cp_rank(cp)
    extra_args[:part_grammian] = cp.factors .* dag.(prime.(cp.factors; tags = tags(cprank)))
    return ALS(target, alg, extra_args, check)
end

function compute_als(
    alg::TargetDecomp,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    kwargs...
)
    cprank = cp_rank(cp)
    extra_args[:part_grammian] = cp.factors .* dag.(prime.(cp.factors; tags = tags(cprank)))
    decomps = Vector{ITensor}()
    targets = Vector{ITensor}()
    for i in inds(target)
        u, s, v = svd(target, i; use_relative_cutoff = false, cutoff = 0)
        push!(decomps, v)
        t = u * s
        push!(targets, t);
    end
    extra_args[:target_decomps] = decomps
    extra_args[:target_transform] = targets

    return ALS(target, alg, extra_args, check)
end