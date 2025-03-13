using ITensors: ITensor
using ITensors.NDTensors: NDTensors

function row_norm(t::ITensor, i...)
    elt = eltype(t)
    dataT = NDTensors.datatype(t)
    λ = t .^ elt(2)
    for is in tuple(i...)
        d = itensor(NDTensors.Diag(dataT(ones(Float32, dim(is)))), is)
        λ = λ * d
    end
    λ = sqrt.(λ)
    l_array = copy(λ)
    for is in tuple(i...)
        d = itensor(NDTensors.Diag(dataT(ones(elt, dim(is)))), is)
        l_array = l_array * d
    end
    return itensor(
        array(t) ./ array(permute(l_array, inds(t); allow_alias = true)),
        inds(t),
    ),
    λ
end
