using ITensors: ITensor, hadamard_product
using ITensors.NDTensors: NDTensors

function row_norm(t::ITensor, i...)
    elt = eltype(t)
    dataT = NDTensors.datatype(t)
    λ = hadamard_product(t, t)
    for is in tuple(i...)
        d = itensor(NDTensors.Diag(dataT(ones(Float32, dim(is)))), is)
        λ = λ * d
    end
    map!(i -> sqrt(i), data(λ), data(λ))
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
