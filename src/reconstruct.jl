using ITensors: ITensor
function reconstruct(cp::CPD)
    return reconstruct(cp.factors, cp.λ)
end

function reconstruct(factors::Vector{<:ITensor}, λ)
    facs = copy(factors)
    ## Scale the original tensor by the scaling factor lambda
    ## TODO this doesn't actually work as I think it does, it makes a square matrix with repeated rows
    ## then uses dispatch to do an elementwise multiplication. Should switch it with diagITensor code to 
    ## make use of gemm.
    λ = λ * delta(ind(facs[1], 2))
    facs[1] = itensor(array(facs[1]) .* array(λ), inds(facs[1]))

    ## loop through every value of rank and contract the component 
    ## vectors together.
    ## starting with the first rank value, its cheaper to do this
    ## than to form an empty tensor.
    its = map(x -> itensor(array(x)[1, :], inds(x)[2]), facs)
    it = contract(its)
    for r = 2:dim(facs[1], 1)
        it .+= contract(map(x -> itensor(array(x)[r, :], ind(x, 2)), facs))
    end
    return it
end
