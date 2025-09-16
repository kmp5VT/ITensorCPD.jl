using ITensors: ITensor
function reconstruct(cp::CPD)
    return reconstruct(cp.factors, cp.λ)
end

function reconstruct(factors::Vector{<:ITensor}, λ)
    ## TODO this is not efficient but does it matter?
    return λ * had_contract(factors, ind(λ, 1))
end
