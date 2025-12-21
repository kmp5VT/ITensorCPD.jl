using ITensors: diag_itensor, hadamard_product!
using ITensors.NDTensors: tensor
abstract type ConvergeAlg end

function norm_factors(partial_gram::Vector, λ::ITensor)
    had = copy(partial_gram[1])
    for i = 2:length(partial_gram)
        hadamard_product!(had, had, partial_gram[i])
    end
    return real(had*(λ*dag(prime(λ))))[]
end

save_mttkrp(::ConvergeAlg, ::ITensor) = nothing

include("no_check.jl")
include("fit_check.jl")
include("cp_diff_check.jl")
include("cp_angle_check.jl")
