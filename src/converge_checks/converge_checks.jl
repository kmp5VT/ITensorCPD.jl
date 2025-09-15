using ITensors: diag_itensor, hadamard_product!
using ITensors.NDTensors: tensor
abstract type ConvergeAlg end

include("no_check.jl")
include("fit_check.jl")
