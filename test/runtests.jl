using Test, Pkg
Pkg.develop(path = "$(@__DIR__)/../")

using ITensorCPD:
    ITensorCPD, als_optimize, direct, random_CPD, row_norm, reconstruct, had_contract
using ITensors: Index, ITensor, itensor, array, contract, dim, norm, random_itensor
using Test

include("./basic_features.jl")
include("./pivot_mapping.jl")
include("./standard_cpd.jl")
include("./itensor_network_cpd.jl")
include("./contract_algs.jl")
