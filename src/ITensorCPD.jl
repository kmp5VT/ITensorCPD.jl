module ITensorCPD
include("row_norm.jl")
include("algorithms.jl")
include("algebra/pseudoinverse.jl")
include("algebra/had_contract.jl")
include("cpd.jl")
include("converge_checks.jl")
include("optimizers/als.jl")
include("optimizers/decompose.jl")
include("reconstruct.jl")
include("algebra/cp_contract.jl")
end
