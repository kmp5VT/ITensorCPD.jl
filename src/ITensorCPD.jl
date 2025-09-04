module ITensorCPD
  include("row_norm.jl")
  include("algorithms.jl")
  include("algebra/had_contract.jl")
  include("algebra/pivot_mapping.jl")
  include("algebra/SEQRCS.jl")
  include("cpd.jl")
  include("converge_checks.jl")
  include("optimizers/als.jl")
  include("optimizers/decompose.jl")
  include("reconstruct.jl")
  include("algebra/cp_contract.jl")

  function __init__()
    lib_ext = Sys.KERNEL == :Darwin ? ".dylib" :
          Sys.KERNEL == :Linux  ? ".so"  :
          Sys.KERNEL == :Windows ? ".dll" :
          error("Unsupported OS")

    if(!isfile(joinpath(@__DIR__, "algebra", "libsparse_sign" * lib_ext)))
      include("../deps/build.jl")
    end

    global libsparse = joinpath(@__DIR__, "algebra", "libsparse_sign" * lib_ext)
  end
end
