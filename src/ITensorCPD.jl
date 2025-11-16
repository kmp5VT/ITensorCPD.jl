module ITensorCPD
  include("math_tools/row_norm.jl")
  include("converge_checks/converge_checks.jl")
  include("algebra/had_contract.jl")
  include("algebra/pivot_mapping.jl")
  include("math_tools/probability.jl")
  include("algebra/SEQRCS.jl")
  include("algebra/ldiv_solve.jl")

  include("cpd.jl")
  include("algorithms.jl")
  include("optimizers/ALS/als.jl")
  
  include("decompose.jl")
  
  include("algebra/reconstruct.jl")
  include("algebra/cp_contract.jl")

  function __init__()
    lib_ext = Sys.KERNEL == :Darwin ? ".dylib" :
          Sys.KERNEL == :Linux  ? ".so"  :
          Sys.KERNEL == :Windows ? ".dll" :
          error("Unsupported OS")

    if(!isfile(joinpath(@__DIR__, "algebra", "libsparse_sign" * lib_ext)))
      include("$(@__DIR__)/../deps/build.jl")
    end

    global libsparse = joinpath(@__DIR__, "algebra", "libsparse_sign" * lib_ext)
  end
end
