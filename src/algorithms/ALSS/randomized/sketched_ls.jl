using ITensors: Index
using ITensors.NDTensors: data
using ITensors.NDTensors.Expose: expose


## This solver is for computing Tomega = A(B ododt C)omega 
## The c1 and C2x vectors are constant vectors for determing the 
## Sketching dimension and sparsity parameter.
struct SketchProjected <: ProjectionAlgorithm
    C1_vect
    C2_vect
end
    SketchProjected()=SketchProjected(nothing,nothing)

    C1_vect(alg::SketchProjected) = alg.C1_vect
    C2_vect(alg::SketchProjected) = alg.C2_vect


    function project_krp(::SketchProjected, als, factors, cp, rank::Index, fact::Int)
        return ITensorCPD.omega_hadamard(factors, rank, als.additional_items[:sketch_matrices][fact])
    end

    function matricize_tensor(:: SketchProjected, als, factors, cp, rank::Index, fact::Int)
        return als.additional_items[:target_transform][fact]
    end


    function post_solve(::SketchProjected, als, factors, λ, cp, rank::Index, fact::Integer) end