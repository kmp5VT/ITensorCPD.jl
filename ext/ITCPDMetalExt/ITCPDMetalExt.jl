module ITCPDMetalExt

using ITensors: Index, array, dag, itensor, ind, svd
using ITensors.NDTensors.Expose
using ITensorCPD: ITensorCPD
using Metal: MtlArray
using LinearAlgebra: pinv

function ITensorCPD.ldiv_solve!!(A::Exposed{<:MtlArray}, B::Exposed{<:MtlArray}; factorizeA = false)
    uA = unexpose(A)
    itA = itensor(uA, Index.(size(uA)))
    U, S, V = svd(dag(itA), ind(itA, 2); use_absolute_cutoff = true, cutoff = 0)
    return array(V *  (1 ./ S) * U)' * unexpose(B)
end

end