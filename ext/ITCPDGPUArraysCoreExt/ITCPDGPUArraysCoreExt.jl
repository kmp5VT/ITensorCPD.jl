module ITCPDGPUArraysCoreExt

using ITensors.NDTensors.Expose
using ITensorCPD: ITensorCPD
using GPUArraysCore: AbstractGPUArray

function ITensorCPD.ldiv_solve!!(A::Exposed{<:AbstractGPUArray}, B::Exposed{<:AbstractGPUArray}; factorizeA = false)
    return unexpose(A) \ unexpose(B)
end

end
