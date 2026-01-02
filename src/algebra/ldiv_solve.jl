using ITensors.NDTensors.Expose
using GPUArraysCore: AbstractGPUArray

function ldiv_solve!!(A::Exposed, B::Exposed; factorizeA = false)
    return ldiv_solve!(unexpose(A), unexpose(B); factorizeA)
end

function ldiv_solve!!(A::Exposed{<:AbstractGPUArray}, B::Exposed{<:AbstractGPUArray}; factorizeA = false)
    return ldiv_solve!(A,B; factorizeA = false)
end

## For now just call with QR CP if factorize. Later this will be more complex.
function ldiv_solve!(A, B; factorizeA = false)
    if factorizeA 
        szA = size(A)
        if (szA[1] == szA[2])
            try
                return cholesky(Hermitian(A), RowMaximum(), check=true, tol=cholesky_epsilon) \ B
            catch
                # println("Warning: Cholesky based solver failed.")
                return qr(A, ColumnNorm()) \ B
            end
        else
            return qr(A, ColumnNorm()) \ B
        end
    else
        return A \ B
    end
end