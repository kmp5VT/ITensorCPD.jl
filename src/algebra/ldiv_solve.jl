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
    sol = nothing
    if factorizeA 
        szA = size(A)
        if (szA[1] == szA[2])
            try
                sol = cholesky(Hermitian(A), RowMaximum(), check=true, tol=cholesky_epsilon) \ B
            catch
                sol = qr(A, ColumnNorm()) \ B
            end
        else
            sol = qr(A, ColumnNorm()) \ B
        end
    else
        sol = A \ B
    end
    return sol
end