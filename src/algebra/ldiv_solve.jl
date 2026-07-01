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
    # MeanDag = mean(diag(A))
    if factorizeA 
        szA = size(A)
        if (szA[1] == szA[2])
            evals = eigvals(A)
            MeanDag = sum(evals)
            (A) .*= 1/(MeanDag)
            Bscale = B .* 1/(MeanDag)
            try
                return cholesky(Hermitian(A), RowMaximum(), check=true, tol=cholesky_epsilon) \ Bscale
            catch
                # println("Warning: Cholesky based solver failed.")
                evals = svdvals(A)
                MeanDag = sum(evals)
                (A) .*= 1/(MeanDag)
                Bscale = B .* 1/(MeanDag)
                return qr(A, ColumnNorm()) \ Bscale
            end
        else
            evals = svdvals(A)
            MeanDag = sum(evals)
            (A) .*= 1/(MeanDag)
            Bscale = B .* 1/(MeanDag)
            return qr(A, ColumnNorm()) \ Bscale
        end
    else
        evals = svdvals(A)
        MeanDag = sum(evals)
        (A) .*= 1/(MeanDag)
        Bscale = B .* 1/(MeanDag)
        return A \ Bscale
    end
end