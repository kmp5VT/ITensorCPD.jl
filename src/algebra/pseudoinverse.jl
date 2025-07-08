using LinearAlgebra: svd

function pseudoinverse(T::ITensor, left_ind; tol = 1e-12)
    U, S, V = svd(T, left_ind; use_absolute_cutoff = true, cutoff = tol)

    return U * (1 ./ S) * V
end
