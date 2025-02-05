using LinearAlgebra: svd

function pseudoinverse(T::ITensor, left_ind; tol = 1e-12)
  U, S, V = svd(T, left_ind)

  s = dim(ind(S, 1))
  elt = eltype(T)
  for i in 1:s
    S[i,i] = (S[i,i] > tol ? one(elt) / S[i,i] : S[i,i])
  end

  return U * S * V
end