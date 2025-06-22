@testset "Norm Row test, elt=$elt" for elt in [Float32, Float64, ComplexF32, ComplexF64]
  i, j = Index.((20, 30))
  A = random_itensor(elt, i, j)
  Ainorm, lam = row_norm(A, i)
  for id = 1:dim(i)
      @test real(one(elt)) ≈ sum(array(Ainorm .^ 2)[:, id])
  end

  Ajnorm, lam = row_norm(A, j)
  for id = 1:dim(i)
      @test real(one(elt)) ≈ sum(array(Ajnorm .^ 2)[id, :])
  end

  Aijnorm, lam = row_norm(A, i, j)
  @test real(one(elt)) ≈ sum(array(Aijnorm .^ 2))
end

@testset "Hadamard contract algorithm, elt=$elt" for elt in [Float32, Float64, ComplexF32, ComplexF64]
  i, j, k = Index.((20, 30, 40))
  r = Index(400, "CP_rank")
  A = random_itensor(elt, i, j, k)

  cp = random_CPD(A, r);

  Dhad = had_contract(cp[1],cp[2], r)

  Dex = ITensor(elt, r,i,j)
  for n in 1:400
      for l in 1:20
          for m in 1:30
              Dex[n,l,m] = cp[1][n,l] * cp[2][n,m]
          end
      end
  end

  @test norm(Dex - Dhad) / norm(Dex) < 1e-7
  
  v = Array(transpose(array(cp[2])))
  B = itensor(v, j,r)

  had_contract(cp[1], B, r)

  Dex = ITensor(elt, r,i,j)
  for n in 1:400
      for l in 1:20
          for m in 1:30
              Dex[n,l,m] = cp[1][n,l] * B[m,n]
          end
      end
  end

  @test norm(Dex - Dhad) / norm(Dex) < 1e-7
end

@testset "reconstruct, elt=$elt" for elt in [Float32, Float64, ComplexF32, ComplexF64]
  i, j = Index.((20, 30))
  r = Index(10, "CP_rank")
  A, B = random_itensor.(elt, ((r, i), (r, j)))
  λ = ITensor(randn(elt, dim(r)), r)
  exact = fill!(ITensor(elt, i, j), zero(elt))
  for R = 1:dim(r)
      for I = 1:dim(i)
          for J = 1:dim(j)
              exact[I, J] += λ[R] * A[R, I] * B[R, J]
          end
      end
  end
  recon = reconstruct([A, B], λ)
  @test 1.0 - norm(array(exact - recon)) / norm(exact) ≈ 1.0 rtol = eps(real(elt))

  k = Index.(40)
  A, B, C = random_itensor.(elt, ((r, i), (r, j), (r, k)))
  λ = ITensor(randn(elt, dim(r)), r)
  exact = fill!(ITensor(elt, i, j, k), zero(elt))
  for R = 1:dim(r)
      for I = 1:dim(i)
          for J = 1:dim(j)
              for K = 1:dim(k)
                  exact[I, J, K] += λ[R] * A[R, I] * B[R, J] * C[R, K]
              end
          end
      end
  end
  recon = reconstruct([A, B, C], λ)

  @test 1.0 - norm(array(exact - recon)) / norm(exact) ≈ 1.0 rtol = eps(real(elt))
end