using LinearAlgebra: ColumnNorm, diagm
function als_optimize(cp::CPD, rank::Index; maxiters=1, kwargs...)
  return als_optimize(cp, rank, NoCheck(maxiters); kwargs...)
end

function als_optimize(cp::CPD, rank::Index, converge)
  iter = 0
  part_grammian = cp.factors .* prime.(cp.factors; tags=tags(rank))
  num_factors = length(cp.factors)
  λ = copy(cp.λ)
  factors = copy(cp.factors)
  while iter < converge.max_counter
    mtkrp = nothing
    for fact in 1:num_factors
      ## compute the matrized tensor time khatri rao product with a provided algorithm.
      mtkrp = mttkrp(cp.mttkrp_alg, factors, cp, rank, fact)

      ## compute the grammian which requires the hadamard product
      grammian = similar(part_grammian[1])
      fill!(grammian, one(eltype(cp)))
      for i in 1:num_factors
        if i == fact
          continue
        end
        grammian = hadamard_product(grammian, part_grammian[i])
      end

      ## potentially better to first inverse the grammian then contract
      ## qr(A, Val(true))
      factors[fact], λ = row_norm(
        itensor(qr(array(grammian), ColumnNorm()) \ array(mtkrp), inds(mtkrp)), ind(mtkrp, 2)
      )
      part_grammian[fact] = factors[fact] * prime(factors[fact]; tags=tags(rank))

      post_solve(cp.mttkrp_alg, factors, λ, cp, rank, fact)
    end

    # potentially save the MTTKRP for the loss function
    save_mttkrp(converge, mtkrp)

    if check_converge(converge, factors, λ, part_grammian)
      break
    end
    iter += 1
  end

  return CPD(cp.target, factors, λ, cp.mttkrp_alg, cp.additional_items)
end

function als_direct_optimize(cp::CPD, rank::Index, converge)
  iter = 0
  part_grammian = cp.factors .* prime.(cp.factors; tags=tags(rank))
  num_factors = length(cp.factors)
  λ = copy(cp.λ)
  factors = copy(cp.factors)
  while iter < converge.max_counter
    mtkrp = nothing
    for fact in 1:num_factors
      ## compute the matrized tensor time khatri rao product with a provided algorithm.
      mtkrp = mttkrp(cp.mttkrp_alg, factors, cp, rank, fact)

      ## compute the grammian which requires the hadamard product
      grammian = similar(part_grammian[1])
      fill!(grammian, one(eltype(cp)))
      for i in 1:num_factors
        if i == fact
          continue
        end
        grammian = hadamard_product(grammian, part_grammian[i])
      end

      w = had_contract(factors[1:end .!= fact], rank)
      col_dim = prod(size(w)[2:end])
      ## potentially better to first inverse the grammian then contract
      ## qr(A, Val(true))
      winv = pseudoinverse(w, inds(w)[2:end])
      factors[fact], λ = row_norm(
        winv * cp.target, ind(mtkrp, 2)
      )
      part_grammian[fact] = factors[fact] * prime(factors[fact]; tags=tags(rank))

      post_solve(cp.mttkrp_alg, factors, λ, cp, rank, fact)
    end

    # potentially save the MTTKRP for the loss function
    save_mttkrp(converge, mtkrp)

    if check_converge(converge, factors, λ, part_grammian)
      break
    end
    iter += 1
  end

  return CPD(cp.target, factors, λ, cp.mttkrp_alg, cp.additional_items)
end
