using ITensors: ITensor, Index
function had_contract(A::ITensor, B::ITensor, had::Index; α=true)
  if had ∉ commoninds(A, B)
    return α .* (A * B)
  else
    position_of_had_A = findfirst(x -> x == had, inds(A))
    position_of_had_B = findfirst(x -> x == had, inds(B))
    slices_A = eachslice(array(A); dims=position_of_had_A)
    slices_B = eachslice(array(B); dims=position_of_had_B)

    @assert length(slices_A) == length(slices_B)
    inds_c = noncommoninds(A, B)
    elt = promote_type(eltype(A), eltype(B))
    C = ITensor(elt, vcat(had, inds_c...))
    ## Right now I have to fill C with zeros because I hate empty tensor
    fill!(C, zero(elt))
    slices_C = eachslice(array(C); dims=1)
    for i in 1:length(slices_A)
      a_inds = [ind(A, x) for x in 1:ndims(A) if x != position_of_had_A]
      b_inds = [ind(B, x) for x in 1:ndims(B) if x != position_of_had_B]
      ITensors.contract!(
        itensor(slices_C[i], inds_c),
        itensor(slices_A[i], a_inds),
        itensor(slices_B[i], b_inds),
        elt(α),
      )
    end
    return C
  end
end

## TODO this is broken when some items have a rank but others do not.
function had_contract(tensors::Vector{<:ITensor}, had::Index; α=true, sequence=nothing)
  had_tensors = Vector{ITensor}([])
  no_had = Vector{ITensor}([])
  for ten in tensors
    if had ∉ inds(ten)
      push!(no_had, ten)
      continue
    end
    push!(had_tensors, ten)
  end
  # if !all(x -> (had ∉ inds(x)) == 0, tensors) 
  #   had_contract([contract(tensors[1], tensors[2]), tensors[3:end]...], had; α, sequence)
  # end
  positions_of_had = Dict(y => (findfirst(x -> x == had, inds(y))) for y in had_tensors)
  slices = [eachslice(array(x); dims=positions_of_had[x]) for x in had_tensors]
  slices_inds = [inds(x)[1:end .!= positions_of_had[x]] for x in had_tensors]
  # inds_c = noncommoninds(A, B)
  # C = ITensor(elt, vcat(had, inds_c...))
  ## TODO could be a better way to contract these given a sequence?
  cslice =
    α .* contract([itensor(slices[x][1], slices_inds[x]) for x in 1:length(had_tensors)])
  # ## Right now I have to fill C with zeros because I hate empty tensor
  C = ITensor(zeros(eltype(cslice), dim(had) * dim(cslice)), (had, inds(cslice)...))
  slices_c = eachslice(array(C); dims=1)
  slices_c[1] .= cslice

  ## TODO would be better to do in place but can't do a list of tensors in place right now
  for i in 2:dim(had)
    slices_c[i] .= array(
      α .* contract([itensor(slices[x][i], slices_inds[x]) for x in 1:length(had_tensors)])
    )
  end

  return isempty(no_had) ? C : contract([C, no_had...])
end
