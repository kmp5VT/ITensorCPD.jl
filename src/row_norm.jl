using ITensors: ITensor
function row_norm(t::ITensor, i...)
  λ = t .^ 2
  for is in tuple(i...)
    λ = λ * delta(is)
  end
  λ = sqrt.(λ)
  l_array = copy(λ)
  for is in tuple(i...)
    l_array = l_array * delta(is)
  end
  return itensor(array(t) ./ array(permute(l_array, inds(t); allow_alias=true)), inds(t)), λ
end
