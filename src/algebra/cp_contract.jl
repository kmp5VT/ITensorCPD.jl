using ITensorNetworks: ITensorNetwork, vertices

# Here we contract a tensor network with a CPD. We assume there
# exists some leges that connect the network to the CPD but it is
# also acceptable to have legs which do not contract with the network.
# the result is 2 vectors of tensors, the first is the set of tensors contracted_cps
# with the CPD and the second are the set of vectors from the CPD which do not connect to the network.
function tn_cp_contract(tn::ITensorNetwork, cp::CPD)
    tnp = copy(tn)
    r = cp_rank(cp)
    ## Go through all the nodes
    contracted_cps = Int[]
    for v in vertices(tnp)
        ## For each node find the indices that are not connected to other tensors in the node
        iss = uniqueinds(tnp, v)
        for is in iss
            cp_pos = findfirst(x -> x == is, inds(cp))

            isnothing(cp_pos) && continue
            tnp[v] = ITensorCPD.had_contract(cp[cp_pos], tnp[v], r)
            push!(contracted_cps, cp_pos)
        end
    end
    v = [cp[x] for x in contracted_cps]
    return tnp, CPD{paramT(cp)}(filter(x -> x ∉ v, cp.factors), cp.λ)
end

# This contracts two sets of CPD factor matrices. It will form a 
# resulting matrix that is rank of cp1 by rank of cp2. The function
# also returns the set of factor matrices from cp1 that do not connect to cp2
# and the factor matrices from cp2 that do not connect to cp1.
function cp_cp_contract(cp1::CPD, cp2::CPD)
    r1 = cp_rank(cp1)#ind(cp1[1], 1)
    r2 = cp_rank(cp2)#ind(cp2[1], 1)
    ## TODO check to see if eltypes are equivalent
    elt = eltype(cp1[1])
    inner = ITensor(elt, r1, r2)
    fill!(inner, one(elt))
    inner_pos_cp1, inner_pos_cp2 = Vector{Int}(), Vector{Int}()
    for i = 1:length(cp1)
        pos = findfirst(x -> x == ind(cp1, i), inds(cp2))
        if isnothing(pos)
            continue
        end
        data(inner) .*= data(cp1[i] * cp2[pos])
        push!(inner_pos_cp1, i)
        push!(inner_pos_cp2, pos)
    end
    v1 = cp1[inner_pos_cp1]
    v2 = cp2[inner_pos_cp2]
    return inner, CPD{paramT(cp1)}(filter(x -> x ∉ v1, cp1.factors), cp1.λ), CPD{paramT(cp2)}(filter(x -> x ∉ v2, cp2.factors), cp2.λ)
end
