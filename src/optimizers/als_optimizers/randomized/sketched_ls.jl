function compute_als(
    alg::SketchProjected,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    kwargs...
)
    C1_v = C1_vect(alg)
    C2_v = C2_vect(alg)
    targets = Vector{ITensor}()
    sketch = Vector{Matrix{Float64}}()
    for (i, n) in zip(inds(target), 1:length(cp))
        C1 = isnothing(C1_v) ? 3 : C1_v[n]
        C2 = isnothing(C2_v) ? 1 : C2_v[n]
        Ris = uniqueinds(target, i)         
        p = dim(Ris)
        m = dim(i)
        l=Int(round(C1 * m * log(m))) 
        s=Int(round(C2*log(m)))
        omega = sparse_sign_matrix(l,p,s)
        TP = sketched_matricization(target, n, omega')
        TP = itensor(TP,i,Index(l,"l"))
    push!(targets, TP)
    push!(sketch,omega)
    end
    extra_args[:target_transform] = targets
    extra_args[:sketch_matrices] = sketch
    
    return ALS(target, alg, extra_args, check)
end