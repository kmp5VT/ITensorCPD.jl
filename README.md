# ITensorCPD.jl

A higher-order tensor decomposition library which supports the canonical polyadic decomposition (CPD) of tensors and tensor-networks.
The code also supports non-covarient contractions which occur in multilinear algebra based algorithms.
The CPD takes advantage of the "smart" indexing system from [ITensors.jl](https://github.com/ITensor/ITensors.jl) and
the tensor-network algorithms developed in [ITensorNetworks.jl](https://github.com/ITensor/ITensorNetworks.jl).

The goal of this code is to support the decomposition of distributed CPU and GPU tensors natively via the backend support of ITensors.jl.

** Warning ** This project is not currently a published package with Julia and is not an official package of ITensors.jl.
To run this project please download the repo from github using Julia package manager's `add` or `develop` functionality.

```julia
julia> using Pkg
julia> Pkg.develop(url="https://github.com/kmp5VT/ITensorCPD.jl")
julia> using ITensorCPD
```

Here is a quick example to decompose a random tensor 
```julia
# Load ITensor and ITensorCPD
julia> using ITensorCPD, ITensors

# Make an ITensor to decompose
julia> i,j,k = Index.((10,10,10))
julia> T = randomITensor(Float64, i,j,k)

# Ask ITensorCPD to decompose the tensor to rank 50
julia> cpd = ITensorCPD.decompose(T, 50);

# Reconstruct the CPD from it's computed factor matrices and check the fit of the decomposition.
julia> diff = ITensorCPD.reconstruct(cpd) - T
julia> println("The Accuracy in the rank 50 CPD is: $(1.0 - norm(diff) / norm(T))")
The Accuracy in the rank 50 CPD is: 0.9999960710854621
```

There are currently two convergence criteria implemented. The first does no check and simply runs a certain number of iterations.
And the second looks at the change in the L2 fit of the tensor decomposition. An example using this fit based criteria is
```julia
# Construct an ITensorCPD convergence check object using the L2 fit.
# This takes the ALS stopping condition epsilon, the number of allowed iterations and 
# the norm of the reference tensor
julia> check = ITensorCPD.FitCheck(1e-3, 100, norm(T))

# One can pass CPD objects into ITensorCPD by calling the `als_optimize`
# function which optimized the given CPD using a standard alternating least squares algorithm.
julia> cpd = ITensorCPD.als_optimize(T, cpd; check, verbose=true);
50       1       0.9999946084173382      0.9999946084173382
50       2       0.9999950065419643      3.981246260442717e-7
50       3       0.9999953749621856      3.684202213305454e-7

# We can also construct a new CPD and pass this into `als_optimize`
julia> cpd_random = ITensorCPD.random_CPD(T, 50);
julia> ITensorCPD.als_optimize(T, cpd_random; check, verbose=true);
50       1       0.6382083068797204      0.6382083068797204
50       2       0.8136191816721784      0.17541087479245798
50       3       0.8716713681765458      0.05805218650436739
50       4       0.902153182574952       0.03048181439840625
50       5       0.9224768607969728      0.020323678222020747
50       6       0.9374218337536796      0.014944972956706826
50       7       0.9485161412080121      0.011094307454332486
50       8       0.956764530142137       0.00824838893412494
50       9       0.9630234519485501      0.006258921806413076
50       10      0.967916732355913       0.004893280407362921
50       11      0.971855634925375       0.003938902569461944
50       12      0.9751056265582695      0.003249991632894522
50       13      0.9778409951953222      0.002735368637052704
50       14      0.9801796336710052      0.0023386384756830525
50       15      0.9822039174208834      0.002024283749878175
50       16      0.9839732373588355      0.0017693199379520408
50       17      0.985531661503246       0.0015584241444105418
50       18      0.9869127390350924      0.001381077531846353
50       19      0.9881426023129034      0.0012298632778110496
50       20      0.9892420307807247      0.001099428467821295
50       21      0.9902278617267223      0.000985830945997579
50       22      0.9911139759293625      0.0008861142026401758
```
