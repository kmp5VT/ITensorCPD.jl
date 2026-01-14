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

Here is a quick example to decompose a tensor 
```julia
# Load ITensor and ITensorCPD
julia> using ITensorCPD, ITensors

# Make an ITensor to decompose
julia> i,j,k = Index.((10,10,10))
julia> T = randomITensor(Float64, i,j,k)

# Call decompose to CPD the tensor to rank 50 with default options
julia> cpd = ITensorCPD.decompose(T, 50)
```
Currently the default options are to use the standard normal-equation based ALS algorithm and run 100 ALS iterations.
All these options can be fine-tuned.

If you have data which is represented as an N dimensional array, ITensorCPD can also decompose this directly
```julia
# Make a random order 3 array
julia> A = randn(10, 10, 10)

## Directly decompose this
julia> cpd = ITensorCPD.decompose(A, 50)
```
Note that the return type is a ITensorCPD.CPD object and factor matrices will be written into ITensor objects.
The factor matrices can be indexed in the CPD object using the `[]` operator
```julia
# Grab the first factor of cpd
julia> Fact1 = cpd[1]

# The scaling factor is indexed by
julia> λ = cpd[]
```
One can also easily reconstruct the CPD into a tensor using the reconstruct function
```julia
# Reconstruct approximation of A. A will be returned in an itensor type.
julia> Acp_itensor = ITensorCPD.reconstruct(A)

# To convert back into an array call array function
julia> Acp_array = array(Acp_itensor)
```

One can modify the initial guess of the decomposition by either passing a random number generator into decompose
```julia
# Make a random number generator
julia> using Random
julia> rng = RandomDevice()
julia> cpd = ITensorCPD.decompose(A, 50; rng)
```
Alternatively, one can manually construct a CPD type using the `random_CPD` function
```julia
# Make a random CPD of rank 50. Random guess generator also accepts a random number generators.
julia> init_guess = ITensorCPD.random_CPD(A, 50; rng)
```
The CPD type can be passed along to the decomposition through `optimize` functions.
```julia
# Optimize the CPD initial guess using the ALS
julia> cpd = ITensorCPD.als_optimize(A, init_guess)
```

ITensorCPD supports decomposition of tensors on both CPU and GPU. To decompose a tensor on GPU, simply move the tensor to the GPU
device and call decompose.
```julia
# Convert A to a MtlArray
julia> using Metal
julia> Ametal = mtl(A)
julia> cpd_metal = ITensorCPD.decompose(A, 50);
```
Though the decompositions are available on GPU, acceleration is currently not guarenteed and depends on problem size and GPU device.
GPU support is handled by the backedn ITensors.jl library.

There are also a number of convergence criteria. Convergence criteria can be used to efficiently oversee the accuracy of a CPD
and stop a decomposition.
The most popular stopping criteria for the CPD is one which considers the absolute change in the CPD fit, where the fit is measured as
`1 - || T - T_{CP} || / ||T||`. These different stopping criteria can be provided to the decomposition using the keyword `check`
```julia
# Construct an ITensorCPD convergence check object using the change in the CPD fit.
# This takes the ALS stopping condition epsilon, the number of allowed iterations and 
# the norm of the reference tensor
julia> check = ITensorCPD.FitCheck(1e-3, 100, norm(T))

# One can pass CPD objects into ITensorCPD by calling the `als_optimize`
# function which optimized the given CPD using a standard alternating least squares algorithm.
julia> cpd = ITensorCPD.decompose(T, 50; check, verbose=true);
50       1       0.9999946084173382      0.9999946084173382
50       2       0.9999950065419643      3.981246260442717e-7
50       3       0.9999953749621856      3.684202213305454e-7

# We can also pass check to the als_optimize function
julia> init_guess = ITensorCPD.random_CPD(T, 50);
julia> ITensorCPD.als_optimize(T, init_guess; check, verbose=true);
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

Currently the library only supports the ALS optimization of tensors and tensor networks. In the future we plan to introduce other optimization strategies which will be named as `xxx_optimize`.
However, the library does support a variety of ALS strategies. The ALS strategy can be modified using the keyword `alg`.
At the moment the names for these algorithms are under development so please note that they may change in the future to improve the codes readability.
The most popular ALS algorithm is via the normal equation. This optimizes factors by updating each factor using the gradient of the following loss function `f = 1/2 || T - Tcp ||^2`.
In the library, there are two algorithms to compute the normal equation `KRP` and `direct`. `KRP` computes the full Khatri-Rao product (KRP) in the construction of the normal equation and `direct` uses tensor products to avoid forming the KRP.
```julia
# Form an algorithm
julia> alg = ITensorCPD.KRP()
julia> cpd = ITensorCPD.decompose(T, 50; alg);
## Decomposes the 99% accuracy in 23.991 ms 

# Change the algorithm to direct
julia> alg = ITensorCPD.direct() 
julia> cpd = ITensorCPD.decompose(T, 50; alg);
## Decomposes the 99% accuracy in 34.609 ms
```
Note that some algorithms may require inputs in the constructor. We are working on documentation for these algorithm objects.
For example, the leverage score based randomized CP-ALS algorithm introduced by [Battaglino et al](https://arxiv.org/abs/1701.06600) can be used via the `LevScoreSampled` algorithm.
```julia
# Use a random sampled CPD algorithm with 500 samples per mode
julia> alg = ITensorCPD.LevScoreSampled(500)
julia> cpd = ITensorCPD.decompose(T, 50; alg);
```

This library also supports:
1. CPD x ITensor contractions,
2. CPD x CPD contractions and,
3. CPD x ITensorNetworks contractions.
   
We are currently working on improving the API for these contractions.

The library is flexible and is it possible to implement custom ALS algorithms, convergence checking algorithms and optimizers which can be picked up seemlessly via Julia's multiple dispatch system.
Please refer to library API for references on custom algorithms and please feel free to contribute to the library.
