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
