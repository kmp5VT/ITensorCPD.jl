# ITensorCPD.jl

A higher-order tensor decomposition library which supports the canonical polyadic decomposition (CPD) of tensors and tensor-networks.
The code also supports non-covarient contractions which occur in multilinear algebra based algorithms.
The CPD takes advantage of the "smart" indexing system from [ITensors.jl](https://github.com/ITensor/ITensors.jl) and
the tensor-network decomposition's are supported by [ITensorNetworks.jl](https://github.com/ITensor/ITensorNetworks.jl).

The goal of this code is to support the decomposition of distributed CPU and GPU tensors natively via the backend support of ITensors.jl.
