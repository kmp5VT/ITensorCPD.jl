using ITensorNetworks
using ITensors

@testset "Tensor Network Times CPD check" begin
  i,j,k,l,m = Index.((10,9,8,7,3))
  A = random_itensor(i,j,k,l)
  cpd = random_CPD(A, 5)

  A = reconstruct(cpd)
  U,S,V = svd(A, i,j)

  tn = ITensorNetwork([U * S, V])

  res = ITensorCPD.tn_cp_contract(tn, cpd)
  val = (cpd[] * had_contract(res[1], ITensorCPD.cp_rank(res[2])))[]
  @test val ≈ (A * A)[]

  A = random_itensor(i,j,k)
  B = random_itensor(j,m,l)

  cpdA = random_CPD(A, 5)
  cpdB = random_CPD(B, 3)

  A = reconstruct(cpdA)
  B = reconstruct(cpdB)

  tn = ITensorNetwork([B,])

  res = ITensorCPD.tn_cp_contract(tn, cpdA)
  @test 1 - norm(cpdA[] * had_contract([res[1]..., res[2].factors...], ITensorCPD.cp_rank(cpdA)) - B * A) ≈ 1
end

@testset "CPD Times CPD test" begin 
  i,j,k,l,m = Index.((10,9,8,7,3))
  A = random_itensor(i,j,k)
  B = random_itensor(j,m,l)

  cpdA = random_CPD(A, 5)
  cpdB = random_CPD(B, 3)

  A = reconstruct(cpdA)
  B = reconstruct(cpdB)
  res = ITensorCPD.cp_cp_contract(cpdA, cpdB)

  Acore = cpdA[] * had_contract([res[1], res[2].factors...], ITensorCPD.cp_rank(cpdA))
  @test 1 - norm(cpdB[] * had_contract([Acore, res[3].factors...], ITensorCPD.cp_rank(cpdB)) - (A * B)) ≈ 1
end

