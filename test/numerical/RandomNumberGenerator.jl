CRRao.set_rng(StableRNG(123))
@test rand(CRRao.CRRao_rng, Float64, 1) ≈ [0.18102554215580358]