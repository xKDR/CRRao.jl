global CRRao_rng = MersenneTwister()

"""
```julia
using StableRNGs
CRRao.set_rng(StableRNG(1234))
```
"""
function set_rng(x)
    global CRRao_rng = x
end

# test commit 