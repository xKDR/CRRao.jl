global CRRao_rng = MersenneTwister()

"""
```julia
set_rng(rng)
```

Set the random number generator. This is useful if you want to work with reproducible results. `rng` must be a random number generator.

# Example

```julia
using StableRNGs
CRRao.set_rng(StableRNG(1234))
```
"""
function set_rng(x)
    global CRRao_rng = x
end