function Poisson_Reg_predicts(obj, newdata::DataFrame)
    formula = obj.formula
    fm_frame = ModelFrame(formula, newdata)
    X = modelmatrix(fm_frame)
    beta = obj.beta
    z = X * beta
    μ = exp.(z)
    μ
end

function poisson_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link)
    formula = apply_schema(formula, schema(formula, data))
    model = glm(formula, data, Poisson(), Link)
    return FrequentistRegression(:PoissonRegression, model)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression)
```

Fit a Poisson Regression model on the input data (with the default link function being the Log link). Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.

# Arguments
- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset. 
- `modelClass`: Object representing the type of regression, which is Poisson Regression in this case.xc
# Example

```julia-repl
julia> using CRRao, RDatasets

julia> sanction = dataset("Zelig", "sanction");

julia> container = @fitmodel(Num ~ Target + Coop + NCost, sanction, PoissonRegression());
```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression)
    return poisson_reg(formula, data, LogLink())
end
