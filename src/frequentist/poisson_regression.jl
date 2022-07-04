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

Fit a Poisson Regression model on the input data (with the default link function being the Log link). Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:PoissonRegression}`.
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression)
    return poisson_reg(formula, data, LogLink())
end
