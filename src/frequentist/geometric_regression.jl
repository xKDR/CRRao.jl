function geometric_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link)
    formula = apply_schema(formula, schema(formula, data))
    model = glm(formula, data, Geometric(), Link)
    return FrequentistRegression(:GeometricRegression, model, formula, typeof(Link))
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::GeometricRegression)
```

Fit a Geometric Regression model on the input data (with the default link function being the Log link). Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:GeometricRegression}`.
"""

function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::GeometricRegression)
    return geometric_reg(formula, data, LogLink())
end
