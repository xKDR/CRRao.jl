"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression)
```

Fit an OLS Linear Regression model on the input data. Uses the `lm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:LinearRegression}`.
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression)
    formula = apply_schema(formula, schema(formula, data))
    model = lm(formula, data)
    return FrequentistRegression(:LinearRegression, model, formula)
end
