function linear_reg_predicts(obj, newdata::DataFrame)
    formula = obj.formula
    fm_frame = ModelFrame(formula, newdata)
    X = modelmatrix(fm_frame)
    beta = coef(obj.res)
    y_pred = X * beta
    y_pred
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression)
```

Fit an OLS Linear Regression model on the input data. Uses the `lm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:LinearRegression}`.
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression)
    formula = apply_schema(formula, schema(formula, data))
    model = lm(formula, data)
    return FrequentistRegression(:LinearRegression, model)
end
