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

Fit an OLS Linear Regression model on the input data. Uses the `lm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Linear Regression in our case.

```julia-repl
julia> using CRRao, RDatasets, StatsPlots

julia> df = dataset("datasets", "mtcars");

julia> container = @fitmodel(MPG ~ HP + WT + Gear, df, LinearRegression());

julia> coeftable(container);

julia> sigma(container);

julia> loglikelihood(container);

julia> aic(container);

julia> bic(container);

julia> r2(container);

julia> adjr2(container);

julia> predict(container);

julia> residuals(container);

julia> cooksdistance(container);

julia> plot(cooksdistance(container));
```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression)
    formula = apply_schema(formula, schema(formula, data))
    model = lm(formula, data)
    return FrequentistRegression(:LinearRegression, model)
end
