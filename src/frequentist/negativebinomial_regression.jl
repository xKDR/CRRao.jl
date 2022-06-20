function negativebinomial_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link)
    formula = apply_schema(formula, schema(formula, data))
    model = glm(formula, data, NegativeBinomial(), Link)
    return FrequentistRegression(:NegativeBinomialRegression, model, formula, typeof(Link))
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression)
```

Fit a Negative Binomial Regression model on the input data (with the default link function being the Log link). Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:NegativeBinomialRegression}`.
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression)
    return negativebinomial_reg(formula, data, LogLink())
end
