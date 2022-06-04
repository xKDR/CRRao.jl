function negativebinomial_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link)
    formula = apply_schema(formula, schema(formula, data))
    model = glm(formula, data, NegativeBinomial(), Link)
    return FrequentistRegression(:NegativeBinomialRegression, model)
end
  
"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression)
```

Fit a Negative Binomial Regression model on the input data (with the default link function being the Log link). Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.

# Arguments
- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset. 
- `modelClass`: Object representing the type of regression, which is Negative Binomial Regression in this case.

# Example

```julia-repl
julia> using CRRao, RDatasets, StatsBase

julia> sanction = dataset("Zelig", "sanction");

julia> container = @fitmodel(Num ~ Target + Coop + NCost, sanction, NegBinomRegression());
```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression)
    return negativebinomial_reg(formula, data, LogLink())
end