function NegBinom_Reg_predicts(obj, newdata::DataFrame)
    formula = obj.formula
    fm_frame = ModelFrame(formula, newdata)
    X = modelmatrix(fm_frame)
    beta = obj.beta
    z = X * beta

    if (obj.Link == "LogLink")
        p = exp.(z)

    else
        println("This link function is not part of NegativeBinomial regression family.")
        println("-------------------------------------------------------------")
    end
    p
end

function negativebinomial_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link)
    formula = apply_schema(formula, schema(formula, data))
    y, X = modelcols(formula, data)
    fm_frame = ModelFrame(formula,data)
    X = modelmatrix(fm_frame)

    model = glm(formula, data, NegativeBinomial(), Link)
    ndims = (size(X, 1), size(X, 2))
    return FrequentistRegression(:NegativeBinomialRegression, model, ndims)
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
julia> using CRRao, RDatasets

julia> sanction = dataset("Zelig", "sanction");

julia> container = @fitmodel(Num ~ Target + Coop + NCost, sanction, NegBinomRegression());
```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression)
    return negativebinomial_reg(formula, data, LogLink())
end
