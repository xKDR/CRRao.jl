function logistic_reg_predicts(obj,newdata::DataFrame)
    
   formula = obj.formula;
   fm_frame=ModelFrame(formula,newdata);
   X=modelmatrix(fm_frame);
   beta = obj.beta
   z = X*beta;
 
   if (obj.Link=="LogitLink")
     p = exp.(z) ./ (1 .+ exp.(z));
       
   elseif (obj.Link=="ProbitLink")
     p = Probit_Link.(z);
       
   elseif (obj.Link=="CauchitLink")
     p = Cauchit_Link.(z);
 
   elseif (obj.Link=="CloglogLink")
     p = Cloglog_Link.(z);
 
   else
       println("This link function is not part of logistic regression family.")
       println("-------------------------------------------------------------")
   end 
   p
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link)
    formula = apply_schema(formula, schema(formula, data))
    model = glm(formula, data, Binomial(), Link)
    return FrequentistRegression(:LogisticRegression, model)
end
 
 """
 ```julia
 fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit)
 ```
 
 Fit a Logistic Regression model on the input data using the Logit link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.
 
 # Arguments
 - `formula`: A formula term representing dependencies between the columns in the dataset.
 - `data`: The dataset. 
 - `modelClass`: Object representing the type of regression, which is Logistic Regression in this case.
 - `Link`: A type to specify the link function to be used, which is Logit in this case.
 
 ```julia-repl
 julia> using CRRao, RDatasets, StatsBase
 
 julia> turnout = dataset("Zelig", "turnout");
 
 julia> container = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit());
 
 julia> coeftable(container.model)
 ────────────────────────────────────────────────────────────────────────────
                   Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
 ────────────────────────────────────────────────────────────────────────────
 (Intercept)  -3.03426    0.325927    -9.31    <1e-19  -3.67307    -2.39546
 Age           0.0283543  0.00346034   8.19    <1e-15   0.0215722   0.0351365
 Race: white   0.250798   0.146457     1.71    0.0868  -0.0362521   0.537847
 Income        0.177112   0.0271516    6.52    <1e-10   0.123896    0.230328
 Educate       0.175634   0.0203308    8.64    <1e-17   0.135786    0.215481
 ────────────────────────────────────────────────────────────────────────────
 
 julia> loglikelihood(container.model)
 -1011.9906318515575
 
 julia> aic(container.model)
 2033.981263703115
 
 julia> bic(container.model)
 2061.9857760008254
 ```
 """
 function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit)
    return logistic_reg(formula, data, LogitLink())
 end
 
 """
 ```julia
 fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit)
 ```
 
 Fit a Logistic Regression model on the input data using the Probit link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.
 """
 function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit)
    return logistic_reg(formula, data, ProbitLink())
 end
 
 """
 ```julia
 fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog)
 ```
 
 Fit a Logistic Regression model on the input data using the Cloglog link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.
 ```
 """
 function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog)
    return logistic_reg(formula, data, CloglogLink())
 end
 
 """
 ```julia
 fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit)
 ```
 
 Fit a Logistic Regression model on the input data using the Cauchit link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.
 ```
 """
 function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit)
    return logistic_reg(formula, data, CauchitLink())
 end