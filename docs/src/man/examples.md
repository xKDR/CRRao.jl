# Examples

In this section, we'll see more examples of how the CRRao interface is used to train regression models (some examples are already given in the [Package Guide](@ref)). Note that the structure of the code in this examples is common to all the models, which makes CRRao powerful.

## Negative Binomial Regression

Here we will cover the frequentist Negative Binomial Regression model along with a bayesian example. As usual, first let's import the required modules, and set a random number generator for getting reproducible results.

```@setup freq_neg_binom_regression
using Logging
Logging.disable_logging(Logging.Warn)
```

```@repl freq_neg_binom_regression
using CRRao, RDatasets, StableRNGs
CRRao.set_rng(StableRNG(123))
```

```@setup freq_neg_binom_regression
CRRao.setprogress!(false)
```

For this example, we will be working with the `sanction` dataset from R's [Zelig](https://zeligproject.org/). So let's load the dataset first.

```@repl freq_neg_binom_regression
sanction = dataset("Zelig", "sanction")
```

Then let's train the model.

```@repl freq_neg_binom_regression
model = @fitmodel((Num ~ Target + Coop + NCost), sanction, NegBinomRegression())
```

For this example, we'll compute the [`aic`](@ref) and [`bic`](@ref) of the model.

```@repl freq_neg_binom_regression
aic(model)
bic(model)
```

As mentioned in the guide, a full list of supported getter functions is available in the [Frequentist Regression Models](@ref) section of the API reference.

Now let's do bayesian inference for the same example.
