## Examples: Setting up the code

```@repl examples
using RDatasets, NLSolversBase, CRRao, Logging, StableRNGs;

Logging.disable_logging(Logging.Warn); CRRao.setprogress!(false);

CRRao.set_rng(StableRNG(123))

```

## Example 1: Linear Regression

```@repl examples
df = dataset("datasets", "mtcars")

m1_1 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression());

m1_1.fit

m1_1.sigma

m1_1.LogLike

m1_1.AIC

m1_1.BIC

m1_1.R_sqr

m1_1.Adjusted_R_sqr

m1_1.fittedResponse

m1_1.residuals

m1_1.Cooks_distance

```

**Linear Regression - Ridge Prior**

```@repl examples
m1_2 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Ridge());

m1_2.summaries

m1_2.quantiles

```


**Linear Regression - Laplace Prior**

```@repl examples
m1_3 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Laplace());

m1_3.summaries

m1_3.quantiles

```

**Linear Regression - Cauchy Prior**
```@repl examples
m1_4 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Cauchy(),20000);

m1_4.summaries

m1_4.quantiles

```

**Linear Regression - T-Distributed Prior**

```@repl examples
m1_5 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_TDist());

m1_5.summaries

m1_5.quantiles

```

**Linear Regression - Uniform Prior**
```@repl examples
m1_6 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_TDist());

m1_6.summaries

m1_6.quantiles

```

## Example 2: Logistic Regression 
```@repl examples
turnout = dataset("Zelig", "turnout")

m2_1 = @fitmodel((Vote ~ Age + Race +Income + Educate)
                       ,turnout,LogisticRegression(),Logit());

m2_1.fit

m2_1.modelClass

m2_1.LogLike

m2_1.AIC

m2_1.BIC

m2_2 = @fitmodel((Vote ~ Age + Race +Income + Educate)
                       ,turnout,LogisticRegression(),Probit());

m2_2.fit

m2_2.BIC


m2_3 = @fitmodel((Vote ~ Age + Race +Income + Educate)
                       ,turnout,LogisticRegression(),Cloglog());

m2_3.fit

m2_3.BIC

m2_4 = @fitmodel((Vote ~ Age + Race +Income + Educate)
                       ,turnout,LogisticRegression(),Cauchit());

m2_4.fit

m2_4.BIC

```

**Logistic Regression - with Ridge Prior**

```@repl examples
m2_5 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_Ridge());

m2_5.summaries

m2_5.quantiles

m2_6 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Ridge(),1.0);

m2_6.summaries

m2_6.quantiles

m2_7 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Ridge(),1.0);

m2_7.summaries

m2_7.quantiles

m2_8 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Ridge(),1.0);

m2_8.summaries

m2_8.quantiles

```

**Logistic Regression - with Laplace Prior**
```@repl examples
m2_9 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_Laplace());

m2_9.summaries

m2_9.quantiles

m2_10 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Laplace());

m2_10.summaries

m2_10.quantiles

m2_11 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Laplace(),1.0);

m2_11.summaries

m2_11.quantiles

m2_12 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Laplace(),1.0);

m2_12.summaries

m2_12.quantiles

```

**Logistic Regression - with Cauchy Prior**
```@repl examples
m2_13 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_Cauchy(),1.0);

m2_13.summaries

m2_13.quantiles

m2_14 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Cauchy(),2.0,30000);

m2_14.summaries

m2_14.quantiles
           
m2_15 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Cauchy(),1.0);

m2_15.summaries

m2_15.quantiles
              
m2_16 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Cauchy(),1.0);

m2_16.summaries

m2_16.quantiles

```

**Logistic Regression - with T-Dist Prior**
```@repl examples
m2_17 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_TDist(),1.0);

m2_17.summaries

m2_17.quantiles

m2_18 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_TDist(),1.0);

m2_18.summaries

m2_18.quantiles

m2_19 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_TDist(),1.0);

m2_19.summaries

m2_19.quantiles

m2_20 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_TDist(),1.0);

m2_20.summaries

m2_20.quantiles

```

**Logistic Regression - with Uniform Prior**
```@repl examples
m2_21 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_Uniform(),1.0);

m2_21.summaries

m2_21.quantiles

m2_22 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Uniform(),1.0);

m2_22.summaries

m2_22.quantiles

m2_23 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Uniform(),1.0);

m2_23.summaries

m2_23.quantiles
                
m2_24 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Uniform(),1.0);

m2_24.summaries

m2_24.quantiles

```

## Example 3: Poisson Regression

**Poisson Regression - Likelihood analysis**
```@repl examples
sanction = dataset("Zelig", "sanction")

m3_1 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression());

m3_1.fit

m3_1.LogLike

m3_1.AIC

m3_1.BIC

```

**Poisson Regression with Ridge Prior**
```@repl examples
m3_2 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Ridge());

m3_2.summaries

m3_2.quantiles

```

**Poisson Regression with Laplace Prior**
```@repl examples
m3_3 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Laplace());

m3_3.summaries

m3_3.quantiles

```

**Poisson Regression with Cauchy Prior**
```@repl examples
m3_4 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Cauchy());

m3_4.summaries

m3_4.quantiles

```

**Poisson Regression with TDist Prior**
```@repl examples
m3_5 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_TDist());

m3_5.summaries

m3_5.quantiles

```

**Poisson Regression with Uniform Prior**
```@repl examples
m3_6 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Uniform());

m3_6.summaries

m3_6.quantiles
       
```

## Example 4: Negative Binomial Regression

```@repl examples
sanction = dataset("Zelig", "sanction")

```

**Negative Binomial Regression - Likelihood method** 
```@repl examples
m4_1 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression());

m4_1.fit

m4_1.AIC

m4_1.BIC

```

**NegativeBinomial Regression with Ridge Prior**
```@repl examples
m4_2 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Ridge());

m4_2.summaries

m4_2.quantiles

```

**NegativeBinomial Regression with Laplace Prior**
```@repl examples
m4_3 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Laplace());

m4_3.summaries

m4_3.quantiles

```

**Negative Binomial Regression with Cauchy Prior**
```@repl examples
m4_4 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Cauchy())

m4_4.summaries

m4_4.quantiles

```

**Negative Binomial Regression with TDist Prior**
```@repl examples
m4_5 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_TDist());

m4_5.summaries

m4_5.quantiles

```

**Negative Binomial Regression with Uniform Prior**
```@repl examples
m4_6 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Uniform(),1.0);

m4_6.summaries

m4_6.quantiles

```