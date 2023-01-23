# Football Odds Prediction using Bayesian Neural Networks

Here I train a Bayesian neural network to predict odds for football matches based on various underlying characteristics of the two teams. If the difference between the out-of-sample predictions and the targets (both available before the match) differ by some statistically significant margin, then this residual could form the basis for a stat-arb-like betting strategy.

I use `pymc3` to build and train the BNN. MCMC samplers such as NUTS do not scale well, so I use [ADVI](https://arxiv.org/abs/1603.00788) instead. This is a mean-field approximation but I don't expect that this is a major bias.

I also use `poetry` here for package management since installing `pymc3` is very problematic when using just `pip`.

The data used is available [here](https://www.kaggle.com/datasets/hugomathien/soccer).