from typing import Tuple

import numpy as np
import pymc as pm


def initialise_weights(dim1, dim2):
    return np.random.randn(dim1, dim2).astype(np.float32)


def construct_weights(name, dim1, dim2):
    init_weights = initialise_weights(dim1, dim2)
    return pm.Normal(name, mu=0, sigma=1, shape=(dim1, dim2), initval=init_weights)


def implement_layer(activ_func, x, weights):
    """Takes a linear combination of x with weights, and then
    passes through activ_func.

    Args:
        - activ_func: the activation function to apply to the
        linear combination.
        - x: the input to the layer.
        - weights: the weights for the linear combination.
    """
    return activ_func(pm.math.dot(x, weights))


def construct_bnn(
    x_train,
    y_train,
    layers: Tuple,
    hidden_activ_func,
):
    """

    Args:
        - layers: a tuple containing the dimension of each layer, including
        both the input and output layer.
    """
    with pm.Model() as bnn:
        # Construct the weights at each layer
        weights_tuple = tuple(
            construct_weights(f"layer_{k}", dim1, dim2)
            for k, (dim1, dim2) in enumerate(zip(layers, layers[1:]))
        )
        # Construct each layer
        layers_tuple = tuple(
            lambda x: implement_layer(hidden_activ_func, x, weights)
            for weights in weights_tuple[:-1]
        )
        # Build BNN up to final layer
        x = pm.MutableData("input", x_train)
        for layer in layers_tuple:
            x = layer(x)
        x = pm.math.dot(x, weights_tuple[-1])
        out = pm.Normal("output", mu=x, observed=y_train.to_frame(), shape=x.shape)

    return bnn


def train_and_predict_bnn(model, X_test, advi_iter, trace_draws):
    with model:
        advi = pm.ADVI()
        approx = pm.fit(n=advi_iter, method=advi)  # perform VI
        trace = approx.sample(draws=trace_draws)  # sample from approximate posterior
        pp_train = pm.sample_posterior_predictive(trace)  # sample posterior predictive
        pm.set_data({"input": X_test})  # switch to test data
        pp_test = pm.sample_posterior_predictive(trace)  # sample posterior predictive
        return trace, pp_train, pp_test, advi


def pp_to_preds(pp):
    preds = pp.to_dict()["posterior_predictive"]["output"]
    return preds.mean(axis=1).reshape(-1)


def pp_to_preds_stds(pp):
    preds = pp.to_dict()["posterior_predictive"]["output"]
    return preds.std(axis=1).reshape(-1)
