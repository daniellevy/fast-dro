"""
Collection of utils used throughout the code.
"""

import io
import json
import os
import pickle

import numpy as np
import pathlib

import torch
from torch.nn import Sequential, Module, Linear
from scipy.sparse import csc_matrix
from scipy import optimize, interpolate
from scipy.stats import norm as Gaussian
from scipy.special import betaln

cifar10_label_names = ['airplane', 'automobile', 'bird',
                       'cat', 'deer', 'dog', 'frog', 'horse',
                       'ship', 'truck']


def aggregate_by_group(v, g, n):
    assert g.max() < n
    return csc_matrix((v, (g, np.zeros_like(g))), shape=(n, 1)
                      ).toarray().squeeze()


def subsample_arrays(arrays, n, seed=0):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    take_inds = np.random.choice(len(arrays[0]),
                                 n, replace=False)
    np.random.set_state(rng_state)

    return [a[take_inds] for a in arrays]


def get_weights_norm(model):
    """Returns the average 2-norm of the weights"""
    return np.mean([torch.norm(p.data).item() for p in model.parameters()])


def copy_state(model_src, model_tgt):
    """Copy weights of model_src in model_tgt"""
    model_tgt.load_state_dict(model_src.state_dict())
    return model_tgt


def average_step(model, model_avg, step, eta=0.):
    """In place averaging step from 
    http://proceedings.mlr.press/v28/shamir13.pdf
    
    Parameters
    ----------

    model : torch.Module
        Current model that we optimize
    model_avg : torch.Module
        Model corresponding to the averaging of the iterates
    step : int
        Current iteration number (starts at 1)
    eta : float, optional
        Parameter of [Shamir & Zhang, 2013], eta=0. corresponds to normal
        averaging

    Returns
    -------
    model_avg : torch.Module
        Updated averaged model
    """

    keys = model.state_dict().keys()

    for k in keys:
        model_avg.state_dict()[k].mul_(1 - ((eta + 1) / (step + eta))).add_(
            model.state_dict()[k].mul((eta + 1) / (step + eta)))
    return model_avg


def average_step_ema(model, model_avg, gamma=0.9):
    """Updates model_avg with an exponential moving average with param gamma"""
    keys = model.state_dict().keys()

    for k in keys:
        model_avg.state_dict()[k].mul_(1 - gamma).add_(
            model.state_dict()[k].mul(gamma))
    return model_avg


class SquaredGaussian(object):
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale if scale > 0 else 1e-12
        self.gaussian = Gaussian(loc=loc, scale=self.scale)

    def ppf(self, x):
        def target(a):
            return self.cdf(a) - x

        v_hi = self.scale ** 2
        while target(v_hi) < 0:
            v_hi *= 2
        return optimize.brentq(target, 0, v_hi)

    def cdf(self, x):
        return self.gaussian.cdf(np.sqrt(x)) - self.gaussian.cdf(-np.sqrt(x))


def analytical_dro_objective(p, invcdf, size=1.0, geometry='cvar', reg=0.0,
                             output_lambda=False):
    if geometry == 'cvar':
        assert reg == 0.0
        opt_lambda = None
        var = interpolate.interp1d(p, invcdf)(1 - size)
        ind = np.where(p > 1 - size)[0][0]
        out = (1 / (p[-1] - 1 + size)) * np.trapz(
            np.concatenate([[var], invcdf[ind:]]),
            np.concatenate([[1 - size], p[ind:]]))
    elif geometry == 'chi-square':
        if size < np.inf:
            assert reg == 0.0

            def chisquare(eta):
                r = np.maximum(invcdf - eta, 0)
                r /= np.trapz(r, p)
                return 0.5 * np.trapz((r - 1.0) ** 2, p) - size

            eta_0 = invcdf[0]
            while chisquare(eta_0) > 0.0:
                eta_0 = 2 * eta_0 - invcdf[-1]
            eta_1 = (invcdf[-1] + invcdf[-2]) / 2
            if chisquare(eta_1) <= 0.0:
                eta = eta_1
            else:
                eta = optimize.brentq(chisquare, eta_0, eta_1)
            r = np.maximum(invcdf - eta, 0)
            opt_lambda = np.trapz(r, p)
            r /= opt_lambda
            out = np.trapz(r * invcdf, p)
        else:
            assert reg > 0.0

            def target(eta):
                r = np.maximum(invcdf - eta, 0)
                return np.trapz(r, p) / reg - 1.0

            eta_0 = invcdf[0]
            while target(eta_0) < 0.0:
                eta_0 = 2 * eta_0 - invcdf[-1]
            eta_1 = (invcdf[-1] + invcdf[-2]) / 2
            if target(eta_1) >= 0.0:
                eta = eta_1
            else:
                eta = optimize.brentq(target, eta_0, eta_1)
            r = np.maximum(invcdf - eta, 0)
            opt_lambda = np.trapz(r, p)  # should be equal to reg!
            r /= opt_lambda
            out = 0.5 * (np.trapz(r * invcdf, p) + eta + reg)

    if output_lambda:
        return out, opt_lambda
    else:
        return out


def binomln(n, k):
    # Assumes binom(n, k) >= 0
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def subsample_cvar_w(n, s, alpha):
    sk_max = np.floor(alpha * s)
    i_s = np.arange(1, n + 1).reshape(-1, 1)
    k_s = np.arange(1 + sk_max).reshape(1, -1)

    log_choices = (binomln(i_s - 1, k_s)
                   + binomln(n - i_s, s - k_s - 1)
                   - binomln(n, s))

    p_bulk = np.sum(np.exp(log_choices[:, :-1]), axis=1)
    p_edge = np.exp(log_choices[:, -1])

    w = 1 / (alpha * s) * p_bulk + (1 - sk_max / (alpha * s)) * p_edge

    nk_max = np.floor(alpha * n)
    w_pop = (1 / (alpha * n) * (i_s <= nk_max).astype('float')
             + (1 - nk_max / (alpha * n)) * (i_s == 1 + nk_max).astype('float'))
    return w, w_pop.squeeze()
