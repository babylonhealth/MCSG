# Copyright 2019 Babylon Partners Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np


def fit_covariance(X, reg_cov=1e-10):
    variances = np.var(X, axis=0, ddof=0) + reg_cov
    return variances


def fit_mean(X):
    mean = np.mean(X, axis=0)
    return mean


def get_score(X, mu, var_diag):
    N, D = X.shape
    log_prob = np.sum((X - mu) ** 2 / var_diag)
    log_det = np.sum(np.log(var_diag))
    return -.5 * (log_prob + N * D * np.log(2 * np.pi) + N * log_det)


def aic(X):
    N, D = X.shape
    K = D * 2
    mu = fit_mean(X)
    var_diag = fit_covariance(X)
    return - 2 * get_score(X, mu, var_diag) + 2 * K


def tic(X):
    N, D = X.shape
    mu = fit_mean(X)
    var_diag = fit_covariance(X)

    jacob = np.zeros((2 * D,))
    hess = np.zeros((2 * D,))

    grad_mu = lambda x: (x - mu) / var_diag
    grad_var = lambda x: .5 * (((x - mu) / var_diag) ** 2 - 1 / var_diag)

    # Jacob of mus
    for n in range(N):
        jacob[:D] += grad_mu(X[n]) ** 2

    # Jacob of sigmas
    for n in range(N):
        jacob[D:] += grad_var(X[n]) ** 2
    jacob /= N

    # Hess of mus
    hess[:D] = - N / var_diag

    # Hess of sigmas
    for n in range(N):
        hess[D:] -= (X[n] - mu) ** 2 / var_diag ** 3
    hess[D:] += N / (var_diag ** 2) / 2
    hess /= N

    # Fisher is negative of Hessian
    fisher = -hess

    penalty = np.sum(jacob / (fisher + 1e-06) )

    score = get_score(X, mu, var_diag)

    return - 2 * (score - penalty)


def fit_covariance_spherical(X, reg_cov=1e-6):
    variances = np.var(X, axis=0, ddof=0) + reg_cov

    spherical = np.mean(variances)

    return spherical


def get_score_sphere(X, mu, var_diag):
    N, D = X.shape
    var_diag =  (var_diag.mean().repeat(D))
    log_prob = np.sum((X - mu) ** 2 / var_diag)
    log_det = np.sum(np.log(var_diag))
    return -.5 * (log_prob + N * D * np.log(2 * np.pi) + N * log_det)


def get_score_spherical(X, mu, var):
    N, D = X.shape
    log_prob = np.sum((X - mu) ** 2) / var
    log_det = D * np.log(var)
    return -.5 * (log_prob + N * D * np.log(2 * np.pi) + N * log_det)


def aic_spherical(X):
    N, D = X.shape
    K = D + 1
    mu = fit_mean(X)
    var = fit_covariance_spherical(X)
    return - 2 * get_score_spherical(X, mu, var) + 2 * K


def tic_spherical(X):
    N, D = X.shape
    mu = fit_mean(X)
    var = fit_covariance_spherical(X)

    jacob = np.zeros((D + 1,))
    hess = np.zeros((D + 1,))

    grad_mu = lambda x: (x - mu) / var
    grad_var = lambda x: .5 * np.sum(((x - mu) / var) ** 2 - 1 / var)

    # Jacob of mus
    for n in range(N):
        jacob[:D] += grad_mu(X[n]) ** 2

    # Jacob of sigmas
    for n in range(N):
        jacob[D] += grad_var(X[n]) ** 2
    jacob /= N

    # Hess of mus
    hess[:D] = - N / var

    # Hess of sigma
    for n in range(N):
        hess[D] -= np.sum((X[n] - mu) ** 2) / var ** 3
    hess[D] += N / (var ** 2) / 2
    hess /= N

    # Fisher is negative of Hessian
    fisher = -hess

    penalty = np.sum(jacob / fisher)

    score = get_score_spherical(X, mu, var)

    return - 2 * (score - penalty)


def get_score_vector(X, mu, var_diag):
    N, D = X.shape
    log_prob = ((X - mu) ** 2 / var_diag).sum(axis=1)
    log_det = np.sum(np.log(var_diag))
    return -.5 * (log_prob + D * np.log(2 * np.pi) + log_det)[:, None]
