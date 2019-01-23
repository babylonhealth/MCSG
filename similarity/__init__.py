import numpy as np
from vmf_utils import aic as vmf_aic, tic as vmf_tic, fit_mean_direction, fit_concentration, log_vMF_gradient
from gaussian_utils import aic, aic_spherical, tic, tic_spherical


def von_mises_correction_aic(Dnew, Dc):
    """
    Atm just a likelihood ratio test. Currently working on creating a model
    penalty that scales with the data (as opposed to just a constant shift)

    :param Dnew[nxd matrix]: set 1
    :param Dc[mxd matrix]: set 2

    :return [float]: semantic similarity measure (approximation of the bayes factors)
    """

    # 0 inputs just cause undefined/numerically unstable solutions
    if (Dnew==0).all() or (Dc==0).all():
        return np.nan # if nan will cleanup later

    D = np.concatenate((Dnew, Dc), axis=0)

    aic_x = -vmf_aic(Dnew)
    aic_y = -vmf_aic(Dc)
    aic_xy = -vmf_aic(D)
    similarity = aic_xy - (aic_x + aic_y)

    return similarity


def von_mises_correction_tic(Dnew, Dc):
    """
    :param Dnew[nxd matrix]: set 1
    :param Dc[mxd matrix]: set 2

    :return [float]: semantic similarity measure (approximation of the bayes factors)
    """

    # 0 inputs just cause undefined/numerically unstable solutions
    if (Dnew==0).all() or (Dc==0).all():
        return np.nan # if nan will cleanup later

    D = np.concatenate((Dnew, Dc), axis=0)

    tic_x = -vmf_tic(Dnew)
    tic_y = -vmf_tic(Dc)
    tic_xy = -vmf_tic(D)
    similarity = tic_xy - (tic_x + tic_y)

    return similarity


def gaussian_correction_aic(Dnew, Dc):
    if Dnew.shape[0] < 2 or Dc.shape[0] < 2:
        return np.nan

    D = np.concatenate((Dnew, Dc), axis=0)

    aic_x = -aic(Dnew)
    aic_y = -aic(Dc)
    aic_xy = -aic(D)
    similarity = aic_xy - (aic_x + aic_y)

    return similarity


def gaussian_correction_aic_fast(Dnew, Dc):
    K, D = Dnew.shape
    L, D = Dc.shape
    if Dnew.shape[0] < 2 or Dc.shape[0] < 2:
        return np.nan

    mu_1 = np.mean(Dnew, axis=0)
    mu_2 = np.mean(Dc, axis=0)
    mu_1_sq = np.mean(Dnew ** 2, axis=0)
    mu_2_sq = np.mean(Dc ** 2, axis=0)
    p1 = K * 1.0 / (K + L)
    mu_3 = p1 * mu_1 + (1 - p1) * mu_2
    mu_3_sq = p1 * mu_1_sq + (1 - p1) * mu_2_sq

    reg = 1e-5
    v_1 = mu_1_sq - mu_1 ** 2 + reg
    v_2 = mu_2_sq - mu_2 ** 2 + reg
    v_3 = mu_3_sq - mu_3 ** 2 + reg

    ll_fast_x = K * np.sum(np.log(v_1))
    ll_fast_y = L * np.sum(np.log(v_2))
    ll_fast_xy = (K + L) * np.sum(np.log(v_3))
    similarity_fast = - ll_fast_xy + ll_fast_x + ll_fast_y

    return similarity_fast + 4 * D


def spherical_gaussian_correction_aic(Dnew, Dc):
    if Dnew.shape[0] < 2 or Dc.shape[0] < 2:
        return np.nan

    D = np.concatenate((Dnew, Dc), axis=0)

    aic_x = -aic_spherical(Dnew)
    aic_y = -aic_spherical(Dc)
    aic_xy = -aic_spherical(D)

    return aic_xy - (aic_x + aic_y)


def gaussian_correction_tic(Dnew, Dc):
    if Dnew.shape[0] < 2 or Dc.shape[0] < 2:
        return np.nan

    D = np.concatenate((Dnew, Dc), axis=0)

    tic_x = -tic(Dnew)
    tic_y = -tic(Dc)
    tic_xy = -tic(D)

    similarity = tic_xy - (tic_x + tic_y)

    return similarity


def spherical_gaussian_correction_tic(Dnew, Dc):
    if Dnew.shape[0] < 2 or Dc.shape[0] < 2:
        return np.nan

    D = np.concatenate((Dnew, Dc), axis=0)

    tic_x = -tic_spherical(Dnew)
    tic_y = -tic_spherical(Dc)
    tic_xy = -tic_spherical(D)

    similarity = tic_xy - (tic_x + tic_y)

    return similarity


aic_total = 0
aic_spherical_total = 0
count = 0


def comparison(Dnew, Dc):
    global aic_total, aic_spherical_total, count
    aic_x = -aic(Dnew)
    aic_y = -aic(Dc)
    aic_sum = aic_x + aic_y

    aic_spherical_x = -aic_spherical(Dnew)
    aic_spherical_y = -aic_spherical(Dc)
    aic_spherical_sum = aic_spherical_x + aic_spherical_y

    aic_total += aic_sum
    aic_spherical_total += aic_spherical_sum
    count += 1

    print(aic_total * 1.0 / count, aic_spherical_total * 1.0 / count, aic_total * 1.0 / aic_spherical_total)
    return np.random.randn()


NAME_TO_SIM = {
    'von_mises_correction_tic': von_mises_correction_tic,
    'von_mises_correction_aic': von_mises_correction_aic,
    'gaussian_correction_aic': gaussian_correction_aic,
    'gaussian_correction_aic_fast': gaussian_correction_aic_fast,
    'gaussian_correction_tic': gaussian_correction_tic,
    'spherical_gaussian_correction_tic': spherical_gaussian_correction_tic,
    'spherical_gaussian_correction_aic': spherical_gaussian_correction_aic,
    'comparison': comparison
}


def get_similarity_by_name(sim_name):
    return NAME_TO_SIM[sim_name]
