"""
Original: https://github.com/WLM1ke/LedoitWolf
Actual source: https://github.com/JalexCode/LedoitWolf

@description:
Ledoit & Wolf constant correlation unequal variance shrinkage estimator.
"""
from typing import Tuple

import numpy as np
class LedoitWolf:
    def __init__(self) -> None:
        self._covariance: np.array = None
        self._average_cor: float = None
        self._shrink: float = None
    
    @property    
    def covariance_(self) -> np.array:
        return self._covariance
    @property  
    def average_cor_(self) -> float:
        return self._average_cor
    @property  
    def shrink_(self) -> float:
        return self._shrink
    
    def shrinkage(self, returns: np.array) -> Tuple[np.array, float, float]:
        """Shrinks sample covariance matrix towards constant correlation unequal variance matrix.

        Ledoit & Wolf ("Honey, I shrunk the sample covariance matrix", Portfolio Management, 30(2004),
        110-119) optimal asymptotic shrinkage between 0 (sample covariance matrix) and 1 (constant
        sample average correlation unequal sample variance matrix).

        Paper:
        http://www.ledoit.net/honey.pdf

        Matlab code:
        https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffde5e2d4e/covCor.m.zip

        Special thanks to Evgeny Pogrebnyak https://github.com/epogrebnyak

        :param returns:
            t, n - returns of t observations of n shares.
        :return:
            Covariance matrix, sample average correlation, shrinkage.
        """
        t, n = returns.shape
        mean_returns = np.mean(returns, axis=0, keepdims=True)
        returns = returns - mean_returns # previus version use returns-=mean_returns, and that raises an UFuncOutputCastingError -> Cannot cast ufunc 'subtract' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
        sample_cov = returns.transpose() @ returns / t

        # sample average correlation
        var = np.diag(sample_cov).reshape(-1, 1)
        sqrt_var = var ** 0.5
        unit_cor_var = sqrt_var * sqrt_var.transpose()
        average_cor = ((sample_cov / unit_cor_var).sum() - n) / n / (n - 1)
        prior = average_cor * unit_cor_var
        np.fill_diagonal(prior, var)

        # pi-hat
        y = returns ** 2
        phi_mat = (y.transpose() @ y) / t - sample_cov ** 2
        phi = phi_mat.sum()

        # rho-hat
        theta_mat = ((returns ** 3).transpose() @ returns) / t - var * sample_cov
        np.fill_diagonal(theta_mat, 0)
        rho = (
            np.diag(phi_mat).sum()
            + average_cor * (1 / sqrt_var @ sqrt_var.transpose() * theta_mat).sum()
        )

        # gamma-hat
        gamma = np.linalg.norm(sample_cov - prior, "fro") ** 2

        # shrinkage constant
        kappa = (phi - rho) / gamma
        shrink = max(0, min(1, kappa / t))

        # estimator
        sigma = shrink * prior + (1 - shrink) * sample_cov

        # set values
        self._covariance = sigma
        self._average_cor = average_cor
        self._shrink = shrink
        # return
        return sigma, average_cor, shrink
