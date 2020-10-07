"""
Unit tests for the implementation of the best responses.
Requires CVXPY (https://www.cvxpy.org/) with the MOSEK backend 
(https://www.mosek.com/)
"""

import unittest

import torch
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar

from robust_losses import (RobustLoss, DualRobustLoss,
                           chi_square_value, cvar_value)

class TestDualLosses(unittest.TestCase):
    def setUp(self):
        self.size = 0.1
        self.reg = 0.5

        self.cvar_layer = RobustLoss(self.size, self.reg, 'cvar',
                                     tol=1e-6, max_iter=20000)
        self.chisquare_layer = RobustLoss(self.size, self.reg,
                                          'chi-square', debugging=True,
                                          tol=1e-6, max_iter=20000)
    def test_dual_chi_square(self):
        m = 1000
        dual_loss = DualRobustLoss(self.size, self.reg, 'chi-square')
        v = torch.abs(torch.randn(m))

        p_star, eta_star = self.chisquare_layer.best_response(v)

        dual_loss.eta.data = torch.Tensor([eta_star])
        val_dual = dual_loss(v)
        val_primal = chi_square_value(p_star, v, self.reg)

        self.assertTrue(
            torch.abs(
                val_primal - val_dual) / max(val_primal, val_dual) <= 1e-4
        )

    def test_dual_cvar(self):
        m = 1000
        dual_loss = DualRobustLoss(self.size, self.reg, 'cvar')
        v = torch.abs(torch.randn(m))

        # we find eta_star with scipy.optimize
        def f(eta):
            dual_loss.eta.data = torch.Tensor([eta])
            return float(dual_loss(v).detach().numpy())

        eta_star = minimize_scalar(f).x
        p_star = self.cvar_layer.best_response(v)

        val_primal = cvar_value(p_star, v, self.reg)
        val_dual = torch.Tensor([f(eta_star)])
        
        rel_error = torch.abs(
                val_primal - val_dual) / max(val_primal, val_dual)
        
        self.assertTrue(
            rel_error <= 1e-4
        )

class TestCVaRBestResponse(unittest.TestCase):
    def setUp(self):
        self.layer = RobustLoss(0.0, 0.0, 'cvar', tol=1e-5, max_iter=10000)

    def test_comparison_cvx(self):
        size_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        reg_vals = [1e-4, 1e-2, 1.0, 1e2]
        m_vals = [200, 2000]

        for size in size_vals:
            for reg in reg_vals:
                for m in m_vals:
                    with self.subTest(m=m, size=size, reg=reg):
                        v = np.abs(np.random.randn(m, ))
                        v_tensor = torch.DoubleTensor(v)

                        self.layer.size = size
                        self.layer.reg = reg

                        p_torch = self.layer.best_response(v_tensor)
                        p_cvx = torch.Tensor(cvar(v, reg, size)).type(
                            p_torch.dtype)

                        val_torch = cvar_value(p_torch, v_tensor, reg)
                        val_cvx = cvar_value(p_cvx, v_tensor, reg)

                        self.assertAlmostEqual(
                            val_torch.numpy(), val_cvx.numpy(), 3)

                        # self.assertTrue(
                        #     (torch.abs(val_torch - val_cvx)
                        #      / max(val_torch, val_cvx)) <= 1e-4
                        # )

    def test_almost_uniform(self):
        size_vals = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        reg_vals = [1e-4, 1e-2, 1.0, 1e2]
        m_vals = [200, 2000]

        for size in size_vals:
            for reg in reg_vals:
                for m in m_vals:
                    with self.subTest(m=m, size=size, reg=reg):
                        v = np.log(10.0) * np.ones(m) + 0e-4 * np.random.randn(
                            m)

                        self.layer.size = size
                        self.layer.reg = reg

                        p_torch = self.layer.best_response(torch.Tensor(v))
                        p_cvx = torch.Tensor(cvar(v, reg, size))

                        val_torch = cvar_value(p_torch, torch.Tensor(v), reg)
                        val_cvx = cvar_value(p_cvx, torch.Tensor(v), reg)

                        self.assertAlmostEqual(
                            val_torch.numpy(), val_cvx.numpy(), 3)

class TestChiSquareBestResponse(unittest.TestCase):
    def setUp(self):
        self.layer = RobustLoss(1.0, 0.1, 'chi-square', tol=1e-8,
                                max_iter=10000)

    def test_comparison_cvx(self):
        size_vals = [1e-2, 1e-1, 1.0, 10.0]
        reg_vals = [1e-4, 1e-2, 1.0, 1e2, 1e4]
        m_vals = [200, 2000, 20000]

        for size in size_vals:
            for reg in reg_vals:
                for m in m_vals:
                    with self.subTest(m=m, size=size, reg=reg):
                        v = np.abs(np.random.randn(m, ))

                        self.layer.size = size
                        self.layer.reg = reg

                        p_torch = self.layer.best_response(torch.Tensor(v))
                        p_cvx = torch.Tensor(chi_square(v, reg, size))

                        val_torch = chi_square_value(p_torch, torch.Tensor(v),
                                                     reg)
                        val_cvx = chi_square_value(p_cvx, torch.Tensor(v), reg)

                        self.assertTrue(
                            torch.abs(val_torch - val_cvx) / max(
                                val_torch, val_cvx) <= 1e-4
                        )

    def test_almost_uniform(self):
        size_vals = [1e-1, 1.0, 10.0, ]
        m_vals = [200, 2000, 20000]
        reg_vals = [1e-4, 1e-2, 1.0, 1e2, 1e4]

        for size in size_vals:
            for reg in reg_vals:
                for m in m_vals:
                    with self.subTest(m=m, size=size, reg=reg):
                        v = np.log(10.0) * np.ones(m)

                        self.layer.size = size
                        self.layer.reg = reg

                        p_torch = self.layer.best_response(torch.Tensor(v))
                        p_cvx = torch.Tensor(chi_square(v, reg, size))

                        val_torch = chi_square_value(p_torch, torch.Tensor(v),
                                                     reg)
                        val_cvx = chi_square_value(p_cvx, torch.Tensor(v), reg)

                        self.assertTrue(
                            torch.abs(val_torch - val_cvx) / max(
                                val_torch, val_cvx) <= 1e-4
                        )


# CVXPY Best responses
def chi_square(v, lam, rho):
    m = v.shape[0]
    p = cp.Variable(m, nonneg=True)
    obj = v * p - (0.5 / m) * lam * cp.sum_squares(m * p - np.ones(m, ))

    constraints = [
        cp.sum(p) == 1,
    ]

    if rho < float('inf'):
        constraints += [(0.5 / m) * \
                        cp.sum_squares(m * p - np.ones(m, )) <= rho]
    problem = cp.Problem(cp.Maximize(obj), constraints)
    problem.solve(solver=cp.MOSEK)

    return p.value


def cvar(v, lam, alpha):
    m = v.shape[0]
    p = cp.Variable(m, nonneg=True)
    obj = v * p + lam * cp.sum(cp.entr(p))

    constraints = [
        cp.max(p) <= 1.0 / (alpha * m),
        cp.sum(p) == 1,
    ]

    problem = cp.Problem(cp.Maximize(obj), constraints)
    problem.solve(solver=cp.MOSEK)
    return p.value


if __name__ == '__main__':
    unittest.main()
