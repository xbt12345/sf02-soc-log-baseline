"""Check conservation and objective identity, including mixed-label inputs."""
import unittest
import numpy as np
from v117_gradient_batch_audit import stratified


class MassStratificationTests(unittest.TestCase):
    def test_conflicting_and_large_counts_are_conserved_per_input_and_class(self):
        counts = np.array([[0, 726, 0], [0, 0, 681], [0, 72, 4], [0, 1, 2], [0, 0, 3]])
        plan = stratified(counts, 7, np.random.default_rng(7))
        actual = np.zeros_like(counts, dtype=float)
        for ids, mass in plan:
            actual[ids] += mass
            np.testing.assert_allclose(mass.sum(0), counts.sum(0) / 7, atol=1e-9)
        np.testing.assert_allclose(actual, counts, atol=1e-9)

    def test_average_batch_loss_and_gradient_equal_original_row_objective(self):
        counts = np.array([[0, 5, 0], [0, 0, 9], [0, 2, 3], [0, 1, 0]])
        rng = np.random.default_rng(98)
        x = rng.normal(size=(4, 5)); w = rng.normal(size=(5, 3))
        z = x @ w
        z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        direct_loss = -(counts * np.log(p)).sum() / counts.sum()
        direct_grad = x.T @ (counts.sum(1, keepdims=True) * p - counts) / counts.sum()
        losses, gradients = [], []
        for ids, mass in stratified(counts, 5, np.random.default_rng(2)):
            losses.append(-(mass * np.log(p[ids])).sum() / (counts.sum() / 5))
            gradients.append(x[ids].T @ (mass.sum(1, keepdims=True) * p[ids] - mass) / (counts.sum() / 5))
        self.assertAlmostEqual(np.mean(losses), direct_loss, places=12)
        np.testing.assert_allclose(np.mean(gradients, axis=0), direct_grad, atol=1e-12)

    def test_repeatable_schedule_and_empty_class(self):
        counts = np.array([[0, 9, 0], [0, 2, 0], [0, 1, 0]])
        a = stratified(counts, 4, np.random.default_rng(27))
        b = stratified(counts, 4, np.random.default_rng(27))
        total = np.zeros_like(counts, dtype=float)
        for (ia, ma), (ib, mb) in zip(a, b):
            np.testing.assert_array_equal(ia, ib)
            np.testing.assert_array_equal(ma, mb)
            total[ia] += ma
        np.testing.assert_allclose(total, counts, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
