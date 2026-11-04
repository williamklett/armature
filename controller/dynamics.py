"""Learn local visual effects of actions; no robot geometry or motion recipes.

The predictor interface can later host a nonlinear world model. This baseline
fits a local Jacobian from measured transitions and tests predictions BEFORE
adding the next transition. It is system identification, not VLA training.
"""
import numpy as np


class LocalDynamics:
    def __init__(self, actions, features, ridge=.03, radius=10.):
        self.actions, self.features = actions, features
        self.ridge, self.radius = ridge, radius
        self.samples = []

    def vector(self, value, size):
        value = np.asarray(value, dtype=float)
        if value.shape != (size,) or not np.all(np.isfinite(value)):
            raise ValueError('Nonfinite or incorrectly shaped observation/action')
        return value

    def fit(self, q):
        q = self.vector(q, self.actions)
        nearby = [s for s in self.samples if np.linalg.norm(np.asarray(s['q']) - q) <= self.radius]
        if len(nearby) < self.actions * 2:
            raise ValueError('Collect independent local probes before goal control')
        x = np.array([s['dq'] for s in nearby])
        y = np.array([s['dy'] for s in nearby])
        w = np.exp(-np.sum((np.array([s['q'] for s in nearby]) - q) ** 2, axis=1) / self.radius ** 2)
        gram = x.T @ (w[:, None] * x)
        eigen = np.linalg.eigvalsh(gram)
        if eigen[0] < .1 or eigen[-1] / eigen[0] > 100:
            raise ValueError('Actions insufficiently excited; model is underdetermined')
        matrix = np.linalg.solve(gram + self.ridge * np.eye(self.actions), x.T @ (w[:, None] * y))
        residual = float(np.sqrt(np.mean((x @ matrix - y) ** 2)))
        if residual > 3:
            raise ValueError('Visual dynamics residual too large for local control')
        return matrix, np.linalg.inv(gram + self.ridge * np.eye(self.actions)), residual

    def update(self, q, dq, dy):
        q = self.vector(q, self.actions)
        dq = self.vector(dq, self.actions)
        dy = self.vector(dy, self.features)
        if np.linalg.norm(dq) < .15:
            raise ValueError('Too little measured movement to learn from')
        prediction, error = None, None
        try:
            matrix, _, _ = self.fit(q)
            prediction = dq @ matrix
            error = float(np.linalg.norm(prediction - dy))
        except ValueError:
            pass
        entry = {'q': q.tolist(), 'dq': dq.tolist(), 'dy': dy.tolist(),
                 'prediction_before_update': None if prediction is None else prediction.tolist(),
                 'error_before_update_px': error}
        self.samples.append(entry)
        return entry

    def suggest(self, q, features, goal, lower, upper, step=2, seed=0):
        """Search continuous actions against a learned forward model and uncertainty.

        Includes zero and a damped inverse solution, then random candidates. Only
        one short action is executed before observing/replanning (receding horizon).
        """
        q = self.vector(q, self.actions)
        current = self.vector(features, self.features)
        goal = self.vector(goal, self.features)
        lower, upper = self.vector(lower, self.actions), self.vector(upper, self.actions)
        if not 0 < step <= 2 or np.any(lower >= upper) or np.any(q < lower) or np.any(q > upper):
            raise ValueError('Invalid action trust region')
        matrix, covariance, residual = self.fit(q)
        desired = goal - current
        inverse = np.linalg.solve(matrix @ matrix.T + .5 * np.eye(self.actions), matrix @ desired)
        inverse *= min(1., step / max(np.max(np.abs(inverse)), 1e-9))
        rng = np.random.default_rng(seed)
        candidates = np.vstack([np.zeros(self.actions), inverse, rng.uniform(-step, step, (2048, self.actions))])
        candidates = np.clip(candidates, np.maximum(-step, lower - q), np.minimum(step, upper - q))
        predictions = current + np.einsum('bi,ij->bj', candidates, matrix, optimize=False)
        uncertainty = np.einsum('bi,ij,bj->b', candidates, covariance, candidates)
        scores = np.sum((predictions - goal) ** 2, axis=1) + .1 * np.sum(candidates ** 2, axis=1) + (1 + residual ** 2) * uncertainty
        if not np.all(np.isfinite(scores)):
            raise ValueError('Nonfinite model prediction')
        best = int(np.argmin(scores))
        if best == 0 or np.linalg.norm(candidates[best]) < .15:
            raise ValueError('No confidently improving action in the trust region')
        return {'delta': candidates[best].tolist(), 'predicted_features': predictions[best].tolist(),
                'predicted_error_px': float(np.linalg.norm(predictions[best] - goal)),
                'model_residual_px': residual, 'uncertainty': float(uncertainty[best]),
                'candidates': len(candidates)}
