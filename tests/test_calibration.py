import numpy as np

from src.evaluation.calibration import apply_temperature, brier, evaluate, fit_temperature, log_loss


def _overconfident(n=20000, true_t=2.5, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 3, (n, 5))
    p_true = apply_temperature(np.exp(z), true_t)  # логиты z/true_t: реальные вероятности «мягче» модельных
    y = np.array([rng.choice(5, p=row) for row in p_true]) + 1
    p_model = apply_temperature(np.exp(z), 1.0)    # модель отдаёт заострённые вероятности
    return p_model, y


def test_temperature_one_is_identity_and_rows_sum_to_one():
    p = np.random.default_rng(1).dirichlet(np.ones(5), 100)
    out = apply_temperature(p, 1.0)
    assert np.allclose(out, p, atol=1e-9) and np.allclose(out.sum(1), 1)


def test_argmax_preserved_for_any_temperature():
    p = np.random.default_rng(2).dirichlet(np.ones(5), 500)
    for t in (0.3, 1.0, 4.0):
        assert (apply_temperature(p, t).argmax(1) == p.argmax(1)).all()


def test_fit_recovers_temperature_and_improves_metrics():
    p, y = _overconfident()
    T = fit_temperature(p, y)
    assert 2.0 < T < 3.1
    cal = apply_temperature(p, T)
    assert log_loss(cal, y) < log_loss(p, y) and brier(cal, y) < brier(p, y)
    assert evaluate(cal, y)["ECE"] < evaluate(p, y)["ECE"]
    assert evaluate(cal, y)["accuracy"] == evaluate(p, y)["accuracy"]


def test_underconfident_model_gets_temperature_below_one():
    p, y = _overconfident(true_t=0.5, seed=3)
    assert fit_temperature(p, y) < 1.0
