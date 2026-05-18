import numpy as np


# ---------------------------------------------------------------------------
# Mixture model definition
# ---------------------------------------------------------------------------
# Components: 1 LogNormal + 10 Normal distributions
# Preserves the original pomegranate GeneralMixtureModel parameters exactly.
# Replaced pomegranate (Cython / macOS-incompatible) with a pure-numpy
# equivalent that produces an identical distribution.

_WEIGHTS = np.array([
    0.2,      # LogNormal(mu=2.9, sigma=1.2)
    0.7,      # Normal(mean=100,  std=0.15)
    0.06,     # Normal(mean=200,  std=0.15)
    0.004,    # Normal(mean=300,  std=0.15)
    0.0329,   # Normal(mean=400,  std=0.15)
    0.001,    # Normal(mean=500,  std=0.15)
    0.0006,   # Normal(mean=600,  std=0.15)
    0.0004,   # Normal(mean=700,  std=0.15)
    0.0005,   # Normal(mean=800,  std=0.15)
    0.0003,   # Normal(mean=900,  std=0.15)
    0.0003,   # Normal(mean=1000, std=0.15)
])
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()  # normalise to a valid probability vector


class OrderSizeModel:
    """
    Mixture model for order size sampling.

    Component 0  — LogNormal(mu=2.9, sigma=1.2): small retail-style orders.
    Components 1–10 — Normal(mean=k*100, std=0.15) for k in 1..10: round-lot
    clusters at 100-share increments up to 1 000 shares.

    Weights are normalised from the original pomegranate JSON definition.
    """

    def __init__(self) -> None:
        pass  # No external model object required.

    def sample(self, random_state: np.random.RandomState) -> float:
        """
        Draw one order-size sample using *random_state*.

        Selects a mixture component proportionally to _WEIGHTS, then samples
        from that component:
          - Component 0: log-normal  → exp(Normal(2.9, 1.2))
          - Components 1–10: normal  → Normal(component * 100, 0.15)

        Returns the sample rounded to the nearest integer, matching the
        original pomegranate-based implementation.
        """
        component = random_state.choice(len(_WEIGHTS), p=_WEIGHTS)

        if component == 0:
            # LogNormal: sample underlying normal, then exponentiate.
            return round(np.exp(random_state.normal(loc=2.9, scale=1.2)))
        else:
            # Normal cluster at component * 100 shares.
            mean = component * 100.0
            return round(random_state.normal(loc=mean, scale=0.15))
