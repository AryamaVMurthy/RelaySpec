"""Conservative finite-sample accuracy-difference intervals for paired outcomes."""

import numpy as np


def paired_accuracy_interval(candidate, reference, confidence=0.95):
    """Combine exact marginal intervals for the two discordance probabilities.

    Under independent, identically distributed request pairs, the beneficial
    and adverse discordance counts each have a binomial marginal. Two exact
    intervals at1-alpha/2 simultaneously cover with probability at least1-alpha
    by the union bound. Subtract their endpoints to bound p_beneficial-p_adverse.
    Independence between those two counts is neither true nor required.
    """
    import scipy
    from scipy.stats import binomtest

    if scipy.__version__ != "1.16.1":
        raise ValueError("paired accuracy analysis requires pinned SciPy1.16.1")

    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if (
        candidate.ndim != 1
        or reference.shape != candidate.shape
        or candidate.size == 0
        or not np.isin(candidate, [0, 1]).all()
        or not np.isin(reference, [0, 1]).all()
        or not 0 < confidence < 1
    ):
        raise ValueError(
            "paired accuracy requires equal nonempty binary outcome vectors and a valid confidence level"
        )
    n = candidate.size
    beneficial = int(((candidate == 1) & (reference == 0)).sum())
    adverse = int(((candidate == 0) & (reference == 1)).sum())
    marginal_confidence = 1 - (1 - confidence) / 2
    positive = binomtest(beneficial, n).proportion_ci(
        marginal_confidence, method="exact"
    )
    negative = binomtest(adverse, n).proportion_ci(marginal_confidence, method="exact")
    return {
        "requests": int(n),
        "beneficial_discordances": beneficial,
        "adverse_discordances": adverse,
        "concordances": int(n - beneficial - adverse),
        "accuracy_difference": (beneficial - adverse) / n,
        "confidence_level": confidence,
        "confidence_interval": [
            positive.low - negative.high,
            positive.high - negative.low,
        ],
        "method": "Bonferroni combination of exact Clopper-Pearson discordance intervals",
        "scipy_version": scipy.__version__,
        "scope": "Conservative two-sided coverage of at least the stated level under IID request pairs and a fixed candidate. No claim of independence between paired models or of global task generalization.",
    }
