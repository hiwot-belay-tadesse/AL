"""Acquisition functions added on top of the existing AL pipeline (walking/active_learning.py).

Nothing in here changes an existing experiment: the new methods are registered at the END of
active_learning.ALL_METHODS so the RNG keys of random / coreset / class_balanced_coreset
(np.random.default_rng([seed, fold, user, 1 + ALL_METHODS.index(m)])) are untouched, and
walking/main.py keeps defaulting to the original three via active_learning.DEFAULT_METHODS.
"""
