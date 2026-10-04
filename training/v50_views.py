"""Finite context-corruption loss; no label, source or evaluation lookup."""
import json
import numpy as np

VERSION = 'v50-auth-context-corruption-1.0'
CORE = frozenset({'auth_result', 'outcome', 'credential_check', 'attempt_count',
    'duration_seconds', 'response', 'authentication_interaction', 'policy_decision'})

def partial_auth(facts):
    result = facts.get('auth_result')
    if result not in ('success', 'failure') or facts.get('outcome') != result:
        return None
    if result == 'success' and (facts.get('credential_check') == 'invalid'
                              or facts.get('response') == 'missing'):
        return None
    return {k: v for k, v in facts.items() if k in CORE}

def training_views(facts, ids, labels):
    """Arguments contain FIT rows only. Same row's two weights sum to one."""
    ids = np.asarray(ids, dtype=np.int64); labels = np.asarray(labels, dtype=np.int64)
    mapping = {}; views = []; view_ids = np.full(len(facts), -1, dtype=np.int64)
    for k in np.unique(ids):
        view = partial_auth(facts[k])
        if view is None:
            continue
        key = json.dumps(view, sort_keys=True)
        if key not in mapping:
            mapping[key] = len(views); views.append(view)
        view_ids[k] = mapping[key]
    eligible = view_ids[ids] >= 0
    origin = np.r_[np.arange(len(ids)), np.flatnonzero(eligible)]
    indices = np.r_[ids, len(facts) + view_ids[ids[eligible]]]
    targets = np.r_[labels, labels[eligible]]
    weights = np.r_[np.where(eligible, .5, 1.), np.full(eligible.sum(), .5)]
    assert np.array_equal(np.bincount(origin, weights=weights), np.ones(len(ids)))
    assert np.array_equal(targets, labels[origin])
    return views, indices, targets, weights, origin, eligible
