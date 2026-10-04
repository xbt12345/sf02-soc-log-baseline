"""Versioned ASA syslog-header view for V123's registered input-only trial.

The event body is passed verbatim to the older residual view. This module does
not alter facts, labels, fold membership, or any historical normalizer.
"""
import re

from v75_views import SYSLOG, view
from v75_corrective import stable
from v99_normalization_feasibility import normalize

IDENTITY_CLOCK = r'(?:(?:USER|HOST|CRED|ORG)-)+[0-9]+(?:-[0-9]+)*'
HEADER = re.compile(
    r'^<\d{1,3}>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+'
    r'(?:(?:\d{4}|' + IDENTITY_CLOCK + r')\s+)?'
    r'(?:\d{2}:\d{2}:\d{2}|' + IDENTITY_CLOCK + r'):?\s*'
    r'(?=USER-0010-0324\s+Deny\b)'
)


def old_text(raw):
    return normalize(stable(view(raw)[0]), 'placeholder_cluster')


def transform(raw):
    """Return new text, parsed header span, and whether the old parser missed it.

    All observed ASA rows are required to match; an unknown header raises,
    rather than silently stripping an unverified event prefix.
    """
    match = HEADER.match(raw)
    if match is None:
        raise ValueError('ASA header did not match registered grammar')
    if SYSLOG.match(raw) is not None:
        return old_text(raw), (0, match.end()), False
    head, body = raw[:match.end()], raw[match.end():]
    if not re.search(IDENTITY_CLOCK + r':?\s*$', head):
        raise ValueError('Old parser miss was not the registered sanitized clock')
    if not body.startswith('USER-0010-0324 Deny '):
        raise ValueError('Header/body boundary changed')
    patched = ' <ABSOLUTE_CLOCK> ' + view(body)[0]
    return normalize(stable(patched), 'placeholder_cluster'), (0, match.end()), True
