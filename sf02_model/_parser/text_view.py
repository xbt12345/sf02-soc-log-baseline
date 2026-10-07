"""Fixed raw-log feature decoder; input-only dependency closure."""
import hashlib
import re
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
BYTE_FEATURES = 256 + 65536
MARKER = re.compile('\\b(?:USER|HOST|CRED|ORG)-[0-9]+(?:-[0-9]+)*\\b')
IP = re.compile('(?<![\\w.])(?:\\d{1,3}\\.){3}\\d{1,3}(?![\\w.])')
UUID = re.compile('\\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\\b')
MAIL = re.compile('[\\w.+-]+@[\\w.-]+')
DOMAIN = re.compile('\\bdomain-[0-9]+(?:\\.[A-Za-z0-9-]+)+\\b')
DEVICE = re.compile('(?i)(?:"(?:classification|severity|riskScore|threatScore|quarantineRule|quarantineFolder|ruleName|attackType)"\\s*:\\s*|\\b(?:severityCode|ruleName|attackType)=)(?:"(?:\\\\.|[^"\\\\])*"|[^\\s,;}]+)')
CLOCK = re.compile('(?i)(?:"(?:timestamp|isotimestamp|created_at)"\\s*:\\s*|\\b(?:created_at|start|rt)=)(?:"(?:\\\\.|[^"\\\\])*"|[^\\s,;}]+)')
SYSLOG = re.compile('^<\\d{1,3}>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+\\d{1,2}\\s+(?:(?:\\d{4}|USER-\\d+)\\s+)?\\d{2}:\\d{2}:\\d{2}:?\\s*')
ISO_CLOCK = re.compile('(?<!\\w)(?:\\d{4}|USER-\\d+)-\\d{2}-\\d{2}[Tt ]\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?(?:[Zz]|[+-]\\d{2}:?\\d{2})?(?!\\d)')
NATIVE_HEADER = re.compile('^<\\d{1,3}>Original Address=\\S+\\s+1\\s+\\S+\\s+\\S+\\s+(?=flows\\b|l7_firewall\\b)')
RECORD_ID = re.compile('(?i)(?:"(?:EventRecordID|RecordNumber|ProcessId|ThreadId)"\\s*:\\s*|\\b(?:EventRecordID|RecordNumber|ProcessID|ThreadID)=)(?:"(?:\\\\.|[^"\\\\])*"|[^\\s,;}]+)')

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def view(raw):
    """Returns text plus a complete, non-overlapping character-span ledger."""
    if raw is None:
        return ('', {'is_null': True, 'sha256': None, 'length': 0, 'spans': []})
    if not isinstance(raw, str):
        raise TypeError('Raw message must be a string or NULL')
    candidates = []
    for rank, (kind, pattern) in enumerate([('device_assertion', DEVICE), ('absolute_clock', CLOCK), ('absolute_clock', NATIVE_HEADER), ('absolute_clock', SYSLOG), ('absolute_clock', ISO_CLOCK), ('identity', RECORD_ID), ('identity', MAIL), ('identity', DOMAIN), ('identity', UUID), ('identity', IP), ('identity', MARKER)]):
        for m in pattern.finditer(raw):
            if pattern is IP and any((int(v) > 255 for v in m[0].split('.'))):
                continue
            candidates.append((m.start(), m.end(), rank, kind))
    tokens = list(re.finditer('\\S+', raw))
    if len(tokens) == 14 and tokens[0][0] == '2' and re.fullmatch('\\d{12}|-|unknown', tokens[1][0], re.I) and (tokens[2][0].startswith('eni-') or MARKER.fullmatch(tokens[2][0]) or tokens[2][0] == '-') and (tokens[12][0] in ('ACCEPT', 'REJECT', '-')):
        for k in (1, 2, 10, 11):
            candidates.append((tokens[k].start(), tokens[k].end(), -2, 'identity' if k < 10 else 'absolute_clock'))
    start = raw.find('CEF:')
    if start >= 0:
        bars = list(re.finditer('(?<!\\\\)\\|', raw[start:]))
        if len(bars) >= 7:
            candidates.append((start, start + bars[6].end(), -1, 'device_assertion'))
    chosen = []
    for a, b, rank, kind in sorted(candidates, key=lambda x: (x[2], x[0], -x[1])):
        if not any((a < d and b > c for c, d, _ in chosen)):
            chosen.append((a, b, kind))
    chosen.sort()
    spans = []
    cursor = 0
    pieces = []
    for a, b, kind in chosen:
        if a > cursor:
            spans.append((cursor, a, 'behavior_or_unresolved'))
            pieces.append(raw[cursor:a])
        spans.append((a, b, kind))
        pieces.append(' <' + kind.upper() + '> ')
        cursor = b
    if cursor < len(raw):
        spans.append((cursor, len(raw), 'behavior_or_unresolved'))
        pieces.append(raw[cursor:])
    assert ''.join((raw[a:b] for a, b, _ in spans)) == raw
    result = ''.join(pieces)
    assert not raw or result
    return (result, {'is_null': False, 'sha256': digest(raw), 'length': len(raw), 'spans': spans})

def byte_matrix(texts):
    """No vocabulary fitting, hashing, OOV or prefix truncation. Order >2 lost."""
    data = []
    indices = []
    indptr = [0]
    for text in texts:
        b = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.int32)
        codes = np.concatenate((b, 256 + b[:-1] * 256 + b[1:])) if len(b) > 1 else b
        ix, count = np.unique(codes, return_counts=True)
        indices.extend(ix)
        data.extend(1 + np.log(count))
        indptr.append(len(indices))
    x = sparse.csr_matrix((np.asarray(data, np.float32), np.asarray(indices, np.int32), np.asarray(indptr, np.int64)), shape=(len(texts), BYTE_FEATURES))
    return normalize(x, norm='l2', copy=False)
