"""v3.9 semantic projection and body identity: separate, label-free contracts."""
import hashlib
import json
import re
import numpy as np
from scipy import sparse
import v38_representation as previous
import v38_learning as learning

VERSION = 'v39-controlled-development-1.1'
SEMANTIC_ENUMS = {
    'icmp_message': ['echo_request', 'echo_reply', 'destination_unreachable', 'time_exceeded', 'parameter_problem'],
    'icmp_unreachable': ['network', 'host', 'protocol', 'port', 'fragmentation_needed', 'source_route', 'administratively_prohibited'],
    'auth_result': ['success', 'failure'],
    'policy_decision': ['allowed', 'denied'],
    'authentication_interaction': ['no_response'],
}
ISO_LITERAL = re.compile(r'(?<!\w)\d{4}-\d{2}-\d{2}[tT ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[zZ]|[+-]\d{2}:?\d{2})?(?!\d)')
TASK_BOUNDARY = re.compile(r'(<(startboundary|endboundary)>)[^<]*(</\2>)', re.I)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).digest()


def auth_values(raw):
    found = previous.prior.prior.auth_object(raw)
    if found is None:
        return {}, []
    start, end, fragment, roots, tail = found
    values, spans = {}, []
    for a, b, path, begin in roots:
        if path[0] in ('result', 'reason'):
            try:
                value = json.loads(fragment[begin:b])
            except (ValueError, TypeError):
                continue
            if isinstance(value, str) and not previous.prior.base.MARKER.search(value):
                values[path[0]] = value.casefold()
                spans.append({'field': path[0], 'span': [start + begin, start + b]})
    return values, spans


def semantic(text, facts, route, authentication=None):
    f = dict(facts)
    audit = {'rules': [], 'removed': {}}
    # Scheduled task activation/expiry boundaries are absolute wall times even
    # when de-identification has destroyed part of the date. Interval/duration
    # fields, commands and other numbers are not changed.
    clean_text, boundaries = TASK_BOUNDARY.subn(lambda m: m[1] + 'absolute_time' + m[3], text)
    clean_text, timestamps = ISO_LITERAL.subn(' ', clean_text)
    # Bare hh:mm:ss may be a duration; do not erase it without context.
    if timestamps:
        audit['rules'].append('remove_absolute_lexical_clock')
        audit['clock_tokens_removed'] = timestamps
    if boundaries:
        audit['rules'].append('task_scheduler_absolute_boundary')
        audit['absolute_boundary_fields'] = boundaries
    if f.get('transport_protocol') == 'icmp':
        kind = {0: 'echo_reply', 8: 'echo_request', 3: 'destination_unreachable',
                11: 'time_exceeded', 12: 'parameter_problem'}.get(f.get('icmp_type'))
        if kind:
            f['icmp_message'] = kind
            audit['rules'].append('RFC792_message_type')
        if f.get('icmp_type') == 3:
            reason = {0: 'network', 1: 'host', 2: 'protocol', 3: 'port',
                      4: 'fragmentation_needed', 5: 'source_route',
                      13: 'administratively_prohibited'}.get(f.get('icmp_code'))
            if reason:
                f['icmp_unreachable'] = reason
                audit['rules'].append('RFC1812_unreachable_reason')
    # ICMPv6 deliberately does not reuse the IPv4 type/code table.
    if route == 'authentication':
        a = authentication or {}
        result = a.get('result')
        if result in ('allow', 'allowed'):
            audit['removed']['outcome'] = f.pop('outcome', None)
            f['policy_decision'] = 'allowed'
        elif result in ('success', 'failure', 'fail', 'denied'):
            f['outcome'] = 'success' if result == 'success' else 'failure'
            f['auth_result'] = f['outcome']
        if a.get('reason') in ('no_response', 'call_not_answered'):
            f['authentication_interaction'] = 'no_response'
        if a.get('reason') == 'allow_unenrolled_user':
            f['policy_decision'] = 'allowed'
        if a:
            audit['rules'].append('documented_auth_result_reason_separation')
    elif route in ('windows_message', 'windows_rendered', 'syslog_auth', 'syslog_body') and f.get('outcome') in ('success', 'failure'):
        # These parser routes assign success/failure only from native auth
        # statements/codes. Audit/process success is in audit_fields, excluded.
        f['auth_result'] = f['outcome']
        audit['rules'].append('existing_native_auth_observation')
    return {'text': clean_text.strip(), 'facts': f, 'audit': audit}


def prepare_message(raw):
    old = previous.prior.prepare_message(raw)
    projected = previous.view_record(previous.project(old['b1'], old['facts'], old['route']), 'C_BOTH')
    values, spans = auth_values(raw) if old['route'] == 'authentication' else ({}, [])
    return semantic(projected['text'], projected['facts'], old['route'], values)


def prepare_record(row):
    return prepare_message(previous.prior.base.string(row.get('message_sanitized')))


def body_identity(raw, route, evidence, row_position=None):
    """Byte-valued native body signature, never a model feature or incident ID.

    Known bounded native fields accompany rendered prose; unknown messages keep
    their full original bytes. No labels, category values, or model vectors.
    """
    if not raw.strip():
        if row_position is None:
            raise ValueError('Empty messages have no observable event identity')
        return digest('empty_unlinked:' + str(row_position)), 'empty_unlinked'
    if route == 'asa':
        part = previous.prior.prior.asa_parts(raw)
        if part is None:
            raise ValueError('ASA body no longer matches grammar')
        body, method = part['body'].strip(), 'complete_asa_body'
    elif route in ('native_flow', 'native_firewall'):
        m = re.match(r'^<\d{1,3}>Original Address=\S+\s+1\s+\S+\s+\S+\s+(.*)$', raw.strip(), re.S)
        if m is None:
            raise ValueError('Native packet boundary missing')
        body, method = m[1], 'complete_packet_body'
    elif route == 'cef_fields':
        start = raw.index('CEF:')
        pipes = list(re.finditer(r'(?<!\\)\|', raw[start:]))
        boundary = start + pipes[6].end()
        ext = raw[boundary:]
        fields = []; cursor = 0
        while True:
            m = previous.prior.old.KV.search(ext, cursor)
            if m is None:
                break
            begin = m.end()
            if ext[begin:begin+1] in ('"', '['):
                end = previous.prior.lexical.value_end(ext, begin)
                if end is None:
                    raise ValueError('Incomplete CEF value')
            else:
                nxt = previous.prior.old.KV.search(ext, begin)
                end = nxt.start() if nxt else len(ext)
            if m[1].casefold() not in ('start', 'rt'):
                fields.append((m[1], ext[begin:end].strip()))
            cursor = end
        body = canonical([raw[start:boundary], sorted(fields)])
        method = 'complete_cef_except_documented_clock_slots'
    elif route in ('windows_message', 'windows_rendered'):
        fragments = []
        for item in evidence:
            if 'span' not in item:
                continue
            lo, hi = item['span']
            if not 0 <= lo <= hi <= len(raw):
                raise ValueError('Bad frozen native span')
            fragments.append((item.get('field', item.get('kind', 'native')), raw[lo:hi]))
        if not fragments:
            raise ValueError('Missing native event body')
        body, method = canonical(sorted(set(fragments))), 'complete_rendered_body_and_native_fields'
    elif route in ('syslog_body', 'syslog_auth', 'audit_fields', 'asa_acl', 'asa_protocol', 'format_payload'):
        found = previous.prior.collector_body(raw)
        body = found[0].strip() if found else raw
        # The application and audit sequence remain literal to avoid merging
        # unrelated native records. A separate near-copy audit reports them.
        method = 'complete_syslog_body' if found else 'original_fallback'
    elif route == 'authentication':
        found = previous.prior.prior.auth_object(raw)
        if found is None:
            raise ValueError('Authentication object boundary missing')
        start, end, fragment, roots, tail = found
        # Two observed redacted spellings of the documented native clock slots
        # are opaque values; no attempt to recover their date or identity.
        clocks = {'timestamp', 'isotimestamp', 'ORG-1526stamp', 'isoORG-1526stamp'}
        fields = [(p[0], fragment[begin:b]) for a, b, p, begin in roots if p[0] not in clocks]
        body = canonical([sorted(fields), fragment[tail:] if tail is not None else ''])
        method = 'complete_auth_object_except_checked_clock_slots'
    else:
        body, method = raw, 'original_fallback'
    return digest(canonical([method, body])), method


class SemanticFacts:
    def __init__(self):
        self.base = learning.FixedFacts('C_BOTH')
        self.extra_names = [k + '=' + v for k, values in SEMANTIC_ENUMS.items() for v in values]
        self.extra_index = {name: i for i, name in enumerate(self.extra_names)}

    def fit(self, facts):
        self.base.fit([{k: v for k, v in f.items() if k not in SEMANTIC_ENUMS} for f in facts])
        return self

    def transform(self, facts):
        base = self.base.transform([{k: v for k, v in f.items() if k not in SEMANTIC_ENUMS} for f in facts])
        rows, cols = [], []
        for i, f in enumerate(facts):
            for k in SEMANTIC_ENUMS:
                if k in f:
                    name = k + '=' + f[k]
                    if name not in self.extra_index:
                        raise ValueError(name)
                    rows.append(i); cols.append(self.extra_index[name])
        extra = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(facts), len(self.extra_names)))
        return sparse.hstack([base, extra], format='csr')

    def names(self):
        return np.concatenate([self.base.names(), self.extra_names])


def availability(facts):
    names = sorted(set(learning.ENUMS) | set(learning.NUMERIC) | set(learning.BIT_FIELDS) | set(SEMANTIC_ENUMS))
    a = np.zeros((len(facts), len(names)))
    for i, f in enumerate(facts):
        for j, k in enumerate(names):
            a[i, j] = k in f and (k not in learning.BIT_FIELDS or f[k] != learning.BIT_FIELDS[k][1])
    return sparse.csr_matrix(a), names
