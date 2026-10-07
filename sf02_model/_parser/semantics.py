"""Fixed raw-log feature decoder; input-only dependency closure."""
import json
import re
import numpy as np
from scipy import sparse
from . import message_projection as previous
from . import fact_schema as learning
SEMANTIC_ENUMS = {'icmp_message': ['echo_request', 'echo_reply', 'destination_unreachable', 'time_exceeded', 'parameter_problem'], 'icmp_unreachable': ['network', 'host', 'protocol', 'port', 'fragmentation_needed', 'source_route', 'administratively_prohibited'], 'auth_result': ['success', 'failure'], 'policy_decision': ['allowed', 'denied'], 'authentication_interaction': ['no_response']}
ISO_LITERAL = re.compile('(?<!\\w)\\d{4}-\\d{2}-\\d{2}[tT ]\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?(?:[zZ]|[+-]\\d{2}:?\\d{2})?(?!\\d)')
TASK_BOUNDARY = re.compile('(<(startboundary|endboundary)>)[^<]*(</\\2>)', re.I)

def auth_values(raw):
    found = previous.prior.prior.auth_object(raw)
    if found is None:
        return ({}, [])
    start, end, fragment, roots, tail = found
    values, spans = ({}, [])
    for a, b, path, begin in roots:
        if path[0] in ('result', 'reason'):
            try:
                value = json.loads(fragment[begin:b])
            except (ValueError, TypeError):
                continue
            if isinstance(value, str) and (not previous.prior.base.MARKER.search(value)):
                values[path[0]] = value.casefold()
                spans.append({'field': path[0], 'span': [start + begin, start + b]})
    return (values, spans)

def semantic(text, facts, route, authentication=None):
    f = dict(facts)
    audit = {'rules': [], 'removed': {}}
    clean_text, boundaries = TASK_BOUNDARY.subn(lambda m: m[1] + 'absolute_time' + m[3], text)
    clean_text, timestamps = ISO_LITERAL.subn(' ', clean_text)
    if timestamps:
        audit['rules'].append('remove_absolute_lexical_clock')
        audit['clock_tokens_removed'] = timestamps
    if boundaries:
        audit['rules'].append('task_scheduler_absolute_boundary')
        audit['absolute_boundary_fields'] = boundaries
    if f.get('transport_protocol') == 'icmp':
        kind = {0: 'echo_reply', 8: 'echo_request', 3: 'destination_unreachable', 11: 'time_exceeded', 12: 'parameter_problem'}.get(f.get('icmp_type'))
        if kind:
            f['icmp_message'] = kind
            audit['rules'].append('RFC792_message_type')
        if f.get('icmp_type') == 3:
            reason = {0: 'network', 1: 'host', 2: 'protocol', 3: 'port', 4: 'fragmentation_needed', 5: 'source_route', 13: 'administratively_prohibited'}.get(f.get('icmp_code'))
            if reason:
                f['icmp_unreachable'] = reason
                audit['rules'].append('RFC1812_unreachable_reason')
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
        f['auth_result'] = f['outcome']
        audit['rules'].append('existing_native_auth_observation')
    return {'text': clean_text.strip(), 'facts': f, 'audit': audit}

def prepare_message(raw):
    old = previous.prior.prepare_message(raw)
    projected = previous.view_record(previous.project(old['b1'], old['facts'], old['route']), 'C_BOTH')
    values, spans = auth_values(raw) if old['route'] == 'authentication' else ({}, [])
    return semantic(projected['text'], projected['facts'], old['route'], values)

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
        rows, cols = ([], [])
        for i, f in enumerate(facts):
            for k in SEMANTIC_ENUMS:
                if k in f:
                    name = k + '=' + f[k]
                    if name not in self.extra_index:
                        raise ValueError(name)
                    rows.append(i)
                    cols.append(self.extra_index[name])
        extra = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(facts), len(self.extra_names)))
        return sparse.hstack([base, extra], format='csr')

    def names(self):
        return np.concatenate([self.base.names(), self.extra_names])
