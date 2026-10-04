"""Message-only B1 payload and B2 observed facts. No target or outer metadata access.

Supported grammars are explicit; legacy fallback is flagged, never called fully
source invariant. Known observation semantics are not SOC target labels.
"""
import functools
import hashlib
import json
import re

import soc_v3_prepare as base
import soc_v32_prepare as lexical
import v331_prepare as prior
import v351_safeguards as guard

VERSION = 'v36-representation-1.0'
REDACTED_WORD = re.compile(r'[\w-]*(?:USER|ORG|CRED)-[0-9][\w-]*', re.I)
KV = re.compile(r'(?<!\S)([A-Za-z][\w-]*)=')
AUTH_TITLES = {'an account failed to log on.': 'failure',
               'an account was successfully logged on.': 'success'}
FACT_FIELDS = {'status', 'substatus', 'failurereason', 'logontype', 'keylength',
               'authenticationpackagename', 'processname', 'commandline', 'objectname',
               'accessmask', 'servicename', 'ticketencryptiontype'}


def literal_string(value):
    """Read complete string boundaries. Decode valid escapes, keep damaged ones.

    Does not infer redacted field names or convert an incomplete value to truth.
    """
    if not (value.startswith('"') and value.endswith('"')):
        return None
    try:
        return json.loads(value)
    except ValueError:
        body = value[1:-1]
        def decode(m):
            token = m.group()
            try:
                return json.loads('"'+token+'"')
            except ValueError:
                return token
        return re.sub(r'\\(?:u[0-9a-fA-F]{4}|["\\/bfnrt]|.)', decode, body)


def payload_text(raw):
    # Unknown redacted *whole words* become one marker; retain operators/paths.
    raw = REDACTED_WORD.sub('entity', raw)
    raw = lexical.IP_CANDIDATE.sub(lambda m: 'entity' if lexical.valid_ip(m.group()) else m.group(), raw)
    for pat in (base.UUID, base.MAC, base.LONG_HEX):
        raw = pat.sub('entity', raw)
    return re.sub(r'\s+', ' ', raw).strip().casefold()


def scalar(value):
    if value.startswith('"'):
        return literal_string(value)
    try:
        return json.loads(value)
    except ValueError:
        return None


def number(value, maximum=None):
    if value is None or base.MARKER.search(str(value)):
        return None
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', str(value).strip()):
        return None
    n = float(value)
    return n if maximum is None or n <= maximum else None


def package(text, facts, route, evidence, supported=True):
    facts = {k: v for k, v in facts.items() if v is not None}
    return {'b1': text, 'facts': facts, 'route': route, 'supported': supported,
            'evidence': evidence, 'no_observed_fact': not facts,
            'b2_key': json.dumps([text, facts], sort_keys=True, ensure_ascii=False, separators=(',', ':'))}


def facts_text(fields):
    return ' '.join(str(k)+' '+str(v) for k, v in sorted(fields.items()) if v not in (None, '', 'unknown_value'))


def asa(raw, parts):
    f = guard.asa_visible_facts(raw)
    facts = {'category': 'network', 'action': 'deny', 'outcome': 'blocked',
             'protocol': f['protocol'], 'src_role': prior.role(parts['facts']['src']),
             'dst_role': prior.role(parts['facts']['dst']),
             'src_port': f['src']['port'], 'dst_port': f['dst']['port']}
    if f['icmp']:
        m = re.fullmatch(r'\s*\(type\s+(\S+),\s*code\s+(\S+)\)', f['icmp'])
        if m:
            facts.update(icmp_type=number(m[1], 255), icmp_code=number(m[2], 255))
    # Keep complete ports/roles in text; ACL identities/hash and severity aren't facts.
    text = facts_text({k: v for k, v in facts.items() if k not in ('category', 'outcome')})
    return package(text, facts, 'asa', [{'span': [parts['body_start'], len(raw)], 'kind': 'complete_acl_body'}])


def authentication(raw, found):
    _, visible, removed, unknown = prior.auth_view(found)
    clean = {k: v for k, v in visible.items() if v != 'unknown_value'}
    facts = {}
    if clean:
        facts['category'] = 'authentication'
    result = clean.get('result', '').casefold()
    if result in ('denied', 'failure', 'fail'):
        facts['outcome'] = 'failure'
    elif result in ('success', 'allow', 'allowed'):
        facts['outcome'] = 'success'
    reason = clean.get('reason')
    if reason == 'invalid_passcode':
        facts['credential_check'] = 'invalid'
    elif reason == 'valid_passcode':
        facts['credential_check'] = 'valid'
    elif reason in ('no_response', 'call_not_answered'):
        facts['response'] = 'missing'
    # Raw result stays unknown when redacted. Reason semantics do not rewrite it.
    text = payload_text(facts_text(clean))
    if not text:
        text = ''
    return package(text, facts, 'authentication', [{'span': [found[0], found[1]], 'kind': 'complete_root_auth_fields'}])


def windows(raw, props):
    if not any(path == ('winlog',) for a, b, path, begin in props):
        return None
    roots, fields, observations, evidence = [], {}, {}, []
    for a, b, path, begin in props:
        value = scalar(raw[begin:b])
        if len(path) == 1 and isinstance(value, str) and len(value) > 40:
            # A known message key or a redacted top-level field containing a
            # conventional event body; not arbitrary nested request contents.
            if path[0] == 'message' or base.MARKER.fullmatch(path[0]):
                roots.append((value, a, b))
        if path[0] == 'winlog' and len(path) >= 3 and path[-1] in FACT_FIELDS and isinstance(value, (str, int, float)):
            fields[path[-1]] = value
            evidence.append({'span': [begin, b], 'field': path[-1]})
        if len(path) == 2 and path[0] != 'winlog' and path[-1] in ('code', 'outcome'):
            observations.setdefault(path[0], {})[path[-1]] = value
    if len(roots) != 1:
        return None
    body, a, b = roots[0]
    evidence.append({'span': [a, b], 'kind': 'bounded_windows_message'})
    facts = {}
    title = body.splitlines()[0].strip().casefold()
    if title in AUTH_TITLES:
        facts.update(category='authentication', outcome=AUTH_TITLES[title])
        # Only the stock authentication explanatory footer, after event fields.
        footer = re.search(r'(?im)^\s*(?:this event|'+base.MARKER.pattern+r'\s+'+base.MARKER.pattern+r')\s+is generated when\b', body)
        if footer:
            body = body[:footer.start()]
        # Strip explicit identity/id slots; preserve statuses and executable facts.
        lines = []
        for line in body.splitlines():
            if re.search(r'(?i)^\s*(?:security id|account name|account domain|workstation name|caller process id|logon id)\s*:', line):
                continue
            lines.append(line)
        body = '\n'.join(lines)
    if facts.get('category') == 'authentication':
        for name in ('status', 'substatus'):
            v = fields.get(name)
            if isinstance(v, str) and re.fullmatch(r'0x[0-9a-f]{1,8}', v, re.I):
                facts[name] = v.casefold()
        if str(fields.get('substatus', '')).casefold() in ('0xc0000064', '0xc000006a'):
            facts['credential_check'] = 'invalid'
        for name in ('logontype', 'keylength'):
            facts[name] = number(fields.get(name))
        v = fields.get('authenticationpackagename')
        if isinstance(v, str) and not base.MARKER.search(v):
            facts['authentication_protocol'] = v.casefold()
    text = payload_text(body)
    # Known structured event parameters survive even when not present in prose.
    retained = {k: payload_text(str(v)) for k, v in fields.items() if not base.MARKER.search(str(v))}
    if retained:
        text += ' '+facts_text(retained)
    return package(text, facts, 'windows_message', evidence)


def cef(raw):
    start = raw.find('CEF:')
    if start < 0:
        return None
    prefix = raw[:start]
    # A bounded syslog prefix; never parse a quoted request pretending to be CEF.
    if prefix and not re.fullmatch(r'\s*<\d{1,3}>[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+', prefix):
        return None
    pipes = list(re.finditer(r'(?<!\\)\|', raw[start:]))
    if len(pipes) < 7:
        return None
    ext_start = start+pipes[6].end()
    extension = raw[ext_start:]
    fields, spans = {}, {}
    cursor = 0
    while True:
        m = KV.search(extension, cursor)
        if m is None:
            break
        name = m[1].casefold()
        begin = m.end()
        if extension[begin:begin+1] in ('"', '['):
            end = lexical.value_end(extension, begin)
            if end is None:
                return None
        else:
            nxt = KV.search(extension, begin)
            end = nxt.start() if nxt else len(extension)
        value = extension[begin:end].strip()
        cursor = end
        if name in fields:
            return None
        fields[name], spans[name] = value, [ext_start+begin, ext_start+end]
    facts, kept, evidence = {}, {}, []
    # Header severity, category WF/TR, rule/attack names and cs* detector output
    # are deliberately absent. Unknown redacted field names aren't restored.
    allowed = {'act', 'outcome', 'app', 'msg', 'request', 'requestmethod', 'requestclientapplication',
               'spt', 'dpt', 'in', 'out'}
    for name, value in fields.items():
        if name not in allowed:
            # Preserve a literal path-like value under a fully redacted key;
            # do not assert what its original field name was.
            if base.MARKER.fullmatch(name) and value.startswith('/'):
                kept['untyped_path_payload'] = payload_text(value)
                evidence.append({'field': 'untyped_path_payload', 'span': spans[name]})
            continue
        if value.startswith('"') and value.endswith('"'):
            value = literal_string(value)
        if value in ('', '-', None):
            continue
        if name in ('spt', 'dpt', 'in', 'out'):
            n = number(str(value).split()[0], 65535 if name in ('spt', 'dpt') else None)
            if n is not None:
                key = {'spt': 'src_port', 'dpt': 'dst_port', 'in': 'bytes_in', 'out': 'bytes_out'}[name]
                facts[key], kept[key] = n, n
        else:
            # Preserve actual scalar payload; container '=' and pipes disappear.
            kept[name] = payload_text(str(value))
        evidence.append({'field': name, 'span': spans[name]})
    action = fields.get('act', '').casefold()
    if action in ('deny', 'drop', 'block', 'blocked'):
        facts.update(action='deny', outcome='blocked')
    elif action in ('allow', 'accept', 'permit'):
        facts.update(action='allow', outcome='allowed')
    app = fields.get('app', '').casefold()
    if re.fullmatch(r'(?:tlsv?|https?|tcp|udp)[0-9.]*', app):
        facts['protocol'] = app
    status = number(fields.get('outcome'), 599)
    if status is not None and 100 <= status <= 599:
        facts['http_status'] = status
    # Do not infer HTTP method or request field names from redaction codes.
    return package(facts_text(kept), facts, 'cef_fields', evidence)


def syslog_auth(raw):
    # Observed aggregate authentication message, anchored whole body outcome.
    m = re.search(r'\bDuration:\s*(?P<duration>\S+)\s+Count:\s*(?P<count>[^;]+);\s+[^:;]+:\s+Authentication (?P<outcome>failure|success)\.\s*\x00?$', raw, re.I)
    if m is None or not re.match(r'^<\d+>[A-Z][a-z]{2}\s+\d+\s+\d\d:\d\d:\d\d\s+\S+\s+\S+:\s+\S+\s+Duration:', raw):
        return None
    facts = {'category': 'authentication', 'outcome': m['outcome'].casefold(), 'attempt_count': number(m['count'])}
    d = m['duration']
    if re.fullmatch(r'\d+:[0-5]\d:[0-5]\d(?:\.\d+)?', d):
        h, minute, second = map(float, d.split(':'))
        facts['duration_seconds'] = 3600*h+60*minute+second
    return package('authentication '+m['outcome'].casefold()+' '+facts_text({k:v for k,v in facts.items() if k not in ('category','outcome')}), facts, 'syslog_auth', [{'span': [m.start(), m.end()], 'kind': 'bounded_aggregate_auth'}])


@functools.lru_cache(maxsize=512)
def prepare_message(raw):
    parts = prior.asa_parts(raw)
    if parts is not None:
        return asa(raw, parts)
    auth = prior.auth_object(raw)
    if auth is not None:
        return authentication(raw, auth)
    c = cef(raw)
    if c is not None:
        return c
    if '"winlog"' in raw:
        w = windows(raw, lexical.members(raw))
        if w is not None:
            return w
    s = syslog_auth(raw)
    if s is not None:
        return s
    # Conserve evidence outside supported adapters; source risks are explicit.
    return package(guard.model_text({'message_sanitized': raw}), {}, 'legacy_fallback', [], False)


def prepare_record(row):
    return prepare_message(base.string(row.get('message_sanitized')))
