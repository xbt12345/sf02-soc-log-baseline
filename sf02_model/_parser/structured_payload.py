"""Fixed raw-log feature decoder; input-only dependency closure."""
import json
import re
from . import foundation as base
from . import json_projection as lexical
from . import acl_auth as prior
from . import native_guard as guard
REDACTED_WORD = re.compile('[\\w-]*(?:USER|ORG|CRED)-[0-9][\\w-]*', re.I)
KV = re.compile('(?<!\\S)([A-Za-z][\\w-]*)=')
AUTH_TITLES = {'an account failed to log on.': 'failure', 'an account was successfully logged on.': 'success'}
SECURITY_AUDIT_GUID = '54849625-5478-4994-a5ba-3e3b0328c30d'
FACT_FIELDS = {'status', 'substatus', 'failurereason', 'logontype', 'keylength', 'authenticationpackagename', 'processname', 'commandline', 'objectname', 'accessmask', 'servicename', 'ticketencryptiontype'}

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
                return json.loads('"' + token + '"')
            except ValueError:
                return token
        return re.sub('\\\\(?:u[0-9a-fA-F]{4}|["\\\\/bfnrt]|.)', decode, body)

def payload_text(raw):
    raw = REDACTED_WORD.sub('entity', raw)
    raw = lexical.IP_CANDIDATE.sub(lambda m: 'entity' if lexical.valid_ip(m.group()) else m.group(), raw)
    for pat in (base.UUID, base.MAC, base.LONG_HEX):
        raw = pat.sub('entity', raw)
    return re.sub('\\s+', ' ', raw).strip().casefold()

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
    if not re.fullmatch('[0-9]+(?:\\.[0-9]+)?', str(value).strip()):
        return None
    n = float(value)
    if maximum is not None and n > maximum:
        return None
    return int(n) if n.is_integer() else n

def package(text, facts, route, evidence, supported=True):
    facts = {k: v for k, v in facts.items() if v is not None}
    return {'b1': text, 'facts': facts, 'route': route, 'supported': supported, 'evidence': evidence, 'no_observed_fact': not facts, 'b2_key': json.dumps([text, facts], sort_keys=True, ensure_ascii=False, separators=(',', ':'))}

def facts_text(fields):
    return ' '.join((str(k) + ' ' + str(v) for k, v in sorted(fields.items()) if v not in (None, '', 'unknown_value')))

def windows_body(body):
    """Only explicit ID slots; preserve status/substatus and numeric parameters."""
    body = re.sub('(?im)(\\b(?:id|handle)\\s*:\\s*)0x[0-9a-f]+\\b', '\\1entity', body)
    body = re.sub('\\bS-1-(?:[0-9]+-)*[0-9]+\\b', 'entity', body, flags=re.I)
    body = re.sub('(?im)(^\\s*(?:account name|account domain|workstation name|security id)\\s*:\\s*)[^\\r\\n]*', '\\1entity', body)
    return body

def native_auth_observation(raw, xml=False, props=None):
    """Map documented Windows event semantics, never the competition label.

    Complete event code plus a checked native provider/layout is necessary.
    An arbitrary occurrence of 4624/4625 or a redacted result is insufficient.
    """
    if xml:
        pattern = '<Provider\\b[^<>]*' + SECURITY_AUDIT_GUID + '[^<>]*/>\\s*<(?P<tag>[\\w-]+)>(?P<code>[0-9]{4,6})</(?P=tag)>\\s*<Version>'
        found = list(re.finditer(pattern, raw, re.I))
        if len(found) != 1:
            return None
        m = found[0]
        tag = m['tag']
        if tag.casefold() != 'eventid' and (not (tag.endswith('ID') and base.MARKER.fullmatch(tag[:-2]))):
            return None
        code = m['code']
        span = [m.start('code'), m.end('code')]
    else:
        containers = {}
        for a, b, path, begin in props if props is not None else lexical.members(raw):
            if len(path) == 2 and path[0] != 'winlog' and (path[-1] in ('code', 'provider')):
                value = scalar(raw[begin:b])
                containers.setdefault(path[0], {})[path[-1]] = (value, [begin, b])
        found = []
        for v in containers.values():
            provider = v.get('provider', ('', None))[0]
            if isinstance(provider, str) and provider.casefold().endswith('-security-auditing') and ('code' in v):
                found.append(v['code'])
        if len(found) != 1:
            return None
        value, span = found[0]
        code = str(value)
    if code not in ('4624', '4625'):
        return None
    return {'outcome': 'success' if code == '4624' else 'failure', 'span': span, 'code': code}

def add_native_auth(facts, evidence, observation):
    if observation is None:
        return
    evidence.append({'kind': 'documented_windows_native_auth_event', 'span': observation['span'], 'event_code': observation['code']})
    if facts.get('outcome') not in (None, observation['outcome']):
        facts.pop('outcome', None)
        evidence.append({'kind': 'conflicting_visible_outcomes', 'span': observation['span']})
        return
    facts.update(category='authentication', outcome=observation['outcome'])

def asa(raw, parts):
    f = guard.asa_visible_facts(raw)
    facts = {'category': 'network', 'action': 'deny', 'outcome': 'blocked', 'protocol': f['protocol'], 'src_role': prior.role(parts['facts']['src']), 'dst_role': prior.role(parts['facts']['dst']), 'src_port': f['src']['port'], 'dst_port': f['dst']['port']}
    if f['icmp']:
        m = re.fullmatch('\\s*\\(type\\s+(\\S+),\\s*code\\s+(\\S+)\\)', f['icmp'])
        if m:
            facts.update(icmp_type=number(m[1], 255), icmp_code=number(m[2], 255))
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
    text = payload_text(facts_text(clean))
    if not text:
        text = ''
    return package(text, facts, 'authentication', [{'span': [found[0], found[1]], 'kind': 'complete_root_auth_fields'}])

def windows(raw, props):
    if not any((path == ('winlog',) for a, b, path, begin in props)):
        provider_ok = any((len(path) == 2 and path[0] != 'winlog' and (path[-1] == 'provider') and isinstance(scalar(raw[begin:b]), str) and scalar(raw[begin:b]).casefold().endswith('-security-auditing') for a, b, path, begin in props))
        if not provider_ok:
            return None
    roots, fields, observations, evidence = ([], {}, {}, [])
    for a, b, path, begin in props:
        value = scalar(raw[begin:b])
        if len(path) == 1 and isinstance(value, str) and (len(value) > 40):
            if path[0] == 'message' or base.MARKER.fullmatch(path[0]):
                roots.append((value, a, b))
        if path[0] == 'winlog' and len(path) >= 3 and (path[-1] in FACT_FIELDS) and isinstance(value, (str, int, float)):
            fields[path[-1]] = value
            evidence.append({'span': [begin, b], 'field': path[-1]})
        if len(path) == 2 and path[0] != 'winlog' and (path[-1] in ('code', 'outcome')):
            observations.setdefault(path[0], {})[path[-1]] = value
    if len(roots) != 1:
        return None
    body, a, b = roots[0]
    evidence.append({'span': [a, b], 'kind': 'bounded_windows_message'})
    facts = {}
    title = body.splitlines()[0].strip().casefold()
    if title in AUTH_TITLES:
        facts.update(category='authentication', outcome=AUTH_TITLES[title])
        footer = re.search('(?im)^\\s*(?:this event|' + base.MARKER.pattern + '\\s+' + base.MARKER.pattern + ')\\s+is generated when\\b', body)
        if footer:
            body = body[:footer.start()]
        lines = []
        for line in body.splitlines():
            if re.search('(?i)^\\s*(?:security id|account name|account domain|workstation name|caller process id|logon id)\\s*:', line):
                continue
            lines.append(line)
        body = '\n'.join(lines)
    if facts.get('category') == 'authentication':
        for name in ('status', 'substatus'):
            v = fields.get(name)
            if isinstance(v, str) and re.fullmatch('0x[0-9a-f]{1,8}', v, re.I):
                facts[name] = v.casefold()
        if str(fields.get('substatus', '')).casefold() in ('0xc0000064', '0xc000006a'):
            facts['credential_check'] = 'invalid'
        for name in ('logontype', 'keylength'):
            facts[name] = number(fields.get(name))
        v = fields.get('authenticationpackagename')
        if isinstance(v, str) and (not base.MARKER.search(v)):
            facts['authentication_protocol'] = v.casefold()
    text = payload_text(windows_body(body))
    retained = {k: payload_text(str(v)) for k, v in fields.items() if not base.MARKER.search(str(v))}
    if retained:
        text += ' ' + facts_text(retained)
    add_native_auth(facts, evidence, native_auth_observation(raw, props=props))
    return package(text, facts, 'windows_message', evidence)

def rendered_windows(raw):
    """Bounded first rendered text in the observed damaged XML layout.

    Read text as text; do not reconstruct anonymized XML names or event IDs.
    Refuse multiple render blocks or a nested element before the first text.
    """
    if not re.match('^<[\\w-]+\\s+xmlns=', raw):
        return None
    blocks = list(re.finditer('<Rendering[\\w-]*(?:\\s+[^<>]*)?>', raw))
    if len(blocks) != 1:
        return None
    begin = blocks[0].end()
    end = raw.find('<', begin)
    if end < 0 or not raw[end:].startswith('</'):
        return None
    body = raw[begin:end]
    if not body.strip() or len(body) < 40:
        return None
    body = body.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\t', '\t')
    title = body.splitlines()[0].strip().casefold()
    facts = {}
    if title in AUTH_TITLES:
        facts.update(category='authentication', outcome=AUTH_TITLES[title])
    evidence = [{'span': [begin, end], 'kind': 'bounded_rendered_plain_text'}]
    add_native_auth(facts, evidence, native_auth_observation(raw, xml=True))
    return package(payload_text(windows_body(body)), facts, 'windows_rendered', evidence)
