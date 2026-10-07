"""Fixed raw-log feature decoder; input-only dependency closure."""
import copy
import functools
import re
from . import structured_payload as old
from . import foundation as base
from . import json_projection as lexical
from . import acl_auth as prior
PORTS = {'src_port', 'dst_port'}
MONTH = '(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
CLOCK = '(?:[0-2][0-9]:[0-5][0-9]:[0-5][0-9](?:\\.[0-9]+)?|' + base.MARKER.pattern + '|-)'
PREFIX_CEF = re.compile('^\\s*<(?P<pri>[0-9]{1,3})>' + MONTH + '\\s+[0-9]{1,2}\\s+' + CLOCK + '\\s+\\S+\\s+$', re.I)
RFC3164 = re.compile('^(?:<(?P<pri>[0-9]{1,3})>)?' + MONTH + '\\s+[0-9]{1,2}\\s+(?:(?:[0-9]{4}|' + base.MARKER.pattern + ')\\s+)?' + CLOCK + ':?\\s+\\S+\\s+(?P<body>.+)$', re.I | re.S)
RFC5424 = re.compile('^<(?P<pri>[0-9]{1,3})>1\\s+(?P<stamp>\\S+)\\s+\\S+\\s+\\S+\\s+\\S+\\s+\\S+\\s+-\\s+(?P<body>.+)$', re.S)
RFC3339 = re.compile('^<(?P<pri>[0-9]{1,3})>(?P<stamp>\\S+)\\s+\\S+\\s+(?P<body>.+)$', re.S)

def packed(text, facts, route, evidence, quality=None, supported=True):
    v = old.package(text.strip(), facts, route, evidence, supported)
    v['quality'] = quality or {}
    return v

def unsigned(value, maximum=None):
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    return old.number(value, maximum)

def typed_facts(facts):
    result = copy.deepcopy(facts)
    protocol = str(result.get('protocol', 'unknown'))
    for name in PORTS:
        if name in result:
            value = result.pop(name)
            if isinstance(value, (int, float)) and float(value).is_integer():
                result[name + '_category'] = protocol + '|' + str(int(value))
    return result

def from_old(v):
    facts = copy.deepcopy(v['facts'])
    text = v['b1']
    evidence = copy.deepcopy(v['evidence'])
    if v['route'] == 'asa':
        text = old.facts_text({k: x for k, x in facts.items() if k not in PORTS | {'category', 'outcome'}})
    return packed(text, typed_facts(facts), v['route'], evidence, {'information_limited': not bool(text or facts)}, v['supported'])

def cef_fields(raw):
    start = raw.find('CEF:')
    if start < 0:
        return None
    prefix = raw[:start]
    m = PREFIX_CEF.fullmatch(prefix) if prefix else None
    if prefix and (m is None or int(m['pri']) > 191):
        return None
    if not re.match('CEF:[0-9]+\\|', raw[start:]):
        return None
    pipes = list(re.finditer('(?<!\\\\)\\|', raw[start:]))
    if len(pipes) < 7:
        return None
    begin_ext = start + pipes[6].end()
    ext = raw[begin_ext:]
    cursor = 0
    fields = {}
    spans = {}
    while True:
        match = old.KV.search(ext, cursor)
        if match is None:
            break
        name = match[1].casefold()
        begin = match.end()
        if ext[begin:begin + 1] in ('"', '['):
            end = lexical.value_end(ext, begin)
            if end is None:
                return None
        else:
            nxt = old.KV.search(ext, begin)
            end = nxt.start() if nxt else len(ext)
        value = ext[begin:end].strip()
        cursor = end
        if name in fields:
            return None
        if value.startswith('"'):
            value = old.literal_string(value)
            if value is None:
                return None
        if isinstance(value, str):
            value = re.sub('\\\\([=|\\\\])', '\\1', value)
        fields[name] = value
        spans[name] = [begin_ext + begin, begin_ext + end]
    kept = {}
    facts = {}
    evidence = []
    quality = {'port_states': {}}
    for name in sorted(fields):
        value = fields[name]
        if name not in {'act', 'outcome', 'app', 'msg', 'request', 'requestmethod', 'requestclientapplication', 'spt', 'dpt', 'in', 'out'}:
            if base.MARKER.fullmatch(name) and isinstance(value, str) and value.startswith('/'):
                kept.setdefault('untyped_path_payload', []).append(old.payload_text(value))
                evidence.append({'field': 'untyped_path_payload', 'span': spans[name]})
            continue
        evidence.append({'field': name, 'span': spans[name]})
        if name in ('spt', 'dpt', 'in', 'out'):
            first = str(value).split()[0] if str(value).split() else ''
            n = unsigned(first, 65535 if name in ('spt', 'dpt') else None)
            key = {'spt': 'src_port', 'dpt': 'dst_port', 'in': 'bytes_in', 'out': 'bytes_out'}[name]
            if name in ('spt', 'dpt'):
                quality['port_states'][key] = 'observed' if n is not None else 'unavailable'
            if n is not None:
                facts[key] = n
        elif value not in ('', '-', None):
            kept[name] = old.payload_text(str(value))
    action = str(fields.get('act', '')).casefold()
    if action in ('deny', 'drop', 'block', 'blocked'):
        facts.update(action='deny', outcome='blocked')
    elif action in ('allow', 'accept', 'permit'):
        facts.update(action='allow', outcome='allowed')
    app = str(fields.get('app', '')).casefold()
    if re.fullmatch('(?:tlsv?|https?|tcp|udp)[0-9.]*', app):
        facts['protocol'] = app
    status = unsigned(fields.get('outcome'), 599)
    if status is not None and 100 <= status <= 599:
        facts['http_status'] = status
    for k in list(kept):
        if isinstance(kept[k], list):
            kept[k] = ' '.join(sorted(kept[k]))
    return packed(old.facts_text(kept), typed_facts(facts), 'cef_fields', evidence, quality)

def collector_body(raw):
    for pattern in (RFC5424, RFC3164, RFC3339):
        m = pattern.match(raw)
        if m is None:
            continue
        if m.groupdict().get('pri') and int(m['pri']) > 191:
            continue
        stamp = m.groupdict().get('stamp')
        return (m['body'], m.start('body'))
    return None

def syslog_body(raw):
    found = collector_body(raw)
    if found is None:
        return None
    body, start = found
    if 'CEF:' in body or body.lstrip().startswith(('{', '[')):
        return None
    app = re.match('^[\\w.-]+(?:\\[[^\\]\\s]+\\])?:\\s+', body)
    if app:
        body = body[app.end():]
        start += app.end()
    native = native_audit_or_acl(body, start, raw)
    if native is not None:
        return native
    if re.match('(?:Site:|\\[|Category\\s*=)', body, re.I):
        return None
    facts = {}
    m = re.fullmatch('\\S+\\s+Duration:\\s*(\\S+)\\s+Count:\\s*([^;]+);\\s+[^:;]+:\\s+Authentication (failure|success)\\.\\s*\\x00?', body, re.I)
    if m:
        facts = {'category': 'authentication', 'outcome': m[3].casefold(), 'attempt_count': unsigned(m[2].strip())}
        if re.fullmatch('\\d+:[0-5]\\d:[0-5]\\d(?:\\.\\d+)?', m[1]):
            h, mi, se = map(float, m[1].split(':'))
            facts['duration_seconds'] = 3600 * h + 60 * mi + se
        facts = {k: v for k, v in facts.items() if v is not None}
        return packed(old.facts_text(facts), facts, 'syslog_auth', [{'span': [start, len(raw)], 'kind': 'aggregate_auth_body'}])
    if re.match('pam_unix\\([^:()]+:auth\\):\\s+authentication failure(?:;|\\s|$)', body, re.I):
        facts.update(category='authentication', outcome='failure')
    elif re.match('pam_unix\\([^:()]+:session\\):\\s+session (opened|closed)\\b', body, re.I):
        facts.update(category='session', action='open' if 'session opened' in body.casefold() else 'close')
    else:
        return None
    return packed(old.payload_text(body), facts, 'syslog_body', [{'span': [start, len(raw)], 'kind': 'pam_body'}])

def native_audit_or_acl(body, start, raw):
    m = re.match('^(?:node=\\S+\\s+)?type=(PATH|SYSCALL|CWD|PROCTITLE|EXECVE|' + base.MARKER.pattern + ')\\s+msg=audit\\([^\\s)]+\\):\\s*', body)
    if m:
        fields = {}
        evidence = []
        for a in re.finditer('(?<!\\S)(name|cwd|exe|comm|proctitle|arch|syscall|success|exit|argc|a[0-9]+)=("[^"\\n]*"|\\S+)', body[m.end():]):
            key = a[1]
            value = a[2]
            if key in fields:
                return None
            fields[key] = value[1:-1] if value.startswith('"') else value
            evidence.append({'field': key, 'span': [start + m.end() + a.start(2), start + m.end() + a.end(2)]})
        facts = {'category': 'audit'}
        texts = []
        if not base.MARKER.search(m[1]):
            facts['operation_record'] = m[1].casefold()
        if fields.get('success') in ('yes', 'no'):
            facts['outcome'] = 'success' if fields['success'] == 'yes' else 'failure'
        if re.fullmatch('[0-9]+', fields.get('syscall', '')) and re.fullmatch('[0-9a-fA-F]+', fields.get('arch', '')):
            facts['syscall_category'] = fields['arch'] + '|' + fields['syscall']
        if re.fullmatch('-?[0-9]+', fields.get('exit', '')):
            facts['exit_category'] = fields['exit']
        for key in ('name', 'cwd', 'exe', 'comm'):
            if key in fields:
                texts.append(key + ' ' + old.payload_text(fields[key]))
        if m[1] == 'EXECVE':
            for key in sorted(fields):
                if re.fullmatch('a[0-9]+', key):
                    texts.append('argument ' + old.payload_text(fields[key]))
        return packed(' '.join(texts), facts, 'audit_fields', evidence, {'record_assembly_not_assumed': True, 'redacted_record_type': bool(base.MARKER.search(m[1])), 'only_record_family_observed': not texts and set(facts) == {'category'}})
    m = re.fullmatch('(TCP|UDP|ICMP)\\s+\\S+\\s+denied by ACL from (\\S+)/(\\S+) to ([^:\\s]+):(\\S+)/(\\S+)\\s*', body, re.I)
    if m:
        facts = {'category': 'network', 'action': 'deny', 'outcome': 'blocked', 'protocol': m[1].casefold(), 'src_role': 'unknown', 'dst_role': prior.role(m[4])}
        for k, idx in [('src_port', 3), ('dst_port', 6)]:
            value = unsigned(m[idx], 65535)
            if value is not None:
                facts[k] = value
        return packed(old.facts_text({k: v for k, v in facts.items() if k not in PORTS and k not in ('category', 'outcome')}), typed_facts(facts), 'asa_acl', [{'span': [start, start + len(body)], 'kind': 'literal_acl_denial'}])
    m = re.fullmatch('Deny protocol ([0-9]{1,3}) src ([^:\\s]+):\\S+ dst ([^:\\s]+):\\S+ by \\S+-group "[^"]+" \\[[^\\]]+\\]\\s*', body, re.I)
    if m and int(m[1]) <= 255:
        facts = {'category': 'network', 'action': 'deny', 'outcome': 'blocked', 'protocol': 'ipproto_' + str(int(m[1])), 'src_role': prior.role(m[2]), 'dst_role': prior.role(m[3])}
        return packed(old.facts_text({k: v for k, v in facts.items() if k not in ('category', 'outcome')}), facts, 'asa_protocol', [{'span': [start, start + len(body)], 'kind': 'literal_protocol_denial'}])
    if body.startswith('[ Category =') or body.startswith(' [ Category ='):
        matches = list(re.finditer('\\[\\s*(FORMAT_MESSAGE|FORMAT_' + base.MARKER.pattern + ')\\s*=\\s*([^\\]]*)\\]', body))
        if len(matches) == 1:
            m = matches[0]
            return packed(old.payload_text(m[2].strip()), {}, 'format_payload', [{'span': [start + m.start(2), start + m.end(2)], 'kind': 'explicit_format_payload'}])
    return None

def network_fields(raw):
    a = raw.strip().split()
    if len(a) == 14 and a[0] == '2' and re.fullmatch('\\d{12}|-|unknown', a[1], re.I):
        if not (a[2].startswith('eni-') or base.MARKER.fullmatch(a[2]) or a[2] == '-'):
            return None
        for addr in a[3:5]:
            if addr != '-' and (not base.MARKER.search(addr)) and (not lexical.valid_ip(addr)):
                return None
        if a[12] not in ('ACCEPT', 'REJECT', '-'):
            return None
        facts = {}
        if a[12] in ('ACCEPT', 'REJECT'):
            facts.update(category='network', action='allow' if a[12] == 'ACCEPT' else 'deny', outcome='allowed' if a[12] == 'ACCEPT' else 'blocked')
        protocol = unsigned(a[7], 255)
        if protocol is not None:
            facts['protocol'] = {6: 'tcp', 17: 'udp', 1: 'icmp', 58: 'icmp6'}.get(protocol, 'ipproto_' + str(protocol))
        for key, i, maxn in [('src_port', 5, 65535), ('dst_port', 6, 65535), ('packets', 8, None), ('bytes', 9, None)]:
            n = unsigned(a[i], maxn)
            if n is not None:
                facts[key] = n
        text = old.facts_text({k: v for k, v in facts.items() if k not in PORTS})
        return packed(text, typed_facts(facts), 'vpc_v2', [{'span': [0, len(raw)], 'kind': 'fixed_v2_flow_fields'}], {'uninterpreted_log_status': a[13], 'port_states': [a[5] == '-', a[6] == '-']})
    m = re.fullmatch('<(?P<pri>\\d{1,3})>Original Address=\\S+\\s+1\\s+\\S+\\s+\\S+\\s+(?P<kind>flows|l7_firewall)\\s+(?:(?P<action>deny|allow)\\s+)?src=\\S+\\s+dst=\\S+(?:\\s+mac=\\S+)?\\s+protocol=(?P<protocol>tcp|udp|icmp|icmpv6|[0-9]{1,3})(?:\\s+sport=(?P<src>\\S+)\\s+dport=(?P<dst>\\S+))?(?:\\s+type=(?P<type>\\S+)(?:\\s+code=(?P<code>\\S+))?)?(?:\\s+pattern:\\s+(?P<pattern>.+)|\\s+decision=(?P<decision>blocked|allowed))?\\s*', raw.strip(), re.I)
    if m and int(m['pri']) <= 191:
        proto = m['protocol'].casefold()
        if proto.isdigit():
            if int(proto) > 255:
                return None
            proto = {6: 'tcp', 17: 'udp', 1: 'icmp', 58: 'icmp6'}.get(int(proto), 'ipproto_' + str(int(proto)))
        if proto == 'icmpv6':
            proto = 'icmp6'
        if m['kind'].casefold() == 'l7_firewall' and m['decision'] is None:
            return None
        if m['kind'].casefold() == 'flows' and m['pattern'] is None and (m['action'] is None):
            return None
        facts = {'category': 'network', 'protocol': proto}
        decision = m['decision'].casefold() if m['decision'] else None
        action = m['action'].casefold() if m['action'] else 'deny' if decision == 'blocked' else 'allow' if decision == 'allowed' else None
        if action:
            facts.update(action=action, outcome='blocked' if action == 'deny' else 'allowed')
        for field, key in [('src', 'src_port'), ('dst', 'dst_port')]:
            n = unsigned(m[field], 65535)
            if n is not None and proto in ('tcp', 'udp'):
                facts[key] = n
        for field, key in [('type', 'icmp_type'), ('code', 'icmp_code')]:
            n = unsigned(m[field], 255)
            if n is not None and proto in ('icmp', 'icmp6'):
                facts[key] = n
        return packed(old.facts_text({k: v for k, v in facts.items() if k not in PORTS}), typed_facts(facts), 'native_flow' if m['kind'].casefold() == 'flows' else 'native_firewall', [{'span': [m.start('kind'), len(raw.strip())], 'kind': 'native_packet_fields'}])
    return None

def bounded_json_payload(raw):
    props = lexical.members(raw)
    kept = {}
    evidence = []
    behavior_root = any((path == ('behaviors',) for a, b, path, begin in props))
    for a, b, path, begin in props:
        allowed = path == ('message',) or (behavior_root and path and (path[0] == 'behaviors') and (path[-1] in ('cmdline', 'filename', 'filepath', 'parent_cmdline', 'parent_filename', 'parent_filepath')))
        if not allowed:
            continue
        value = old.scalar(raw[begin:b])
        if not isinstance(value, str):
            continue
        key = '.'.join(map(str, path))
        kept.setdefault(key, []).append(old.payload_text(value))
        evidence.append({'field': key, 'span': [begin, b]})
    if not kept:
        return None
    text = ' '.join((k + ' ' + ' '.join(sorted(values)) for k, values in sorted(kept.items())))
    return packed(text, {}, 'bounded_payload', evidence, {'no_attack_truth_inferred': True})

@functools.lru_cache(maxsize=512)
def prepare_message(raw):
    if not raw.strip():
        return packed('', {}, 'empty', [], {'information_limited': True}, False)
    p = prior.asa_parts(raw)
    if p is not None:
        return from_old(old.asa(raw, p))
    auth = prior.auth_object(raw)
    if auth is not None:
        return from_old(old.authentication(raw, auth))
    v = cef_fields(raw)
    if v is not None:
        return v
    if '"winlog"' in raw:
        v = old.windows(raw, lexical.members(raw))
        if v is not None:
            return from_old(v)
    v = old.rendered_windows(raw)
    if v is not None:
        return from_old(v)
    for parser in (syslog_body, network_fields, bounded_json_payload):
        v = parser(raw)
        if v is not None:
            return v
    return packed('', {}, 'unsupported', [], {'information_limited': True, 'reason': 'no_verified_payload_boundary'}, False)
