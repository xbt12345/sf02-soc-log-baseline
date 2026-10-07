"""Fixed raw-log feature decoder; input-only dependency closure."""
import json
import re
from . import foundation as base
from . import json_projection as old
from . import final_text as final
ASA_BODY = re.compile('(?P<action>Deny)\\s+(?P<protocol>tcp|udp|icmp6?|[0-9]+)\\s+src\\s+(?P<src>\\S+)\\s+dst\\s+(?P<dst>\\S+)(?P<icmp>\\s+\\(type\\s+\\S+,\\s*code\\s+\\S+\\))?\\s+by\\s+(?P<aclword>[A-Za-z0-9_-]+[-_]group)\\s+"(?P<acl>[^"\\r\\n]*)"\\s+\\[(?P<hex>0x[0-9a-f]+,\\s*0x[0-9a-f]+)\\]\\s*$', re.I)
ASA_CODE = re.compile('%(?:ASA|PIX)-([0-7])-([0-9]{6}):\\s*$', re.I)
WRAPPED_OBJECT = re.compile('(?:^|:::)\\s*[\\w-]+\\s*=\\s*(\\{)')
AUTH_SIGNATURE = {'application', 'user', 'result', 'reason', 'txid'}
AUTH_FACTS = {'result', 'reason', 'factor', 'event_type', 'remembered_factor'}
AUTH_UPSTREAM = {'adaptive_trust_assessments', 'rbfs_triggered_attacks', 'risk_score', 'trust_level', 'trusted_endpoint_status'}
AUTH_IDENTITIES = {'application', 'user', 'alias', 'txid', 'timestamp', 'isotimestamp', 'access_device', 'auth_device'}

def asa_parts(raw):
    """Only a complete observed ACL grammar after a checked collection prefix."""
    if 'deny' not in raw.casefold():
        return None
    matches = list(ASA_BODY.finditer(raw))
    if len(matches) != 1:
        return None
    m = matches[0]
    prefix = raw[:m.start()]
    event = ASA_CODE.search(prefix)
    event_text = ''
    if event:
        event_text = 'event_code {} native_severity {} '.format(event.group(2), event.group(1))
        prefix = prefix[:event.start()]
    if prefix.strip():
        check = re.sub('^\\s*<\\d{1,3}>\\s*(?:1\\s+)?', '', prefix)
        found_time = False
        for pat in (base.PARTIAL_DATE, base.ISO, base.SYSLOG_DATE, base.CLOCK):
            check, n = pat.subn(' TIME ', check)
            found_time |= n > 0
        check = base.MARKER.sub(' ENTITY ', check)
        tokens = check.replace(':', ' ').split()
        others = [t for t in tokens if t not in {'TIME', 'ENTITY', '-'}]
        if not found_time or len(others) > 2 or any((not re.fullmatch('[A-Za-z][\\w.-]*', t) for t in others)):
            return None
    return {'body_start': m.start(), 'body_end': len(raw), 'body': raw[m.start():], 'event_text': event_text, 'facts': m.groupdict()}

def role(endpoint):
    value = endpoint.split(':', 1)[0].casefold()
    if value in {'inside', 'outside', 'identity', 'internet'}:
        return value
    if re.fullmatch('dmz(?:[-_]\\d+)?', value):
        return 'dmz'
    return 'other_interface'

def auth_object(raw):
    if not all(('"' + k + '"' in raw for k in AUTH_SIGNATURE)):
        return None
    candidates = [m.start(1) for m in WRAPPED_OBJECT.finditer(raw)]
    if raw.lstrip().startswith('{'):
        candidates.append(len(raw) - len(raw.lstrip()))
    accepted = []
    for start in candidates:
        end = old.value_end(raw, start)
        fragment = raw[start:end] if end is not None else raw[start:]
        props = old.members(fragment)
        roots = [p for p in props if len(p[2]) == 1]
        keys = [p[2][0] for p in roots]
        partial_tail = None
        if end is None and roots and (AUTH_SIGNATURE - {'user'}).issubset(keys):
            last = max((q[1] for q in roots))
            tail = re.match('\\s*,\\s*"user"\\s*:\\s*\\{', fragment[last:])
            if tail and 'user' not in keys:
                partial_tail = last
                end = len(raw)
        enough = AUTH_SIGNATURE.issubset(keys) or partial_tail is not None
        if enough and end is not None and (len(keys) == len(set(keys))):
            accepted.append((start, end, fragment, roots, partial_tail))
    return accepted[0] if len(accepted) == 1 else None

def auth_view(found):
    start, end, fragment, roots, partial_tail = found
    facts = {}
    removed = []
    unknown = []
    for a, b, path, begin in roots:
        key = path[0]
        if key in AUTH_FACTS:
            try:
                value = json.loads(fragment[begin:b])
            except (ValueError, TypeError):
                value = None
            if not isinstance(value, str) or base.MARKER.search(value):
                unknown.append(key)
                value = 'unknown_value'
            facts[key] = value
        else:
            reason = 'upstream_or_post_event' if key in AUTH_UPSTREAM else 'identity_or_collection' if key in AUTH_IDENTITIES else 'unverified_field'
            removed.append({'start': start + a, 'end': start + b, 'path': key, 'reason': reason})
    if partial_tail is not None:
        removed.append({'start': start + partial_tail, 'end': end, 'path': 'unparsed_identity_tail', 'reason': 'unverified_field'})
    text = ' '.join(('{} {}'.format(k, facts[k]) for k in sorted(facts)))
    text = final.finalize_text(old.normalize_body(text, 'json_fragment'))[0]
    return (text, facts, removed, unknown)
