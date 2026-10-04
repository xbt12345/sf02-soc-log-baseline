"""Bounded Windows authentication status repair; no label or identity inputs."""
import json
import re
import v39_core as prior

old = prior.previous.prior.old
lexical = prior.previous.prior.lexical
HEX = re.compile(r'0x[0-9a-f]{1,8}', re.I)
LINE = re.compile(r'(?im)^[ \t]*(?P<key>sub[ \t]*status|status)[ \t]*:[ \t]*(?P<value>0x[0-9a-f]{1,8})[ \t]*$')
FAILURE_BLOCK = re.compile(r'(?im)^[ \t]*failure reason:[^\r\n\"\']*\r?\n(?P<block>(?:[ \t]*(?:sub[ \t]*status|status)[ \t]*:[ \t]*0x[0-9a-f]{1,8}[ \t]*(?:\r?\n|$)){1,2})')
# This scan may veto a repair but NEVER supplies a model observation. A damaged
# parent cannot become a trusted native field merely because its child matches.
UNBOUNDED = re.compile(r'"(?P<key>status|substatus)"\s*:\s*"(?P<value>0x[0-9a-f]{1,8})"', re.I)


def extract(raw):
    result = {'state': 'outside_scope', 'facts': {}, 'carriers': [], 'native_title': None,
              'unbounded_candidate_fields': 0, 'reason': None}
    bodies = []
    props = []
    if '"winlog"' in raw:
        props = lexical.members(raw)
        native = old.native_auth_observation(raw, props=props)
        if native is None:
            result['reason'] = 'no_unique_native_auth_provider_code'
            return result
        for a, b, path, begin in props:
            if len(path) == 1 and (path[0] == 'message' or old.base.MARKER.fullmatch(path[0])):
                value = old.scalar(raw[begin:b])
                if isinstance(value, str) and len(value) > 40:
                    bodies.append((value, [begin, b]))
    else:
        native = old.native_auth_observation(raw, xml=True)
        if native is None:
            result['reason'] = 'no_unique_native_auth_provider_code'
            return result
        rendered = old.rendered_windows(raw)
        if rendered is not None:
            for e in rendered['evidence']:
                if e.get('kind') == 'bounded_rendered_plain_text':
                    a, b = e['span']
                    value = raw[a:b].replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\t', '\t')
                    bodies.append((value, [a, b]))
    if len(bodies) != 1:
        result['state'] = 'unknown'; result['reason'] = 'ambiguous_or_missing_native_body'
        return result
    body, span = bodies[0]
    title = body.splitlines()[0].strip().casefold()
    if title not in old.AUTH_TITLES:
        result['state'] = 'unknown'; result['reason'] = 'masked_or_unrecognized_native_title'
        return result
    if old.AUTH_TITLES[title] != native['outcome']:
        result['state'] = 'unknown'; result['reason'] = 'native_title_event_disagreement'
        return result
    result['native_title'] = title
    footer = re.search(r'(?im)^\s*(?:this event|'+old.base.MARKER.pattern+r'\s+'+old.base.MARKER.pattern+r')\s+is generated when\b', body)
    if footer:
        body = body[:footer.start()]
    seen = {'status': set(), 'substatus': set()}
    # Rendered codes must be in the contiguous native failure-information block,
    # before process/command sections. A quoted example elsewhere is not a field.
    blocks = list(FAILURE_BLOCK.finditer(body))
    process = re.search(r'(?im)^\s*(?:caller process|process information|command line|detailed authentication)\b', body)
    block = blocks[0] if len(blocks) == 1 and (process is None or blocks[0].start() < process.start()) else None
    block_text = block['block'] if block is not None else ''
    block_offset = block.start('block') if block is not None else 0
    for m in LINE.finditer(block_text):
        key = re.sub(r'\s+', '', m['key']).lower(); value = '0x%08x' % int(m['value'], 16)
        seen[key].add(value)
        result['carriers'].append({'field': key, 'value': value, 'carrier': 'native_rendered_line',
                                   'raw_container_span': span, 'decoded_value_span': [block_offset+m.start('value'), block_offset+m.end('value')]})
    for a, b, path, begin in props:
        if len(path) == 3 and path[0] == 'winlog' and path[-1] in seen:
            value = old.scalar(raw[begin:b])
            if isinstance(value, str) and HEX.fullmatch(value):
                value = '0x%08x' % int(value, 16); seen[path[-1]].add(value)
                result['carriers'].append({'field': path[-1], 'value': value, 'carrier': 'bounded_winlog_field',
                                           'raw_value_span': [begin, b]})
    # Treat any conflicting duplicate candidate conservatively, even if its
    # malformed parent prevents proving that it belongs to the native event.
    for m in UNBOUNDED.finditer(raw):
        key = m['key'].lower(); value = '0x%08x' % int(m['value'], 16)
        result['unbounded_candidate_fields'] += 1
        if seen[key] and value not in seen[key]:
            result['state'] = 'conflict'; result['reason'] = 'candidate_carrier_disagreement'
            return result
    if any(len(v) > 1 for v in seen.values()):
        result['state'] = 'conflict'; result['reason'] = 'bounded_carrier_disagreement'
        return result
    result['facts'] = {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}
    result['state'] = 'observed' if result['facts'] else 'unknown'
    if not result['facts']: result['reason'] = 'no_bounded_status_observation'
    return result


def repair(base, observation):
    result = {'text': base['text'], 'facts': dict(base['facts']), 'audit': {'native_auth': observation}}
    if observation['state'] != 'observed':
        return result  # Explicit B fallback; do not pretend legacy facts are newly verified.
    facts = observation['facts']
    # Replace only typed status slots; arbitrary hex strings in commands remain.
    # Matching values use a nonnumeric marker and the numeric code enters facts once.
    pattern = re.compile(r'(?i)(?<!\w)(sub\s*status|status)(\s*:\s*|\s+)(0x[0-9a-f]{1,8})(?![0-9a-f])')
    def replace(m):
        key = re.sub(r'\s+', '', m[1]).lower()
        if key in facts and int(m[3], 16) == int(facts[key], 16):
            return m[1]+m[2]+'native_code'
        return m[0]
    result['text'] = pattern.sub(replace, result['text'])
    result['facts'].update(facts)
    return result


def prepare_record(row):
    raw = prior.previous.prior.base.string(row.get('message_sanitized'))
    base = prior.prepare_message(raw)
    route = prior.previous.prior.prepare_message(raw)['route']
    if route not in ('windows_message', 'windows_rendered'):
        return base
    return repair(base, extract(raw))
