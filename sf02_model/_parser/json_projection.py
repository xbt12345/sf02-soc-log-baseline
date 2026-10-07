"""Fixed raw-log feature decoder; input-only dependency closure."""
import ipaddress
import json
import re
from . import foundation as base
IP_CANDIDATE = re.compile('(?<![\\w.])(?:\\d{1,3}\\.){3}\\d{1,3}(?![\\w.])')
ESCAPED_IP = re.compile('(\\\\[nrt])((?:\\d{1,3}\\.){3}\\d{1,3})(?![\\w.])')

def quoted_end(raw, start):
    if raw[start:start + 1] != '"':
        return None
    i = start + 1
    while i < len(raw):
        if raw[i] == '\\':
            i += 2
        elif raw[i] == '"':
            return i + 1
        else:
            i += 1
    return None

def value_end(raw, start):
    if start >= len(raw):
        return None
    if raw[start] == '"':
        return quoted_end(raw, start)
    if raw[start] in '[{':
        stack = [raw[start]]
        i = start + 1
        while i < len(raw):
            if raw[i] == '"':
                i = quoted_end(raw, i)
                if i is None:
                    return None
                continue
            if raw[i] in '[{':
                stack.append(raw[i])
            elif raw[i] in ']}':
                if not stack or (stack[-1], raw[i]) not in [('[', ']'), ('{', '}')]:
                    return None
                stack.pop()
                if not stack:
                    return i + 1
            i += 1
        return None
    i = start
    while i < len(raw) and raw[i] not in ',}]\r\n':
        i += 1
    return i if i > start else None

def object_root(raw):
    previous = base.object_root(raw)
    if previous is not None:
        return previous
    if re.match('\\s*(?:USER|ORG|CRED)-[\\w.-]+\\s*:::', raw, re.I):
        for m in re.finditer(':::\\s*([A-Za-z0-9_-]+)\\s*=\\s*(\\{)', raw):
            if not base.MARKER.fullmatch(m.group(1)):
                continue
            root = m.start(2)
            k = root + 1
            while k < len(raw) and raw[k].isspace():
                k += 1
            end = quoted_end(raw, k)
            if end is not None and re.match('\\s*:', raw[end:]) and (raw[k:end] == '"behaviors"'):
                last = value_end(raw, root)
                if last is not None and 'crowdstrike' in (raw[:root] + raw[last:]).casefold():
                    return root
    return None

def members(raw):
    """Lexical field spans survive invalid value escapes; no guessed decoding."""
    result = []

    def walk(start, stop, path):
        i = start + 1
        while i < stop:
            if raw[i] != '"':
                i += 1
                continue
            after = quoted_end(raw, i)
            if after is None or after > stop:
                break
            try:
                key = json.loads(raw[i:after])
            except ValueError:
                i = after
                continue
            j = after
            while j < stop and raw[j].isspace():
                j += 1
            if j >= stop or raw[j] != ':':
                i = after
                continue
            begin = j + 1
            while begin < stop and raw[begin].isspace():
                begin += 1
            end = value_end(raw, begin)
            if end is None or end > stop:
                if begin < stop and raw[begin] == '"':
                    break
                i = after
                continue
            full_path = path + (str(key).casefold(),)
            result.append((i, end, full_path, begin))
            if raw[begin:begin + 1] == '{':
                walk(begin, end, full_path)
            elif raw[begin:begin + 1] == '[':
                k = begin + 1
                while k < end:
                    if raw[k] in '[{"':
                        last = value_end(raw, k)
                        if last is None:
                            break
                        if raw[k] == '{':
                            walk(k, last, full_path + ('[]',))
                        k = last
                    else:
                        k += 1
            i = end
    start = object_root(raw)
    if start is not None:
        walk(start, value_end(raw, start) or len(raw), ())
    return result

def valid_ip(value):
    try:
        ipaddress.IPv4Address(value)
        return True
    except ipaddress.AddressValueError:
        return False

def normalize_body(raw, fmt):
    raw = re.sub('\\b(?:sport|dport|src_port|dst_port)=([^\\s,;]+)', lambda m: m.group(0).split('=', 1)[0] + '=unknown_port' if base.MARKER.search(m.group(1)) else m.group(0), raw, flags=re.I)
    if fmt == 'asa_like':
        raw = re.sub('\\b(src|dst)\\s+(\\S+)', lambda m: m.group(1) + ' ' + (m.group(2).rsplit('/', 1)[0] + '/unknown_port' if '/' in m.group(2) and base.MARKER.search(m.group(2).rsplit('/', 1)[1]) else m.group(2)), raw, flags=re.I)
    if fmt == 'vpc14':
        toks = raw.split()
        if len(toks) == 14:
            for i in (1, 2, 3, 4):
                toks[i] = 'entity'
            for i in (5, 6, 7, 8, 9):
                if base.MARKER.search(toks[i]):
                    toks[i] = 'unknown_value'
            toks[10] = toks[11] = 'time'
            raw = ' '.join(toks)
    for pattern in (base.PARTIAL_DATE, base.ISO, base.SYSLOG_DATE, base.CLOCK):
        raw = pattern.sub('time', raw)
    protected = []

    def keep_version(m):
        protected.append(m.group(0))
        return 'versionvalueplaceholderz' + str(len(protected) - 1) + 'z'
    raw = re.sub('\\b(?:version|file_version|product_version|build)\\s*[:= ]\\s*(?:\\d+\\.){2,}\\d+', keep_version, raw, flags=re.I)
    raw = ESCAPED_IP.sub(lambda m: m.group(1) + ('entity' if valid_ip(m.group(2)) else m.group(2)), raw)
    raw = IP_CANDIDATE.sub(lambda m: 'entity' if valid_ip(m.group()) else m.group(), raw)
    for pattern in (base.UUID, base.MAC, base.FQDN, base.LONG_HEX):
        raw = pattern.sub('entity', raw)
    raw = base.MARKER.sub('entity', raw)
    raw = base.KV_ID.sub(lambda m: m.group(0).split('=', 1)[0].strip() + '=entity', raw)
    raw = re.sub('^\\s*<\\d{1,3}>\\s*(?:1\\s+)?', '', raw)
    raw = re.sub('\\s+', ' ', raw).strip().casefold()
    for i, value in enumerate(protected):
        raw = raw.replace('versionvalueplaceholderz' + str(i) + 'z', value.casefold())
    return raw
