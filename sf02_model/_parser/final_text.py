"""Fixed raw-log feature decoder; input-only dependency closure."""
import functools
import ipaddress
import re
QUAD = re.compile('(?<![0-9.])[0-9]+(?:\\.[0-9]+){3}(?![0-9.])')
EXPLICIT_VERSION = re.compile('\\b(?:version|file_version|product_version|build)\\s*[:= ]\\s*(?:\\d+\\.){2,}\\d+', re.I)
COMPOUND_TAIL = re.compile('[0-9a-f]*:[0-9a-f:.]+', re.I)

@functools.lru_cache(maxsize=512)
def finalize_text(text):
    protected = [m.span() for m in EXPLICIT_VERSION.finditer(text)]
    changes = []
    for m in QUAD.finditer(text):
        if any((a <= m.start() < b for a, b in protected)):
            continue
        try:
            ipaddress.IPv4Address(m.group())
        except ValueError:
            continue
        end = m.end()
        tail = COMPOUND_TAIL.match(text, end)
        if tail:
            end = tail.end()
        changes.append({'start': m.start(), 'end': end, 'old_value': text[m.start():end], 'reason': 'unresolved_address_or_version_fragment', 'coordinate_space': 'v32_stage1_text'})
    if not changes:
        return (text, ())
    chunks = []
    pos = 0
    for change in changes:
        if change['start'] < pos:
            continue
        chunks.extend((text[pos:change['start']], 'unknown_dotted_value'))
        pos = change['end']
    chunks.append(text[pos:])
    return (''.join(chunks), tuple(changes))
