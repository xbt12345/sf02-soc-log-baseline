"""Fixed raw-log feature decoder; input-only dependency closure."""
import re
from . import foundation as base
from . import acl_auth as legacy

def endpoint_facts(endpoint):
    """Conservative visible fields for collision diagnosis; identities excluded."""
    interface, sep, remainder = endpoint.partition(':')
    if not sep:
        return {'interface': None, 'port': None, 'port_state': 'unparsed'}
    address, slash, port = remainder.rpartition('/')
    if not slash:
        return {'interface': interface.casefold(), 'port': None, 'port_state': 'absent'}
    if base.MARKER.search(port):
        number, state = (None, 'redacted')
    elif re.fullmatch('[0-9]+', port) and 0 <= int(port) <= 65535:
        number, state = (int(port), 'observed')
    else:
        number, state = (None, 'invalid')
    return {'interface': interface.casefold(), 'port': number, 'port_state': state}

def asa_visible_facts(raw):
    """Not an attack label or complete behavioral context.

    Exact interface is retained in this audit fingerprint to avoid hiding a
    genuine role difference. Deployment role encoding remains a B2 decision.
    """
    p = legacy.asa_parts(raw)
    if p is None:
        return None
    f = p['facts']
    return {'action': f['action'].casefold(), 'protocol': f['protocol'].casefold(), 'src': endpoint_facts(f['src']), 'dst': endpoint_facts(f['dst']), 'icmp': f['icmp'], 'native_event': p['event_text'], 'aclword': f['aclword'], 'acl': f['acl'], 'acl_hashes': f['hex']}
