"""Fixed raw-log feature decoder; input-only dependency closure."""
import math
import re
MARKER = re.compile('(?:USER|ORG|CRED)(?:-(?:USER|ORG|CRED))*-\\d+(?:-\\d+)*', re.I)
UUID = re.compile('\\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\\b', re.I)
MAC = re.compile('\\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\\b', re.I)
ISO = re.compile('\\b\\d{4}-\\d{2}-\\d{2}(?:[Tt ]\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?(?:[Zz]|[+-]\\d{2}:?\\d{2})?)?\\b')
PARTIAL_DATE = re.compile('(?:USER|ORG|CRED)-\\d+(?:-\\d+)*-\\d{2}-\\d{2}[Tt ]\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?(?:[Zz]|[+-]\\d{2}:?\\d{2})?', re.I)
SYSLOG_DATE = re.compile('\\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+\\d{1,2}(?:\\s+\\d{4})?(?:\\s+\\d{2}:\\d{2}:\\d{2})?\\b', re.I)
CLOCK = re.compile('\\b\\d{1,2}:\\d{2}:\\d{2}(?:\\.\\d+)?\\b')
LONG_HEX = re.compile('\\b[0-9a-f]{16,}\\b', re.I)
FQDN = re.compile('(?<![\\w.])(?:[a-z0-9][a-z0-9-]*\\.)+(?:com|net|org|local|internal|example|invalid)(?![\\w.])', re.I)
KV_ID = re.compile('\\b(?:src_ip|dst_ip|src_host|dst_host|username|user_name|hostname)\\s*=\\s*(?:\\"[^\\"]*\\"|[^\\s,;]+)', re.I)

def string(value):
    return '' if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)

def object_root(raw):
    start = len(raw) - len(raw.lstrip())
    if raw[start:start + 1] == '{':
        return start
    first = raw.find('{')
    if first < 0:
        return None
    prefix = raw[:first]
    if re.fullmatch('\\s*(?:USER|ORG|CRED)-[\\w.-]+\\s*:::\\s*', prefix, re.I):
        return first
    if re.match('\\s*(?:USER|ORG|CRED)-', prefix, re.I) and 'crowdstrike' in prefix.casefold() and re.search(':::\\s*(?:USER|ORG|CRED)-[\\w.-]+=\\s*$', prefix, re.I):
        return first
    return None
