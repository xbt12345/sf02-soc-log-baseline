"""Fixed raw-log feature decoder; input-only dependency closure."""
import re
FRAGMENT = re.compile('((?:CRED|HOST|USER|ORG)-)([ \\t]*)(?=<IDENTITY>)')
CLUSTER = re.compile('[ \\t]*(?:(?:CRED|HOST|USER|ORG)-[ \\t]*)*<IDENTITY>[ \\t]*')

def normalize(text, mode):
    if mode == 'literal_only':
        return FRAGMENT.sub(lambda m: m.group(2), text)
    if mode == 'placeholder_cluster':
        return CLUSTER.sub(' <IDENTITY> ', text)
    raise ValueError(mode)
