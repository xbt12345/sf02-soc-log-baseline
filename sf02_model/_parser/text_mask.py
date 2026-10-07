"""Fixed raw-log feature decoder; input-only dependency closure."""
import re
EMBEDDED = re.compile('(?:USER|HOST|CRED|ORG)-[0-9]+(?:-[0-9]+)*')

def stable(text):
    return EMBEDDED.sub(' <IDENTITY> ', text)
