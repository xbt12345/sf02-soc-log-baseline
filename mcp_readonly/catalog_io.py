"""Read one bounded manifest plus one hash-bound historical catalog."""
from __future__ import annotations
import hashlib,hmac,json,re
from pathlib import Path
from typing import Any

class CatalogReadError(ValueError):
    pass

def load_catalog(project_root: Path,catalog_path: Path,max_bytes: int) -> dict[str,Any]:
    def parse(raw: bytes) -> dict[str,Any]:
        if len(raw)>max_bytes:raise CatalogReadError('Catalog is unexpectedly large')
        try:value=json.loads(raw.decode('utf-8'))
        except (ValueError,UnicodeError) as exc:raise CatalogReadError('Catalog must be UTF-8 JSON') from exc
        if not isinstance(value,dict) or value.get('schema_version')!=1:raise CatalogReadError('Unsupported catalog schema')
        if not isinstance(value.get('documents'),list):raise CatalogReadError('Catalog documents must be a list')
        return value
    data=parse(catalog_path.read_bytes());reference=data.get('historical_catalog')
    if reference is None:return data
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:raise CatalogReadError('Invalid historical catalog reference')
    relative=Path(str(reference['path']));expected=str(reference['sha256']).lower()
    if relative.is_absolute() or '..' in relative.parts or relative.suffix.lower()!='.json' or not re.fullmatch('[0-9a-f]{64}',expected):raise CatalogReadError('Unsafe historical catalog reference')
    lexical=project_root/relative
    if lexical.is_symlink():raise CatalogReadError('Symbolic links are not allowed')
    try:
        resolved=lexical.resolve(strict=True);resolved.relative_to(project_root.resolve(strict=True))
    except (OSError,ValueError) as exc:raise CatalogReadError('Historical catalog path escapes or is missing') from exc
    if not resolved.is_file() or resolved.stat().st_size>max_bytes:raise CatalogReadError('Historical catalog exceeds the unchanged file limit')
    raw=resolved.read_bytes()
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(),expected):raise CatalogReadError('Historical catalog integrity check failed')
    historical=parse(raw)
    if historical.get('historical_catalog') is not None:raise CatalogReadError('Nested historical catalogs are not allowed')
    result=dict(data);result['documents']=list(data['documents'])+list(historical['documents']);ids=set()
    for entry in result['documents']:
        identifier=entry.get('id') if isinstance(entry,dict) else None
        if not isinstance(identifier,str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,63}',identifier):raise CatalogReadError('Invalid historical document id')
        if identifier in ids:raise CatalogReadError('Duplicate document id: '+identifier)
        ids.add(identifier)
    return result
