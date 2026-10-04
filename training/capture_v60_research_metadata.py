"""Record public model/code metadata only; never uploads local SOC content."""
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/2026-09-14/v60_research_metadata'
SOURCES = {
    'securebert_config.json': 'https://huggingface.co/cisco-ai/SecureBERT2.0-base/raw/main/config.json',
    'securebert_requirements.txt': 'https://raw.githubusercontent.com/cisco-ai-defense/securebert2/main/requirements.txt',
    'logllm_model.py.txt': 'https://raw.githubusercontent.com/guanwei49/LogLLM/master/model.py',
    'securebert_repository.json': 'https://api.github.com/repos/cisco-ai-defense/securebert2/commits/main',
    'logllm_repository.json': 'https://api.github.com/repos/guanwei49/LogLLM/commits/master',
    'securebert_model_metadata.json': 'https://huggingface.co/api/models/cisco-ai/SecureBERT2.0-base',
}


def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    receipts = []
    for name, url in SOURCES.items():
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SF02-public-research'})
            with urllib.request.urlopen(req, timeout=25) as response:
                b = response.read(2_000_001)
            assert len(b) <= 2_000_000
            (OUT / name).write_bytes(b)
            receipts.append({'file': name, 'url': url, 'bytes': len(b),
                'sha256': hashlib.sha256(b).hexdigest(), 'status': 'read'})
        except Exception as exc:
            receipts.append({'file': name, 'url': url, 'status': 'unavailable', 'error': str(exc)})
    result = {'accessed_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Public configuration/source metadata only. No model weights, training data, execution of downloaded code, or private uploads.', 'sources': receipts}
    (OUT / 'receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
