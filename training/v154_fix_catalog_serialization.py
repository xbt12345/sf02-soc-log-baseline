"""Preserve all catalog values and the byte cap; remove JSON formatting overhead."""
import json
from pathlib import Path
from experiment_review import read,sha,check_bindings
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/v154_directional_neighborhood_v2_20261001'

def main():
    p=ROOT/'mcp_readonly/catalog.json';backup=BASE/'catalog_publication_attempt1.json'
    assert not backup.exists() and not (BASE/'publication_format_repair.json').exists()
    before=p.read_bytes();data=json.loads(before.decode('utf-8'));after=(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
    assert len(before)>256*1024 and len(after)<=256*1024
    assert json.loads(after.decode('utf-8'))==data
    backup.write_bytes(before);p.write_bytes(after)
    result=dict(status='serialization_only_repair_complete_tests_pending',first_MCP_suite='10 tests, 1 failure 6 errors from catalog byte cap',
        first_catalog_bytes=len(before),current_catalog_bytes=len(after),all_parsed_catalog_values_identical=True,
        historical_documents_preserved=len(data['documents']),server_byte_cap_unchanged=256*1024,
        new_classifier_forwards=0,new_classifier_gradients=0,new_classifier_fits=0,new_updates=0,
        source_sha256={z.relative_to(ROOT).as_posix():sha(z) for z in [Path(__file__),p,backup,BASE/'publication.json',ROOT/'mcp_readonly/server.py']})
    (BASE/'publication_format_repair.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    check_bindings(result['source_sha256']);print(json.dumps({k:result[k] for k in ['status','first_catalog_bytes','current_catalog_bytes','historical_documents_preserved','all_parsed_catalog_values_identical']},ensure_ascii=False))

if __name__=='__main__':main()
