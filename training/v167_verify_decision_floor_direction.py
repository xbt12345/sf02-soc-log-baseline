"""Verify the prospective root plan and lossless current reader catalog."""
import json,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

def main():
    out=ROOT/'artifacts/v167_direction_20261002';target=out/'validation.json';assert not target.exists();publication=read(out/'publication.json');check_bindings(publication['source_sha256']);manifest=ROOT/'mcp_readonly/catalog.json';catalog=load_catalog(ROOT,manifest,256*1024)
    previous=load_catalog(ROOT,out/'previous/mcp_readonly/catalog.json',256*1024);old={r['id']:r for r in previous['documents']};now={r['id']:r for r in catalog['documents']};assert len(old)==450 and len(now)==451 and all(now[k]==v for k,v in old.items())
    for row in now.values():assert sha(ROOT/row['path'])==row['sha256'] and (ROOT/row['path']).stat().st_size<=256*1024
    assert catalog['project']['authoritative_direction_id']=='v168-plan' and catalog['project']['authoritative_delivery_id']=='v159-delivery'
    body=(out/'plan.md').read_bytes();plan=ROOT/'docs/V167_ROOT_RESULTS_AND_V168_DECISION_AWARE_FLOOR_PLAN_20261002.md';draft=ROOT/'training/review_policy/v168_decision_aware_floor_draft.json';registry=ROOT/'training/review_policy/v167_observed_runtime_boundaries.json';assert body.startswith(plan.read_bytes()) and draft.read_bytes() in body and registry.read_bytes() in body and not read(draft)['execution_authority']
    log=ROOT/'artifacts/v168_prospective_direction_MCP_tests_original_console_20261002.txt';text=log.read_text(encoding='utf-8');assert 'Ran 23 tests' in text and text.rstrip().endswith('OK')
    paths=[Path(__file__).resolve(),out/'publication.json',out/'plan.md',plan,draft,registry,log,manifest,ROOT/read(manifest)['historical_catalog']['path'],ROOT/'README.md',ROOT/'HANDOFF.md',ROOT/'mcp_readonly/tests/test_readonly_mcp.py']
    target.write_bytes((json.dumps(dict(status='V168_prospective_direction_exact_root_plan_450_preserved_records_and_23_local_MCP_tests_verified',historic_document_metadata_preserved=450,effective_documents=451,MCP_tests_passed=23,official_calls=0,execution_authority=False,full_quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}),ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status='V168_prospective_direction_verified',MCP_tests=23,official_calls=0)))

if __name__=='__main__':main()
