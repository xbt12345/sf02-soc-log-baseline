"""Verify exact root plan, frozen cohorts, historic documents and local tools."""
import json,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

def main():
    out=ROOT/'artifacts/v168_direction_20261002';target=out/'validation.json';assert not target.exists();publication=read(out/'publication.json');check_bindings(publication['source_sha256']);manifest=ROOT/'mcp_readonly/catalog.json';current=load_catalog(ROOT,manifest,256*1024);before=load_catalog(ROOT,out/'previous/mcp_readonly/catalog.json',256*1024);historic={r['id']:r for r in before['documents']};now={r['id']:r for r in current['documents']};assert len(historic)==454 and len(now)==455 and all(now[k]==v for k,v in historic.items())
    for row in now.values():assert sha(ROOT/row['path'])==row['sha256'] and (ROOT/row['path']).stat().st_size<=256*1024
    archive=ROOT/read(manifest)['historical_catalog']['path'];assert archive.stat().st_size==262091<=256*1024 and sha(archive)==read(manifest)['historical_catalog']['sha256'];assert current['project']['authoritative_direction_id']=='v169-plan' and current['project']['authoritative_delivery_id']=='v159-delivery'
    body=(out/'plan.md').read_bytes();rootplan=ROOT/'docs/V168_INDEPENDENT_RESULTS_AND_NEXT_TRAINING_DECISION_20261002.md';draft=ROOT/'training/review_policy/v169_learnable_prior_pair_draft.json';assert body.startswith(rootplan.read_bytes()) and not read(draft)['execution_authority'] and not read(draft)['new_fit_permission'];check_bindings(read(draft)['source_sha256'])
    for source in [draft,ROOT/'training/review_policy/v168_observed_runtime_boundaries.json',ROOT/'artifacts/v168_saved_accepted_learning_trajectory_review_20261002/review.json',ROOT/'artifacts/v169_saved_prospective_budget_review_20261002/review.json']:assert source.read_bytes() in body
    for cohort in read(draft)['initial_S_cohorts']:assert sha(ROOT/cohort['path'])==cohort['sha256']
    old=(ROOT/'artifacts/v168_results_20261002/review.md').read_text(encoding='utf-8').split('## 既有证据限制完整保留\n\n',1)[1];assert old in body.decode('utf-8') and all(v in body.decode('utf-8') for v in before['project']['known_limits'])
    log=ROOT/'artifacts/v169_prospective_direction_MCP_tests_original_console_20261002.txt';text=log.read_text(encoding='utf-8');assert 'Ran 23 tests' in text and text.rstrip().endswith('OK');assert (ROOT/'mcp_readonly/catalog.json').stat().st_size==publication['catalog_bytes']<=256*1024
    paths=[Path(__file__).resolve(),out/'publication.json',out/'plan.md',rootplan,draft,log,manifest,archive,ROOT/'README.md',ROOT/'HANDOFF.md',ROOT/'mcp_readonly/tests/test_readonly_mcp.py'];target.write_bytes((json.dumps(dict(status='V169_exact_root_direction_frozen_S_cohorts_454_preserved_records_and23_local_MCP_tests_verified',historical_document_metadata_preserved=454,effective_documents=455,MCP_tests_passed=23,current_direction='v169-plan',latest_actual_training='V164',latest_complete_quality_delivery='V159',official_calls=0,new_fits=0,permanent_updates=0,execution_authority=False,method_effect_not_claimed=True,classification_mastery=False,quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}),ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status='V169_prospective_direction_verified',historical_records=454,MCP_tests=23,official_calls=0)))

if __name__=='__main__':main()
