"""Verify lossless reader history, actual report and the complete local MCP run."""
import json,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

def main():
    out=ROOT/'artifacts/v167_results_20261002';target=out/'validation.json';assert not target.exists()
    publication=read(out/'publication.json');repair_path=ROOT/'artifacts/v167_publication_test_repair_20261002/repair.json';repair=read(repair_path);check_bindings(repair['source_sha256'])
    original_bindings=dict(publication['source_sha256']);original_test_sha=original_bindings.pop('mcp_readonly/tests/test_readonly_mcp.py');assert sha(repair_path.parent/'test_readonly_mcp_before_repair.py')==original_test_sha;check_bindings(original_bindings)
    actual=read(out/'actual_result_summary.json');check_bindings(actual['source_sha256'])
    previous_manifest=read(out/'previous/mcp_readonly/catalog.json');archive=ROOT/previous_manifest['historical_catalog']['path'];previous=read(archive);current=load_catalog(ROOT,ROOT/'mcp_readonly/catalog.json',256*1024)
    old={r['id']:r for r in previous['documents']};now={r['id']:r for r in current['documents']}
    assert len(old)==447 and len(now)==450 and all(now[k]==v for k,v in old.items())
    assert sha(archive)==previous_manifest['historical_catalog']['sha256'] and archive.stat().st_size==262091<=256*1024
    assert current['project']['authoritative_delivery_id']=='v159-delivery' and current['project']['authoritative_direction_id']=='v167-review'
    assert (ROOT/'mcp_readonly/catalog.json').stat().st_size==publication['catalog_bytes']<=256*1024
    for row in now.values():assert sha(ROOT/row['path'])==row['sha256'] and (ROOT/row['path']).stat().st_size<=256*1024
    report=(out/'review.md').read_bytes();assert report==(ROOT/'docs/V167_COMPLETE_TRIAL_POINT_RESTORATION_RESULTS_20261002.md').read_bytes()
    text=report.decode('utf-8');assert all(v in text for v in previous_manifest['project']['known_limits'])
    retained=(ROOT/'artifacts/v166_direction_20261002/plan.md').read_text(encoding='utf-8').split('## 既有证据限制完整保留\n\n',1)[1];assert retained in text
    log=ROOT/'artifacts/v167_complete_results_MCP_tests_v2_original_console_20261002.txt';console=log.read_text(encoding='utf-8');assert 'Ran 23 tests' in console and console.rstrip().endswith('OK')
    assert actual['actual_new_heads']==296 and actual['actual_new_complete_margin_derivatives']==100 and actual['actual_QP_solves']==2 and actual['actual_finite_proposals']==4
    assert actual['cumulative_heads']==19080 and actual['cumulative_all_complete_derivatives']==718 and not actual['supports_new_short_training_registration'] and actual['new_fits']==actual['permanent_updates']==0
    assert actual['all_restored_full_tensors_exact'] and actual['all_restored_original_argmax_exact'] and actual['latest_actual_training']=='V164' and actual['latest_complete_quality_delivery']=='V159'
    assert sha(ROOT/'training/experiment_review.py')=='a0faa3312cbb39d3c6bcb62d7a2b290fe18eae14579278b0ff26479fdf2b768e'
    paths=[Path(__file__).resolve(),log,repair_path,out/'publication.json',out/'actual_result_summary.json',out/'review.md',ROOT/'README.md',ROOT/'HANDOFF.md',ROOT/'mcp_readonly/catalog.json',archive,ROOT/'mcp_readonly/server.py',ROOT/'mcp_readonly/catalog_io.py',ROOT/'mcp_readonly/tests/test_readonly_mcp.py',ROOT/'mcp_readonly/tests/test_catalog_archive.py']
    target.write_bytes((json.dumps(dict(status='V167_full_actual_publication_447_historic_sources_and_23_local_MCP_tests_verified',catalog_bytes=publication['catalog_bytes'],historical_archive_bytes=archive.stat().st_size,historical_document_metadata_preserved=447,new_documents=3,effective_documents=450,all_old_source_hashes_and_limits_preserved=True,current_direction='v167-review',latest_actual_training='V164',latest_complete_quality_delivery='V159',MCP_tests_passed=23,MCP_validation_scope='local_TestClient_and_tool_interface_only_no_cloud_live_acceptance',official_calls_by_verification=0,new_fits=0,permanent_updates=0,root_goal_complete=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}),ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(status='V167_actual_publication_verified',historical_records=447,MCP_tests=23,official_calls=0)))

if __name__=='__main__':main()
