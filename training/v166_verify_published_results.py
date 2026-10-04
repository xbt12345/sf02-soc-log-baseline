"""Verify saved publication, preserved historical records and actual MCP log."""
import json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

def main():
    out=ROOT/'artifacts/v166_results_20261002';target=out/'validation.json';assert not target.exists();publication=read(out/'publication.json');check_bindings(publication['source_sha256']);actual=read(out/'actual_result_summary.json');check_bindings(actual['source_sha256']);previous=read(out/'previous/mcp_readonly/catalog.json');current=read(ROOT/'mcp_readonly/catalog.json');historic={r['id']:r for r in previous['documents']};now={r['id']:r for r in current['documents']}
    assert len(historic)==444 and len(now)==446 and all(now[key]==value for key,value in historic.items());assert current['project']['authoritative_delivery_id']=='v159-delivery' and current['project']['authoritative_direction_id']=='v166-review';assert (ROOT/'mcp_readonly/catalog.json').stat().st_size==publication['catalog_bytes']<=256*1024
    for row in now.values():assert sha(ROOT/row['path'])==row['sha256'] and (ROOT/row['path']).stat().st_size<=256*1024
    report=(out/'review.md').read_text(encoding='utf-8');assert all(value in report for value in previous['project']['known_limits']);assert report==(ROOT/'docs/V166_COMPLETE_COVERAGE_DIAGNOSTIC_RESULTS_20261002.md').read_text(encoding='utf-8')
    log=ROOT/'artifacts/v166_complete_results_MCP_tests_original_console_20261002.txt';text=log.read_text(encoding='utf-8');assert 'Ran 19 tests' in text and text.rstrip().endswith('OK');assert not actual['supports_new_short_training_registration'] and actual['cumulative_heads']==18784 and actual['cumulative_all_complete_derivatives']==618 and actual['new_fits']==actual['permanent_updates']==0
    paths=[Path(__file__).resolve(),log,out/'publication.json',out/'actual_result_summary.json',out/'review.md',ROOT/'README.md',ROOT/'HANDOFF.md',ROOT/'mcp_readonly/catalog.json',ROOT/'mcp_readonly/tests/test_readonly_mcp.py'];bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths};target.write_text(json.dumps(dict(status='V166_complete_actual_results_publication_and_19_MCP_tests_verified',catalog_bytes=publication['catalog_bytes'],historical_document_metadata_preserved=444,new_documents=2,all_old_limits_preserved_in_report=True,current_direction='v166-review',latest_actual_training='V164',latest_complete_quality_delivery='V159',MCP_tests_passed=19,official_calls_by_verification=0,new_fits=0,permanent_updates=0,root_goal_complete=False,source_sha256=bindings),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status='V166_publication_verified',MCP_tests=19,official_calls=0)))

if __name__=='__main__':main()
