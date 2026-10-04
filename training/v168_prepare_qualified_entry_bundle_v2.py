"""Bind the completed qualification bundle plus independent reader and pending seal."""
import ast,json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v168_decision_floor_qualification_bundle_v2_20261002'

def main():
    assert not OUT.exists()
    prior=ROOT/'artifacts/v168_decision_floor_qualification_bundle_20261002/bundle.json';value=read(prior);check_bindings(value['source_sha256']);assert value['execution_authority'] is False
    reader=ROOT/'training/v168_independent_actual_decision_floor_review.py';sealer=ROOT/'training/v168_seal_decision_floor_diagnostic.py'
    for p in [reader,sealer,Path(__file__).resolve()]:ast.parse(p.read_text(encoding='utf-8-sig'))
    assert not (ROOT/'artifacts/v168_independent_actual_decision_floor_review_20261002').exists() and not (ROOT/'artifacts/v168_decision_floor_diagnostic_20261002').exists() and not (ROOT/'training/review_policy/v168_decision_floor_contract.json').exists()
    sources={ROOT/p for p in value['source_sha256']}|{prior,reader,sealer,Path(__file__).resolve()};bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)};check_bindings(bindings)
    value.update(independent_actual_reader_source=reader.relative_to(ROOT).as_posix(),independent_actual_reader_only_parsed_not_executed=True,pending_sealer_source=sealer.relative_to(ROOT).as_posix(),physical_sealer_parsed_not_executed=True,sealer_requires_independent_root_supports_physical_seal_report=True,prior_immutable_bundle=prior.relative_to(ROOT).as_posix(),source_sha256=bindings)
    OUT.mkdir();(OUT/'bundle.json').write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status='V168_qualified_bundle_v2_binds_root_reader_and_pending_sealer',sources=len(bindings),official_calls=0,execution_authority=False)))

if __name__=='__main__':main()
