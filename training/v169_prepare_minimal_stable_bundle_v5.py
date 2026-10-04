"""Minimal rebind after an independently owned preseal reviewer stabilized."""
import ast,json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OLD=ROOT/'artifacts/v169_full_preseal_bundle_v4_20261002/bundle.json'
OUT=ROOT/'artifacts/v169_full_preseal_bundle_v5_20261002'

def main():
    assert not OUT.exists();old=read(OLD);bindings=dict(old['source_sha256'])
    reviewer='training/v169_root_final_preseal_review.py';old_reviewer_sha=bindings.pop(reviewer,None)
    assert old_reviewer_sha is not None
    # This is an independent, previously unexecuted reviewer. Its actual final
    # source identity belongs to its report, then the physical seal includes it.
    assert sha(ROOT/reviewer)=='1be96fb1b54369f5112fde4b185dedbdf6db898594fc31a20cefbc7c9f72e874'
    check_bindings(bindings)
    source=ROOT/'training/v169_seal_prior_pair_training_v3.py';target=ROOT/'training/v169_seal_prior_pair_training_v4.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('v169_full_preseal_bundle_v4_20261002','v169_full_preseal_bundle_v5_20261002');ast.parse(s);target.write_text(s,encoding='utf-8')
    for path in [OLD,Path(__file__).resolve(),target]:bindings[path.relative_to(ROOT).as_posix()]=sha(path)
    report=dict(old);report.update(source_sha256=bindings,physical_dependency_files=len(bindings),previous_immutable_bundle_path=OLD.relative_to(ROOT).as_posix(),previous_immutable_bundle_sha256=sha(OLD),removed_independent_active_reviewer_binding=dict(path=reviewer,previous_sha256=old_reviewer_sha,stable_current_sha256=sha(ROOT/reviewer),final_independent_report_will_bind_its_own_source=True),sealer_path=target.relative_to(ROOT).as_posix(),sealer_sha256=sha(target),same_v13_training_entry_schedule_and_qualifications=True)
    OUT.mkdir();(OUT/'bundle.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],physical_sources=len(bindings),bundle=(OUT/'bundle.json').relative_to(ROOT).as_posix(),sealer=target.relative_to(ROOT).as_posix(),official_calls=0)))

if __name__=='__main__':main()
