"""Prepare new qualifications without editing already executed sources."""
from experiment_review import ROOT

def main():
    source=ROOT/'training/v169_actual_backend_synthetic_qualification.py';target=ROOT/'training/v169_actual_backend_synthetic_qualification_v2.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('v169_prior_pair_training_entry_v10','v169_prior_pair_training_entry_v11').replace('v169_actual_backend_synthetic_qualification_20261002','v169_actual_backend_synthetic_qualification_v2_20261002').replace('v169_pair_execution_review_v2.py','v169_pair_execution_review_v3.py')
    s=s.replace("plan=dict(role_caps={'0':caps})","plan=dict(role_caps={'0':caps},resources=dict(fixed_free_disk_reserve_bytes=2*1024**3))")
    s=s.replace("rv,q,lp,_=risk(model,ctx,ids);dq,dlp=original.probabilities(model,ctx,'deployment',np.arange(22546),True) if False else (None,None)","rv,q,lp,_=risk(model,ctx,ids)")
    old="backend.last_observation=old;count=backend.counter.counts();backend.close_resources()"
    new="""backend.last_observation=old;count=backend.counter.counts()
            backend.counter.reserve=10**30
            try:backend.counter.enter('proposal')
            except RuntimeError as e:assert 'disk reserve exhausted' in str(e)
            else:raise AssertionError('Reserve violation must stop before a model call')
            assert backend.counter.counts()==count
            backend.close_resources()"""
    assert old in s;s=s.replace(old,new);compile(s,str(target),'exec');target.write_text(s,encoding='utf-8')
    source=ROOT/'training/v169_factorized_resource_budget_v3.py';target=ROOT/'training/v169_factorized_resource_budget_v4.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('v169_factorized_resource_budget_v3_20261002','v169_factorized_resource_budget_v4_20261002').replace('v169_prior_pair_training_entry_v9.py','v169_prior_pair_training_entry_v11.py').replace('V169_v9_factorized_whole_run_resource_candidate_pending_runtime_caps_and_independent_review','V169_v11_factorized_whole_run_resource_candidate_runtime_caps_implemented_pending_independent_review')
    s=s.replace("JSON_and_final_table_caps_must_be_runtime_enforced=True", "JSON_and_final_table_caps_runtime_enforced=True,per_operation_fixed_disk_reserve_enforced=True")
    s=s.replace("assert 'factorized_scope' in text", "assert 'Registered JSON size bound exceeded' in text and 'Final complete table storage bound exceeded' in text and 'resource_check()' in text\n    assert 'factorized_scope' in text")
    compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print('Prepared final backend v2 and resource v4 qualifications')

if __name__=='__main__':main()
