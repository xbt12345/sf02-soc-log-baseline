"""Final per-file metadata ceilings, with unchanged scientific arithmetic."""
from experiment_review import ROOT

def main():
    source=ROOT/'training/v169_prior_pair_training_entry_v11.py';target=ROOT/'training/v169_prior_pair_training_entry_v12.py';assert not target.exists();s=source.read_text(encoding='utf-8')
    old="    cap=8*1024**2 if path.name in ['fit.json','partial_inconclusive.json','completed_or_inconclusive.json'] else (16384 if path.name=='identity_repeat.json' else 65536)"
    new="""    caps={'fit.json':512*1024,'partial_inconclusive.json':4*1024**2,'completed_or_inconclusive.json':4*1024**2,'identity_repeat.json':16384,'proof.json':16384,'displacement_recipe.json':8192,'original_reference_identity.json':4096,'line_search.json':4096,'commit.json':4096,'full_original_quality_and_fixed_S_slices.json':16384,'original_unit_review.json':32768,'identity.json':32768,'same_parameter_replay.json':16384,'fit_counts.json':32768}
    cap=caps.get(path.name,8192 if path.name.startswith('class') and path.name.endswith('_repeat.json') else 65536)"""
    assert old in s;s=s.replace(old,new)
    old="        if len(encoded.encode('utf-8'))>512:raise RuntimeError('Registered counter-line size exceeded')"
    new="        cap=128 if value.get('kind') in ['head','feature'] else 512\n        if len(encoded.encode('utf-8'))>cap:raise RuntimeError('Registered counter-line size exceeded')"
    assert old in s;s=s.replace(old,new);compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target.relative_to(ROOT))
    source=ROOT/'training/v169_actual_backend_synthetic_qualification_v2.py';target=ROOT/'training/v169_actual_backend_synthetic_qualification_v3.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('v169_prior_pair_training_entry_v11','v169_prior_pair_training_entry_v12').replace('v169_actual_backend_synthetic_qualification_v2_20261002','v169_actual_backend_synthetic_qualification_v3_20261002');target.write_text(s,encoding='utf-8')
    source=ROOT/'training/v169_factorized_resource_budget_v4.py';target=ROOT/'training/v169_factorized_resource_budget_v5.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('v169_factorized_resource_budget_v4_20261002','v169_factorized_resource_budget_v5_20261002').replace('v169_prior_pair_training_entry_v11.py','v169_prior_pair_training_entry_v12.py').replace('V169_v11_factorized','V169_v12_factorized').replace('+40*225614','+120*225614').replace('allocation_and_directories=512*1024**2','allocation_and_directories=640*1024**2')
    old="    total=sum(parts.values());resources="
    new="""    metadata=dict(candidate_proofs=1146*16384,candidate_recipes=1146*8192,candidate_original_reference=1146*4096,line_search=912*4096,common_QP=114*65536,correction_QP=228*32768,target_repeats=252*8192,target_identities=126*32768,commits=120*4096,accepted_quality=120*16384,accepted_reference=240*8192,initial_protection=6*65536,final_replay=6*16384,fit_results=6*524288,fit_counts=6*32768,pair_results=3*4096,batch_results=2*4194304,failure_receipts=18*65536,head_and_feature_log_lines=4*56328*128,derivative_log_lines=2*29688*512,QP_log_lines=2*342*512,proposal_update_fit_log_lines=(1146+120+6)*512)
    parts['other_metadata_and_logs']=sum(metadata.values())
    total=sum(parts.values());resources="""
    assert old in s;s=s.replace(old,new).replace('storage_components_upper_bytes=parts,resources=resources','storage_components_upper_bytes=parts,metadata_components_upper_bytes=metadata,maximum_files=110000,maximum_directories=17000,filesystem_padding_bound_formula="110000*4096+17000*8192 < 640MiB",accepted_full_masks_three_copies_counted=True,resources=resources')
    compile(s,str(target),'exec');target.write_text(s,encoding='utf-8')

if __name__=='__main__':main()
