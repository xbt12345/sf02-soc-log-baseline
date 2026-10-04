"""Preserve failed zero-compute reader; zero-update stages need no progress file."""
import json
from pathlib import Path
from experiment_review import ROOT,read,sha

def main():
    original=ROOT/'training/v158_assemble_legal_bank.py';old=ROOT/'artifacts/v158_legal_fusion_bank_20261001'
    assert old.is_dir() and not list(old.iterdir())
    missing=ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001/outer0_inner2/V140_C_readout/progress.json'
    assert not missing.exists() and read(missing.with_name('fit.json'))['accepted_updates']==0
    failure=dict(status='saved_array_binding_reader_failed_before_any_array_recombination',
        error_type='FileNotFoundError',missing_path=missing.relative_to(ROOT).as_posix(),
        source_sha256=sha(original),official_model_calls=0,new_features_calls=0,new_gradients=0,new_fits=0,new_updates=0,
        saved_score_softmax_recalculations=0,failed_before_prospective_array_bindings_written=True,
        source_and_failed_directory_preserved=True)
    (old/'failure.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    source=original.read_text(encoding='utf-8').replace("OUT=ROOT/'artifacts/v158_legal_fusion_bank_20261001'","OUT=ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001'")
    source=source.replace("'mean_probability.npy','progress.json','FIT_reference.parquet'","'mean_probability.npy','FIT_reference.parquet'")
    source=source.replace("paths.add(folder/file)\n                if name", "paths.add(folder/file)\n                if (folder/'progress.json').exists():paths.add(folder/'progress.json')\n                else:assert read(folder/'fit.json')['accepted_updates']==0\n                if name")
    target=ROOT/'training/v158_assemble_legal_bank_v2.py';assert not target.exists();target.write_text(source,encoding='utf-8')
    print(dict(read_only_assembly_retry_created=True,no_actual_training_repeated=True,new_fits=0,new_updates=0))

if __name__=='__main__':main()
