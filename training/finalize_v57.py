"""Close the completed staged study without promoting failed candidates."""
import json
import re
from pathlib import Path
import pandas as pd
import run_v54 as v
import run_v56 as t
import run_v57_stage1 as a
import run_v57_stage2 as b
import run_v57_port_transfer as c
from verify_v57_stage1 import independent_gate

OUT = t.ROOT / 'evidence/2026-09-14/v57_delivery'

def main():
    assert not OUT.exists()
    for loader in [a.load, b.load, c.load]:
        loader()
    frozen = {}
    for source in ['evidence/2026-09-14/v56_delivery/delivery.json', 'evidence/2026-09-14/v56_followup_delivery.json']:
        binding = v.read(t.ROOT / source)['bindings']
        for p, h in binding.items():
            assert v.sha(t.ROOT / p) == h, p
        frozen[source] = len(binding)
    models, evaluations = [], []
    for stage, root in [('applicability', a.RUN), ('role_missing', b.RUN), ('port_transfer', c.RUN)]:
        protocols = v.read(root / 'configuration.json')['protocols']
        for protocol in protocols:
            folder = root / protocol
            completion = v.read(folder / 'complete.json')
            for p, h in completion['bindings'].items():
                assert v.sha(folder / p) == h, p
            models.extend(folder.glob('*.joblib'))
            assert completion['evaluation_executed'] == (folder / 'evaluation.parquet').exists()
            if completion['evaluation_executed']:
                baseline = t.RUN / protocol / 'base_evaluation.parquet' if stage == 'applicability' else folder / 'base_evaluation.parquet'
                comparison = independent_gate(pd.read_parquet(baseline), pd.read_parquet(folder / 'evaluation.parquet'))
                evaluations.append({'stage': stage, 'protocol': protocol, **comparison})
    assert len(models) == 22
    assert sum(p.name == 'base.joblib' for p in models) == 6
    assert len(evaluations) == 2 and not any(x['quality_passed'] for x in evaluations)
    evidence_dirs = ['v57_stage1_verification', 'v57_stage2_verification', 'v57_coverage', 'v57_port_verification']
    scores = 0
    for name in evidence_dirs:
        folder = t.ROOT / 'evidence/2026-09-14' / name
        receipt = v.read(folder / 'receipt.json')
        for p, h in receipt['files'].items():
            assert v.sha(folder / p) == h
        if (folder / 'verification.json').exists():
            check = v.read(folder / 'verification.json')
            assert check['all_checks_passed']
            scores += check['score_rows_replayed']
    report = t.ROOT / 'docs/V57_EXECUTION_REVIEW.md'
    for link in re.findall(r'\]\(([^)]+)\)', report.read_text(encoding='utf-8')):
        if not link.startswith('https:'):
            assert (report.parent / link).exists(), link
    OUT.mkdir(parents=True)
    result = {'new_fits': 22, 'new_base_fits': 6, 'new_residual_fits': 16, 'reused_stage1_bases': 4, 'independent_score_replays': scores, 'frozen_prior_bindings_verified': frozen, 'evaluation_results': evaluations, 'port_transfer_selected_zero': all(v.read(c.RUN / p / 'selection.json')['selected'] == 'zero' for p in v.read(c.RUN / 'configuration.json')['protocols']), 'promoted': False, 'new_external_data': False, 'original_pressure_evaluation_executed': False, 'platform_used': False, 'OOF_residual_training_executed': False, 'tests_executed': {'test_v57.py': 3, 'test_v57_stage2.py': 2}, 'scope': 'Completed three bounded experimental stages and independent verification. Numeric correctness, feature applicability and branch isolation validated; no stable M/S quality gain. No new formal classifier or submission produced.'}
    v.save(OUT / 'final_review.json', result)
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    paths = [report] + list((t.ROOT / 'training').glob('*v57*.py'))
    for root in [a.RUN, b.RUN, c.RUN] + [t.ROOT / 'evidence/2026-09-14' / d for d in evidence_dirs] + [OUT]:
        paths.extend(p for p in root.rglob('*') if p.is_file())
    bindings = {p.relative_to(t.ROOT).as_posix(): v.sha(p) for p in sorted(set(paths))}
    v.save(OUT / 'delivery.json', {'count': len(bindings), 'bindings': bindings})
    for p, h in v.read(OUT / 'delivery.json')['bindings'].items():
        assert v.sha(t.ROOT / p) == h
    print(json.dumps({k: z for k, z in result.items() if k != 'evaluation_results'}), flush=True)
    print(json.dumps({'delivery_files_rechecked': len(bindings)}), flush=True)

if __name__ == '__main__':
    main()
