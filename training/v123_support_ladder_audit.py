"""No-fit support audit. Keys are diagnostic projections, never label rules.

Full original ASA population is retained. Reference truth is independently
checked against official row positions; support always excludes query fold.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v123_targeted_plan_20260929'
PREV = ROOT / 'artifacts/v122_evidence_review_20260929'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
COARSE = ('action', 'outcome', 'transport_protocol', 'src_role', 'dst_role')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def key(f, level):
    if level == 'exact':
        v = f
    elif level == 'destination':
        v = {k: v for k, v in f.items() if not k.startswith('src_port')}
    else:
        v = {k: f.get(k) for k in COARSE}
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def known(f):
    if f.get('transport_protocol') == 'icmp':
        return all(k in f and 0 <= f[k] <= 255 for k in ('icmp_type', 'icmp_code'))
    return f.get('transport_protocol') in ('tcp', 'udp') and 0 <= f.get('dst_port_fixed', 65536) <= 65535


def main():
    if OUT.exists():
        raise FileExistsError('Preserve existing audit evidence')
    checks = {}
    for name, path in [('v121', ROOT/'artifacts/v121_paired_batch_training_20260929/delivery.json'),
                       ('v122', PREV/'verification.json')]:
        bindings = read(path)['artifact_sha256']
        for rel, h in bindings.items():
            assert sha(ROOT/rel) == h, rel
        checks[name] = len(bindings)
    d = pd.read_parquet(PREV/'row_diagnosis.parquet')
    trace = pd.read_parquet(TRACE)
    for col in ('row_position', 'local', 'root', 'fold', 'truth', 'facts_json'):
        assert np.array_equal(d[col], trace[col]), col
    assert len(d) == 112807 and not d.row_position.duplicated().any()
    assert d.groupby('root').fold.nunique().max() == 1
    off = pd.read_parquet(ROOT/'data/official/train.parquet', columns=['event_id', 'label_binary'])
    rowmap = pd.read_parquet(ROOT/'artifacts/v75_four_arm_20260921_r2/rows.parquet', columns=['row_position','event_id']).sort_values('row_position')
    assert len(off) == len(rowmap) == 2056871
    assert np.array_equal(rowmap.row_position, np.arange(len(off)))
    assert np.array_equal(off.event_id.astype(str), rowmap.event_id.astype(str))
    yt = off.label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2})
    assert np.array_equal(d.truth, yt.iloc[d.row_position])
    facts = d.facts_json.map(json.loads)
    d['parameter_observed'] = facts.map(known)
    report = []
    for level in ('exact', 'destination', 'behavior'):
        name = f'{level}_key'
        d[name] = [key(f, level) for f in facts]
        for cls, label in [(1,'M'),(2,'S')]:
            for stat in ('rows','roots'):
                d[f'{level}_{label}_{stat}'] = 0
            for fold in range(3):
                q = d.fold.eq(fold)
                t = d[(~q)&d.truth.eq(cls)].groupby(name).agg(rows=('truth','size'), roots=('root','nunique'))
                for stat in ('rows','roots'):
                    d.loc[q,f'{level}_{label}_{stat}'] = d.loc[q,name].map(t[stat]).fillna(0).astype(int)
        for pop, q in [('all_S', d.truth.eq(2)), ('persistent_S', d.truth.eq(2)&d.A_all_seven_wrong)]:
            s = d[q]
            m = s[f'{level}_M_rows'].gt(0); z = s[f'{level}_S_rows'].gt(0)
            report.append(dict(level=level, population=pop, rows=len(s),
                same_only=int((~m&z).sum()), opposite_only=int((m&~z).sum()),
                both=int((m&z).sum()), neither=int((~m&~z).sum()),
                both_classes_two_roots=int((s[f'{level}_M_roots'].ge(2)&s[f'{level}_S_roots'].ge(2)).sum())))
    q = d.truth.eq(2)&d.A_all_seven_wrong
    multi = d.destination_M_roots.ge(2)&d.destination_S_roots.ge(2)
    d['diagnostic_bucket'] = 'outside_persistent_S'
    d.loc[q&multi&d.parameter_observed,'diagnostic_bucket'] = 'known_parameter_two_roots_per_class'
    d.loc[q&multi&~d.parameter_observed,'diagnostic_bucket'] = 'unknown_parameter_pooled_support'
    d.loc[q&~multi&d.destination_S_rows.gt(0),'diagnostic_bucket'] = 'same_class_support_but_insufficient_dual_root_coverage'
    d.loc[q&d.destination_S_rows.eq(0),'diagnostic_bucket'] = 'no_same_class_at_destination_resolution'
    assert int(q.sum()) == 2236
    assert (d.loc[q,'diagnostic_bucket']!='outside_persistent_S').all()
    buckets = d[q].groupby('diagnostic_bucket').agg(rows=('truth','size'),roots=('root','nunique')).reset_index()
    # Unknown destinations do not count as one observed shared destination.
    assert len(d[q&multi&d.parameter_observed]) == 62
    assert len(d[q&multi&~d.parameter_observed]) == 186
    d['protocol'] = facts.map(lambda f:f.get('transport_protocol'))
    d['icmp_type'] = facts.map(lambda f:f.get('icmp_type'))
    d['icmp_code'] = facts.map(lambda f:f.get('icmp_code'))
    ic = d[d.protocol.eq('icmp')].groupby(['fold','icmp_type','icmp_code','truth'],dropna=False).agg(
        rows=('truth','size'),roots=('root','nunique'),A_correct=('A_correct','sum')).reset_index()
    # All cards are retrospective evidence references, not training assignments.
    cards=[]
    for bucket, s in d[q].groupby('diagnostic_bucket',sort=True):
        ids=s.row_position.tolist()
        cards.append({'bucket':bucket,'rows':len(s),'first_row_position':ids[0],
                      'first_raw_message':trace.loc[s.index[0],'raw_message'],
                      'role':'retrospective diagnosis only; actual class/error not available at inference'})
    bound_paths=[Path(__file__), PREV/'row_diagnosis.parquet',PREV/'verification.json', TRACE,
                 ROOT/'data/official/train.parquet',ROOT/'artifacts/v75_four_arm_20260921_r2/rows.parquet']
    summary={'status':'v123_support_ladder_audit_no_training','new_classifier_fits':0,'optimizer_steps':0,
        'latest_actual_training':'V121','model_promoted':False,'quality_acceptance':False,
        'verified_historical_bindings':checks,'population':len(d),'persistent_S':2236,
        'support_by_resolution':report,'persistent_buckets':buckets.to_dict('records'),
        'bucket_roots_are_not_additive':True,
        'scope':'Support counts from opposite folds only; already inspected development population; no causal identification or new blind validation.',
        'limitations':['Destination key deliberately omits source port for support analysis, never deletes source port from model input.',
            'A key matching both labels does not establish the correct M/S boundary.',
            'Two roots per class is a feasibility count, not sufficient evidence of stable transfer; roots are isolation components.',
            'Unknown parameter pools combine different hidden values; 186 rows cannot be described as known-parameter support.',
            'Buckets use retrospective truth/errors for diagnosis only, never for routing or sample weights.'],
        'diagnostic_issues':['Preliminary top-table mental count omitted 12 unknown-UDP rows: known dual-root support corrected from 74 to 62; unknown from 174 to 186. Full-population pre-write assertion caught it; no model or historical file changed.'],
        'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in bound_paths}}
    OUT.mkdir()
    d.to_parquet(OUT/'support_ladder.parquet',index=False)
    ic.to_csv(OUT/'icmp_by_fold.csv',index=False)
    save(OUT/'diagnostic_cards.json',cards)
    save(OUT/'support_summary.json',summary)
    save(OUT/'audit_outputs.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps({k:summary[k] for k in ('verified_historical_bindings','support_by_resolution','persistent_buckets')},ensure_ascii=False))


if __name__ == '__main__':
    main()
