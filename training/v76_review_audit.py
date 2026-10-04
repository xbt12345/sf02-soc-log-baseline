"""Read-only V75 reanalysis for V76 planning; no new fitting or raw relabeling."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'artifacts/v75_four_arm_20260921_r2'
DEST=ROOT/'evidence/2026-09-22/v76_review'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    delivery=read(ROOT/'evidence/2026-09-21/v75_four_arm/delivery.json')
    for path,h in delivery['artifact_sha256'].items():assert sha(ROOT/path)==h,path
    table=pd.read_csv(P/'analysis/all_formats_all_classes.csv');d=table[table.arm.eq('D')]
    full=read(P/'official_replay/scoring.json');f=read(P/'F_diagnostic/scoring.json')
    fullroutes={s['route']:s for s in full['slices']};rows=[]
    m_total=full['after']['support'][1];inner_m=int(d[d.label.eq('malicious')].validation_rows.sum())
    for route in sorted(set(d.route)|set(fullroutes)):
        z=d[(d.route==route)&(d.label=='malicious')].iloc[0] if route in set(d.route) else None
        s=fullroutes.get(route);n=s['after']['support'][1] if s else 0
        ir=None if z is None or z.validation_rows==0 else 1-z.validation_errors/z.validation_rows
        fr=None if not n else s['after']['recall'][1]
        rows.append({'route':route,'train_M':int(z.official_rows) if z is not None else 0,
            'internal_M':int(z.validation_rows) if z is not None else 0,'target_M':n,
            'internal_M_share':int(z.validation_rows)/inner_m if z is not None else 0,
            'target_M_share':n/m_total,'internal_D_M_recall':ir,'full_D_M_recall':fr})
    pd.DataFrame(rows).to_csv(DEST/'malicious_support_shift.csv',index=False)
    # Standardization is DESCRIPTIVE only: target class proportions are from
    # previously inspected development answers, never selection/training weights.
    known=sum(x['target_M_share']*x['internal_D_M_recall'] for x in rows if x['internal_D_M_recall'] is not None)
    missing=sum(x['target_M_share'] for x in rows if x['internal_D_M_recall'] is None)
    r=pd.read_parquet(P/'rows.parquet',columns=['row_position','route','label_index','component','body_group','source_symbol','is_validation'])
    support=[]
    for (route,label),z in r.groupby(['route','label_index']):
        fit=z[~z.is_validation];hold=z[z.is_validation]
        support.append({'route':route,'label_index':int(label),'rows':len(z),'components':int(z.component.nunique()),
            'body_groups':int(z.body_group.nunique()),'observed_source_symbols':int(z.loc[z.source_symbol>=0,'source_symbol'].nunique()),
            'fit_rows':len(fit),'fit_components':int(fit.component.nunique()),'validation_rows':len(hold),
            'validation_components':int(hold.component.nunique())})
    pd.DataFrame(support).to_csv(DEST/'independent_support.csv',index=False)
    c=read(P/'four_arm/complete.json');cc=read(P/'corrective/complete.json')
    curves={a:[e['validation'][a]['errors'] for e in rec['curves']] for a,rec in [('D',c),('E',cc),('F',cc)]}
    result={'new_fits':0,'historical_receipt_hashes_all_match':True,
        'malicious_support_shift':rows,'unsupported_plus_vpc_target_M':(6226+2664)/m_total,
        'native_flow_fraction_of_A_to_D_error_reduction':2320/2406,
        'standardized_internal_M_recall_interval':[known,known+missing],
        'unmeasured_target_M_mass':missing,
        'standardization_warning':'Descriptive mix analysis only. Fold D and full D have different fitted parameters; not a causal effect or valid target score estimate. Missing validation cells are an interval, never zero-filled.',
        'last_epoch_error_trajectories':curves,
        'convergence':'Six partial_fit epochs and heldout metrics do not certify objective convergence; no gradient certificate recorded.',
        'F_full_fixed':sum(s['fixed'] for s in f['slices']),'F_full_broken':sum(s['broken'] for s in f['slices']),
        'official_model_promoted':False,'source_sha256':sha(Path(__file__)),
        'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [P/'configuration.json',P/'analysis/all_formats_all_classes.csv',P/'official_replay/scoring.json',P/'F_diagnostic/scoring.json',P/'corrective/complete.json']}}
    (DEST/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs_sha256','malicious_support_shift']},ensure_ascii=False,indent=2))
    print(pd.DataFrame(rows)[lambda t:(t.target_M>0)|(t.internal_M>0)].to_string(index=False))

if __name__=='__main__':main()
