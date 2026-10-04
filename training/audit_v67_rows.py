"""Individual evidence ledger. Diagnostic labels never become inference features."""
import argparse
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from train_v65_rank_heads import load_fit,ROOT,DATA
from run_v67_targeted import ARMS,counts
from run_v67_specialist import hard_mask
from v61_common import FIELDS,save,read,sha


def keys(frame,columns):return frame[columns].astype(str).agg('|'.join,axis=1)


def support(frame, query,columns):
    tr=frame.copy();tr['key']=keys(frame,columns);querykeys=keys(query,columns)
    out={}
    for label in [1,2]:
        table=tr[tr.label.eq(label)].groupby('key').group.nunique()
        out[label]=querykeys.map(table).fillna(0).to_numpy(int)
    return out


def ranking(frame,score):
    result={}
    for proto in ['tcp','udp']:
        mask=hard_mask(frame)&frame.transport_protocol.eq(proto).to_numpy();d=frame[mask]
        w=np.zeros(len(d))
        for label in [1,2]:
            m=d.label.eq(label);cnt=d[m].groupby('group').size()
            w[m]=.5/len(cnt)/d[m].group.map(cnt).to_numpy()
        result[proto]={'source_balanced_ROC_AUC':float(roc_auc_score(d.label.eq(2),score[mask],sample_weight=w)),
            'source_balanced_AP':float(average_precision_score(d.label.eq(2),score[mask],sample_weight=w))}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    f,c=load_fit();run=ROOT/'artifacts/v67_targeted_20260920';special=ROOT/'artifacts/v67_specialist_20260920';exp=ROOT/'artifacts/v67_expansion_20260920'
    manifest=pd.read_parquet(run/'manifest.parquet');target=manifest.target_578.to_numpy();base=manifest.baseline_pred_with_normal_gate.to_numpy()
    receipt=pd.read_parquet(ROOT/'artifacts/v66_issue_resolution_20260920/raw_roundtrip_receipts.parquet')
    assert np.array_equal(receipt.row_position,f.row_position)
    ledger=manifest[target].copy();ledger['event_id']=f.event_id[target];ledger['normalized_text']=f.text[target]
    ledger['raw_sha256']=receipt.raw_sha256[target]
    allf=pd.read_parquet(DATA/'records.parquet');extra=allf[allf.role.ne('fit')]
    # Partition is an operational evidence status, NOT an assertion of unique causal blame.
    ledger['evidence_bucket']=np.where(ledger.training_exact_view_1_sources.gt(0),'same_encoded_input_seen_as_M',
        np.where(ledger.empty_context,'novel_input_without_context','novel_input_with_context'))
    for fold in range(3):
        tr=f[f.fold.ne(fold)];q=f[target&f.fold.eq(fold).to_numpy()]
        for name,columns in [('single_event',FIELDS),('same_destination_port',FIELDS[:5]+['dst_port_fixed']),('coarse',FIELDS[:5])]:
            s=support(tr,q,columns)
            for label in [1,2]:ledger.loc[q.index,name+'_train_'+str(label)+'_sources']=s[label]
    s=support(extra,f[target],FIELDS)
    for label in [1,2]:ledger['auxiliary_single_event_'+str(label)+'_sources']=s[label]
    scores={};effects=[];all_predictions={}
    for folder,arms,score_col,prefix in [(run,ARMS,'p_S_given_threat','control'),
            (special,['coarse_plain','numeric_plain','coarse_balanced','numeric_balanced'],'hard_score','specialist'),
            (exp,['coarse','numeric'],'hard_score','expanded')]:
        for arm in arms:
            d=pd.read_parquet(folder/(arm+'_oof.parquet'));assert np.array_equal(d.row_position,f.row_position)
            name=prefix+'_'+arm;p=d.pred.to_numpy();sc=d[score_col].to_numpy();all_predictions[name]=p
            ledger[name+'_score']=sc[target];ledger[name+'_pred']=p[target]
            scores[name]=ranking(f,sc)
            eff=counts(f,p,base,target);effects.append({'candidate':name,'target_fixed':eff['target_fixed'],
                'S_fixed':eff['S']['fixed'],'S_broken':eff['S']['broken'],'M_fixed':eff['M']['fixed'],'M_broken':eff['M']['broken'],
                'S_source_recall':eff['S']['source_recall'],'M_source_recall':eff['M']['source_recall'],'total_errors':eff['total_errors']})
    # No per-row winner can be deployed: union is explicitly diagnostic only.
    any_fixed=np.any(np.array([p==2 for p in all_predictions.values()]),axis=0)
    ledger['any_candidate_fixed_diagnostic_only']=any_fixed[target]
    ledger['resolution_status']=np.where(ledger.any_candidate_fixed_diagnostic_only,
        'isolated_repair_with_rejected_candidate_not_accepted','unresolved_after_all_tested_candidates')
    ledger.to_parquet(a.out/'target_578_ledger.parquet',index=False)
    ledger.to_csv(a.out/'target_578_ledger.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(effects).to_csv(a.out/'candidate_effects.csv',index=False)
    group=ledger.groupby(['fold','group']).agg(rows=('row_position','size'),
        empty_context=('empty_context','sum'),src_redacted=('src_redacted','sum'),dst_redacted=('dst_redacted','sum'),
        any_candidate_repairs=('any_candidate_fixed_diagnostic_only','sum')).reset_index()
    group.to_csv(a.out/'target_162_sources.csv',index=False)
    # Every rejected candidate includes collateral rows, not just the selected failures.
    collateral=f[['row_position','event_id','label','group','fold','text']].copy();collateral['base']=base
    changed=np.zeros(len(f),bool)
    for name,pred in all_predictions.items():collateral[name]=pred;changed|=pred!=base
    collateral[changed].to_parquet(a.out/'all_changed_rows.parquet',index=False)
    summary={'target_rows':len(ledger),'target_sources':int(ledger.group.nunique()),
        'evidence_buckets':ledger.evidence_bucket.value_counts().to_dict(),
        'flags':{k:int(ledger[k].sum()) for k in ['empty_context','source_port_OOV','destination_port_OOV','src_redacted','dst_redacted','cap_removed_entries']},
        'auxiliary_same_single_input_S_support_rows':int(ledger.auxiliary_single_event_2_sources.gt(0).sum()),
        'auxiliary_same_single_input_S_only_support_rows':int((ledger.auxiliary_single_event_2_sources.gt(0)&ledger.auxiliary_single_event_1_sources.eq(0)).sum()),
        'any_candidate_repair_union_NOT_a_model':int(ledger.any_candidate_fixed_diagnostic_only.sum()),
        'ranking_from_saved_soft_scores':scores,'effects':effects,'quality_acceptance':False,
        'source_sha256':sha(__file__),
        'limitations':['Non-conflict originally meant within validation fold, not globally label-consistent.',
          'Exact support absent does not prove generalization impossible; finite conflict counts are representation-specific.',
          'One-hot-derived AP/CE fields in training analysis are decision summaries, not soft-score discrimination. Use ranking here.',
          'Target diagnosis and all experiments are adaptive development; no independent future data assessed.',
          'No row-by-row model switching, label rewriting, or removal from evaluation was performed.']}
    save(a.out/'audit.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
