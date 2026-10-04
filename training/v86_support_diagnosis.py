"""Train-only behavioural support for v85 regressions, without fitting."""
import json
import numpy as np
import pandas as pd
from run_v75 import ROOT,OUT,read,save,sha
from v79_execute import rows
from v82_capacity import LAST
from v86_boundary_diagnosis import DEST,OLD


def main():
    if (DEST/'review_receipt.json').exists():raise FileExistsError('Frozen review')
    r=rows();fid=np.load(LAST/'row_feature_id.npy');sel=np.load(OLD/'fold1/selected_rows.npy')
    p=pd.read_parquet(OUT/'projections.parquet',columns=['facts']);facts=[json.loads(v) for v in p.facts]
    cases=read(DEST/'regression_neighbors.json');result=[]
    fields=['action','outcome','transport_protocol','src_role','dst_role']
    for case in cases:
        target=case['target_facts'];out={'fid':case['fid'],'target_row':case['target_row'],'levels':[]}
        for name,keys in [('behavior',fields),('behavior_and_destination_port',fields+['dst_port_fixed']),('all_parsed_facts',sorted(target))]:
            # Compare existence as well as values; full-facts level requires exact dict equality.
            eligible=np.array([f==target if name=='all_parsed_facts' else all((k in f)==(k in target) and f.get(k)==target.get(k) for k in keys) for f in facts])
            take=sel[eligible[r.projection_id.to_numpy()[sel]]];z=r.iloc[take]
            out['levels'].append({'name':name,'rows':len(z),'by_class_components':[
                {'class':c,'rows':int((z.label_index==c).sum()),'components':int(z.loc[z.label_index==c,'component'].nunique()),
                 'unique_encoded_inputs':int(np.unique(fid[take[z.label_index.to_numpy()==c]]).size)} for c in range(3)]})
        result.append(out)
    prog=read(OLD/'fold1/C_progress.json')
    trace=read(OLD/'fold1/C_selection_trace.json')
    # Pair eligibility only; no contrastive fitting. Distinct-input filtering must
    # still be enforced when constructing actual pairs, so this is an upper bound.
    old=np.load(OLD/'fold1/teacher_prediction.npy');asa=sel[(r.route.to_numpy()[sel]=='asa')&(r.label_index.to_numpy()[sel]>0)]
    coverage=[]
    for level,keys in [('coarse_behavior',fields),('behavior_and_destination_port',fields+['dst_port_fixed'])]:
        keyvals=[json.dumps({k:f[k] for k in keys if k in f},sort_keys=True) for f in facts]
        fr=r.iloc[asa][['label_index','component']].copy();fr['key']=[keyvals[int(pid)] for pid in r.projection_id.iloc[asa]]
        fr['old_wrong']=old[fid[asa]]!=fr.label_index.to_numpy()
        comp=fr.groupby(['key','label_index']).component.nunique();labels=fr.groupby('key').label_index.nunique()
        fr['positive_available']=[comp.loc[(k,c)]>=2 for k,c in zip(fr.key,fr.label_index)]
        fr['negative_available']=fr.key.map(labels).ge(2);fr['both']=fr.positive_available&fr.negative_available
        coverage.append({'level':level,'scope':'Train-only candidate upper bound; pair construction must further exclude identical encoded mixed-label inputs and apply semantic compatibility checks.',
             'by_class':[{'class':int(c),'old_wrong':bool(w),'rows':len(g),'positive_crosscomponent_available':int(g.positive_available.sum()),
                          'both_candidate_conditions':int(g.both.sum())} for (c,w),g in fr.groupby(['label_index','old_wrong'])]})
    out={'source_sha256':sha(__file__),'new_classifier_fits':0,'train_only_support':result,
        'train_only_contrastive_candidate_coverage':coverage,
        'C_aux_nonzero_epochs':sum(q['aux_before_step']!=0 for q in prog),
        'C_first_rejected_epoch':next(q['epoch'] for q in prog if q['fully_rejected']),
        'C_rejected_epochs':[q['epoch'] for q in prog if q['fully_rejected']],
        'C_replay_scope':'Observed all 60 hinge penalties zero; all final 28 steps rejected with restored state. GPU arithmetic may vary at float32 roundoff, no accepted update after32.',
        'C_saved_minimum_P_margin':[{'epoch':q['epoch'],'value':q['protection']['min_protected_margin']} for q in trace],
        'scope':'Support stratification is diagnostic only. Coarse behavior keys can merge different intent/policy; mixed coarse labels are not proof of raw-label error or irreducibility.'}
    save(DEST/'support_and_solver.json',out);print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
