"""Independent post-run K/U selection and arithmetic verification, no new fit."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from v116_preflight import ROOT, DEST, VIEW, sha, save


def read(p): return json.loads(p.read_text(encoding='utf-8'))


def main():
    reg=read(DEST/'training_registration.json')
    for rel,h in reg['input_sha256'].items(): assert sha(ROOT/rel)==h,rel
    m=pd.read_parquet(DEST/'inner_split_manifest.parquet')
    locked=read(DEST/'locked_selection.json'); result=read(DEST/'evaluation.json')
    rows=[];chosen={};checks={}
    for fold in range(3):
        d=m[m.outer_fold.eq(fold)].reset_index(drop=True)
        folder=DEST/f'inner_fold{fold}'
        ids=np.load(folder/'inference_local_ids.npy')
        assert np.array_equal(ids,np.unique(d.loc[d.fold.ne(fold),'local']))
        lookup={int(v):i for i,v in enumerate(ids)}
        reference=read(folder/'checkpoints.json'); candidates=[]
        for rec in reference:
            epoch=rec['epoch'];path=folder/f'epoch{epoch}_prob.npy'
            assert sha(path)==rec['prob_sha256']
            assert sha(folder/f'epoch{epoch}_model.pt')==rec['model_sha256']
            prob=np.load(path)
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
            entry={'epoch':epoch,'tasks':{}}
            for task in ('K','U'):
                q=d[d.role.eq(task)];y=q.truth.to_numpy()
                pred=prob[[lookup[int(v)] for v in q.local]].argmax(1)
                cm=confusion_matrix(y,pred,labels=[0,1,2])
                assert cm.tolist()==rec['tasks'][task]['confusion_matrix_true_rows_pred_columns']
                score=float(f1_score(y,pred,labels=[1,2],average='macro',zero_division=0))
                assert abs(score-rec['tasks'][task]['MS_equal_F1'])<1e-12
                mean=np.mean([pd.DataFrame({'root':q.loc[q.truth.eq(c),'root'],
                    'correct':pred[y==c]==c}).groupby('root').correct.mean().mean() for c in (1,2)])
                entry['tasks'][task]={'correct':cm.diagonal()[1:].tolist(),'f1':score,'root_macro':float(mean)}
                for c,name in [(1,'M'),(2,'S')]:
                    rows.append({'fold':fold,'epoch':epoch,'task':task,'class':name,'support':int((y==c).sum()),
                        'correct':int(cm[c,c]),'errors':int((y==c).sum()-cm[c,c]),'recall':float(cm[c,c]/(y==c).sum()),
                        'task_MS_F1':score,'task_root_macro_recall':float(mean)})
            candidates.append(entry)
        end=next(q for q in candidates if q['epoch']==25)
        admissible=[q for q in candidates if all(np.all(np.array(q['tasks'][t]['correct']) >=
                   np.array(end['tasks'][t]['correct'])) for t in ('K','U'))]
        ranked=sorted(admissible,key=lambda q:(-min(q['tasks'][t]['f1'] for t in ('K','U')),
                    -sum(q['tasks'][t]['root_macro'] for t in ('K','U'))/2,q['epoch']))
        chosen[str(fold)]=ranked[0]['epoch']
        assert chosen[str(fold)]==locked['folds'][str(fold)]['selected_epoch']
        # No fitting or selection exposure of outer records, no K/U leakage.
        fit=d[d.role.eq('fit')];val=d[d.role.isin(['K','U','validation_collateral'])]
        assert set(fit.root).isdisjoint(val.root) and set(fit.local).isdisjoint(val.local)
        assert set(d.loc[d.role.eq('K'),'parameter']) <= set(fit.parameter.dropna())
        assert set(d.loc[d.role.eq('U'),'parameter']).isdisjoint(set(fit.parameter.dropna()))
    pd.DataFrame(rows).to_csv(DEST/'checkpoint_class_metrics.csv',index=False)
    dd=pd.read_parquet(DEST/'OOF_ASA_decisions.parquet')
    for arm,col in [('A','A_epoch25'),('B','B_selected')]:
        y=dd.truth.to_numpy();pred=dd[col].to_numpy()
        cm=confusion_matrix(y,pred,labels=[0,1,2])
        assert cm.tolist()==result['ASA'][arm]['confusion_matrix_true_rows_pred_columns']
    for c,name in [(1,'M'),(2,'S')]:
        mask=dd.truth.eq(c)
        assert int((mask & dd.A_epoch25.eq(c) & dd.B_selected.ne(c)).sum())==result['ASA']['class_comparison'][name]['regressed']
        assert int((mask & dd.A_epoch25.ne(c) & dd.B_selected.eq(c)).sum())==result['ASA']['class_comparison'][name]['repaired']
    checks.update(all_inner_checkpoint_hashes_and_metrics=True,independent_sklearn_MS_F1=True,
                  independent_epoch_selection=True,root_input_parameter_isolation=True,
                  independent_outer_confusion_and_negative_flips=True,
                  exactly_six_successful_trajectories=len(list(DEST.glob('*_fold*/fit.json')))==6)
    assert all(checks.values())
    out={'all_checks_passed':True,'checks':checks,'selected_epochs':chosen,
         'classifier_fits_added_by_verifier':0,'source_sha256':sha(__file__),
         'evaluation_sha256':sha(DEST/'evaluation.json'),'curve_sha256':sha(DEST/'checkpoint_class_metrics.csv'),
         'scope':'Independent arithmetic and protocol verification; does not certify transfer quality.'}
    save(DEST/'independent_verification.json',out)
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
