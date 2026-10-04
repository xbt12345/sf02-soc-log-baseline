"""Phase B: fixed historical B2 probabilities, never a new quality experiment."""
import argparse,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soc_v3_prepare as base
import v37_learning as learning
from v37_contract import CONTRACT,TASKS,BUDGETS
from run_v36_train import save
from run_v37_train import assessment

def run(args):
    root=Path(args.previous_results);prepared=Path(args.previous_prepared)
    product=np.asarray(pq.read_table(prepared/'prepared.parquet',columns=['product'])['product'].to_pylist(),object)
    groups=pq.read_table(root.parent/'prepared_attempt001/groups.parquet')['union_group'].to_numpy()
    results=[]
    for task in TASKS:
        folder=root/(task+'_B2');identity=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        for name in ['model.joblib','calibration.parquet','evaluation.parquet']:assert base.file_hash(folder/name)==identity['artifacts'][name]
        tables=[]
        for role in ['calibration','evaluation']:
            t=pq.read_table(folder/(role+'.parquet'));pos=t['row_position'].to_numpy();y=t['label_index'].to_numpy()
            p=np.column_stack([t[n].to_numpy() for n in ['p_benign','p_malicious','p_suspicious']]);tables.append((pos,y,p))
        pos,y,p=tables[0]
        policies=[learning.calibration_policies(y,p,product[pos],groups[pos],a) for a in BUDGETS]
        pos,y,p=tables[1]
        results.append({'task':task,'assessment':assessment(y,p,groups[pos],product[pos],policies),'policies':policies,
            'source_artifacts':{name:identity['artifacts'][name] for name in ['model.joblib','calibration.parquet','evaluation.parquet']}})
    save(Path(args.output),{'version':'v37-fixed-score-audit-1.0','contract':CONTRACT,'tasks':results,
        'model_trained':False,'historical_examined_targets':True,'new_prepared_roles_comparable':False,
        'scope':'Decision implementation on frozen historical B2 scores. Paired input improvement requires newly matched C fits.'})
    print(json.dumps({'phase_B_completed':True,'tasks':len(results),'model_trained':False}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous-results',required=True);p.add_argument('--previous-prepared',required=True);p.add_argument('--output',required=True);run(p.parse_args())
