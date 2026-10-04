"""Read-only v6.1 checkpoint diagnosis and frozen feature extraction for head tests."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from v61_common import read, save, sha
from v61_runtime import load_data, score, predictions
from train_v61_neural import Classifier, Inputs, amp, predict

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/v61_models_20260914'
DATA=ROOT/'artifacts/v61_source_factorial_20260914_r2'
ASSET=ROOT/'artifacts/v61_asset_cache/securebert2'
OUT=ROOT/'artifacts/v62_capacity_20260915'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'cache.npz').exists()
    torch.set_num_threads(4);device=torch.device('cuda');assert torch.cuda.is_available()
    df,ctx,_=load_data(DATA);pp=read(OLD/'D_20260915/preprocessing.json')
    inputs=Inputs(df,ctx,ASSET,device,pp['vocab'],pp['scaling'])
    fit=np.flatnonzero(df.role.eq('fit'));dev=np.flatnonzero(df.role.eq('selection'))
    diagnosis={}
    for arm in 'BD':
        p=OLD/f'{arm}_20260915/latest_training_state.pt';digest=sha(p)
        state=torch.load(p,map_location='cpu',weights_only=True);assert state['epoch']==3
        model=Classifier(ASSET,inputs.vocab,arm,pretrained=False)
        model.load_state_dict(state['model']);model.to(device);del state
        result={}
        for role,indices in [('fit',fit),('selection',dev)]:
            prob=predict(model,inputs,indices)
            result[role]=score(df.label.iloc[indices],prob)
            predictions(OUT/f'{arm}_last_{role}.parquet',df,indices,prob)
        diagnosis[arm]={'checkpoint_epoch':3,'checkpoint_sha256':digest,'metrics':result,
            'selected_epoch_one_fit':read(OLD/f'{arm}_20260915/selection.json')['fit']}
        del model;torch.cuda.empty_cache()
        print(arm+' third-epoch checkpoint diagnosis complete; no outer predictions.',flush=True)
    save(OUT/'checkpoint_diagnosis.json',diagnosis)
    p=OLD/'D_20260915/model.pt'
    assert sha(p)==read(OLD/'D_20260915/selection.json')['model_sha256']
    model=Classifier(ASSET,inputs.vocab,'D',pretrained=False)
    model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));model.to(device).eval()
    active=np.flatnonzero(df.role.ne('evaluation'));codes=inputs.text_codes[active].cpu().numpy()
    unique,inverse=np.unique(codes,return_inverse=True);text=[];facts=[]
    with torch.no_grad():
        for start in range(0,len(unique),256):
            indices=torch.tensor(unique[start:start+256],device=device)
            tok={k:v[indices] for k,v in inputs.tokens.items()}
            with amp(device):
                hidden=model.encoder(**tok).last_hidden_state
                mask=tok['attention_mask'].unsqueeze(-1)
                vec=(hidden*mask).sum(1)/mask.sum(1)
            text.append(vec.float().cpu().numpy())
        for start in range(0,len(active),2048):
            idx=active[start:start+2048]
            with amp(device):vec=model.fact_encode(inputs.facts[idx])
            facts.append(vec.float().cpu().numpy())
    text=np.concatenate(text)[inverse];facts=np.concatenate(facts)
    remap=np.full(len(df),-1,dtype=np.int32);remap[active]=np.arange(len(active))
    old_neighbors=ctx['neighbors'][active];neighbors=remap[np.maximum(old_neighbors,0)]
    neighbors[old_neighbors<0]=-1
    assert ((neighbors>=0)|(old_neighbors<0)).all()
    sub=df.iloc[active][['row_position','group','role','label']].reset_index(drop=True)
    assert sub.role.ne('evaluation').all()
    np.savez_compressed(OUT/'cache.npz',text=text,facts=facts,stats=inputs.stats[active].cpu().numpy(),
                        neighbors=neighbors,relation=ctx['relation'][active])
    sub.to_parquet(OUT/'rows.parquet',index=False)
    save(OUT/'cache_receipt.json',{'encoder_and_fact_backbone':'Frozen v61 D selected epoch 1, fitted only on original fit role',
        'model_sha256':sha(p),'rows':len(sub),'no_outer_features_or_labels_in_cache':True,
        'learned_backbone_does_not_see_selection_labels_in_gradient':True,
        'selection_was_used_previously_for_epoch_choice':True,
        'scope':'Adaptive development capability diagnostic, not a new blind test',
        'source_sha256':sha(__file__),'bindings':{n:sha(OUT/n) for n in ['cache.npz','rows.parquet']}})
    print('Frozen common backbone cache prepared; old outer role excluded.',flush=True)


if __name__=='__main__':main()
