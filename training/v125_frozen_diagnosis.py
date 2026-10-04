"""V125: no-fit input-chain, fit-support and crossed frozen-inference audit.

All labels are inspected development evidence. No optimizer or selector is run.
Historical artifacts are verified and never rewritten.
"""
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import sparse
from sklearn.preprocessing import normalize

from v75_views import BYTE_FEATURES, byte_matrix, matrix_hashes
from v75_metadata import encode
from v124_header import old_text
from v104_phase_b import SparseTabM, predict_all

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v125_review_20260929'
PREV = ROOT/'artifacts/v124_header_trial_20260929'
TRACE = ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
LADDER = ROOT/'artifacts/v123_targeted_plan_20260929/support_ladder.parquet'
OFFICIAL = ROOT/'data/official/train.parquet'
VIEW = ROOT/'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'
ENCODER = ROOT/'artifacts/v75_four_arm_20260921_r2/facts_encoder.joblib'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def errors(y,p):
    return {str(c):{'rows':int((y==c).sum()),'errors':int(((y==c)&(p!=c)).sum())} for c in (1,2)}


def main():
    if OUT.exists():raise FileExistsError(OUT)
    bound=read(PREV/'delivery.json')['artifact_sha256']
    for rel,h in bound.items():assert sha(ROOT/rel)==h,rel
    d=pd.read_parquet(TRACE)
    ladder=pd.read_parquet(LADDER)
    assert d.row_position.equals(ladder.row_position) and len(d)==112807
    pos=d.row_position.to_numpy(); idx=d.local.to_numpy(); y=d.truth.to_numpy()
    source=pq.read_table(OFFICIAL,columns=['event_id','message_sanitized','label_binary','src_port']).take(pos).to_pandas()
    assert source.message_sanitized.equals(d.raw_message)
    assert np.array_equal(source.label_binary.map({'benign':0,'malicious':1,'suspicious':2}),y)
    identity=pd.read_parquet(ROOT/'artifacts/v75_four_arm_20260921_r2/rows.parquet',columns=['row_position','event_id'])
    assert np.array_equal(identity.row_position,np.arange(len(identity)))
    assert np.array_equal(source.event_id,identity.iloc[pos].event_id)
    del identity
    x={'A':sparse.load_npz(VIEW),'B':sparse.load_npz(PREV/'B_header_ASA.npz')}
    facts=[json.loads(s) for s in d.facts_json]
    enc=joblib.load(ENCODER); names=list(enc.names())
    fx=normalize(enc.transform(facts).astype('f4'),copy=False)
    diff=fx-x['A'][idx,BYTE_FEATURES:-18]
    assert diff.nnz==0 or np.max(np.abs(diff.data))<1e-7
    meta,_,meta_report=encode(source.src_port,[f.get('src_port_fixed',65536) for f in facts])
    assert (meta!=x['A'][idx,-18:]).nnz==0
    unique=d.drop_duplicates('local').sort_values('local')
    assert np.array_equal(unique.local,np.arange(x['A'].shape[0]))
    text=byte_matrix([old_text(s) for s in unique.raw_message])
    diff=text-x['A'][:,:BYTE_FEATURES]
    assert diff.nnz==0 or np.max(np.abs(diff.data))<1e-7
    # Independent readback of bit coordinates and raw numeric endpoint fields.
    fields=['src_port_fixed','dst_port_fixed','icmp_type','icmp_code']
    decoded={}
    for field in fields:
        columns=[i for i,n in enumerate(names) if n.startswith(field+':bit')]
        powers=np.array([1<<int(names[i].split(':bit')[1]) for i in columns])
        decoded[field]=np.asarray((x['A'][idx,BYTE_FEATURES:-18][:,columns]>0)@powers).ravel()
        present=np.array([field in f for f in facts])
        expected=np.array([f.get(field,0) for f in facts])
        assert np.array_equal(decoded[field][present],expected[present]),field
    ports=re.compile(r'\b(?P<role>src|dst)\s+[^\s:]+:[^\s/]+/(?P<token>[^\s]+)')
    icmp=re.compile(r'\(type\s+(\d+),\s*code\s+(\d+)\)')
    chain=[]
    for i,(raw,f) in enumerate(zip(d.raw_message,facts)):
        numeric=unknown=0
        if f['transport_protocol'] in ('tcp','udp'):
            tokens={m['role']:m['token'] for m in ports.finditer(raw)}
            assert set(tokens)=={'src','dst'},int(pos[i])
            for role,tok in tokens.items():
                key=role+'_port_fixed'
                if tok.isdecimal() and 0<=int(tok)<=65535:
                    assert int(tok)==f[key]==decoded[key][i],int(pos[i]); numeric+=1
                else:
                    assert f[key]==65536,int(pos[i]); unknown+=1
        else:
            assert f['transport_protocol']=='icmp'
            m=icmp.search(raw);assert m,int(pos[i])
            assert [int(m[1]),int(m[2])]==[f['icmp_type'],f['icmp_code']],int(pos[i])
            numeric+=2
        chain.append({'row_position':int(pos[i]),'numeric_parameters_checked':numeric,
                      'unavailable_ports_not_reconstructed':unknown,'parameter_chain_matches':True})
    keys={arm:np.array([h.hex() for h in matrix_hashes(xx)],dtype=object)[idx] for arm,xx in x.items()}
    OUT.mkdir()
    pd.DataFrame(chain).to_parquet(OUT/'parameter_chain.parquet',index=False)
    support=[]; checkpoints=[]; conflict_report=[]
    cards=read(PREV/'capability_qualification.json')['cards']
    for fold in range(3):
        for arm in ('A','B'):
            fit=d.fold.ne(fold).to_numpy()
            # True matrix hashes, not the historical local identifier.
            tmp=pd.DataFrame({'key':keys[arm][fit],'truth':y[fit]})
            ct=pd.crosstab(tmp.key,tmp.truth);mixed=set(ct.index[(ct>0).sum(1)>1])
            p=np.load(PREV/f'fold{fold}_{arm}/epoch25_prob.npy')[idx]
            conflict_report.append({'fold':fold,'arm':arm,'fit_rows':int(fit.sum()),
                'empirical_matrix_conflict_floor':int((ct.sum(1)-ct.max(1)).sum()),
                'M_errors':int((fit&(y==1)&(p.argmax(1)!=y)).sum()),
                'S_errors':int((fit&(y==2)&(p.argmax(1)!=y)).sum()),
                'S_error_rows_in_mixed_inputs':int((fit&(y==2)&(p.argmax(1)!=y)&np.isin(keys[arm],list(mixed))).sum())})
            for card in cards:
                if card['fold']!=fold:continue
                take=np.flatnonzero(fit & ladder.destination_key.eq(card['destination_key']).to_numpy())
                for i in take:
                    support.append({'outer_fold':fold,'arm':arm,'row_position':int(pos[i]),'truth':int(y[i]),
                        'local':int(idx[i]),'root':int(d.root.iloc[i]),'behavior':card['destination_key'],
                        'prediction':int(p[i].argmax()),'p_S_minus_p_M':float(p[i,2]-p[i,1]),
                        'same_input_has_both_labels':keys[arm][i] in mixed})
            priority=(ladder.diagnostic_bucket.eq('known_parameter_two_roots_per_class').to_numpy()&~fit)
            for epoch in (1,2,5,10,15,20,25):
                pp=np.load(PREV/f'fold{fold}_{arm}/epoch{epoch}_prob.npy')[idx]
                ss=np.zeros(len(d),bool)
                for card in cards:
                    if card['fold']==fold:ss|=fit&ladder.destination_key.eq(card['destination_key']).to_numpy()&(y==2)
                checkpoints.append({'fold':fold,'arm':arm,'epoch':epoch,
                    'priority_fit_S_rows':int(ss.sum()),'priority_fit_S_errors':int((ss&(pp.argmax(1)!=y)).sum()),
                    'priority_held_S_rows':int(priority.sum()),'priority_held_S_errors':int((priority&(pp.argmax(1)!=y)).sum())})
    support=pd.DataFrame(support)
    assert not support.duplicated(['outer_fold','arm','row_position']).any()
    support.to_parquet(OUT/'priority_fit_roles.parquet',index=False)
    save(OUT/'checkpoint_diagnosis.json',checkpoints)
    # Frozen factorial: each endpoint model on both encodings; only changed inputs need extra compute.
    gaps=pd.read_parquet(PREV/'header_span_ledger.parquet')
    gap=d.row_position.isin(gaps.row_position).to_numpy()
    crossed=[]; replays=[]
    with torch.no_grad():
        for fold in range(3):
            take=np.flatnonzero(gap&d.fold.eq(fold).to_numpy())
            used=np.unique(idx[take]); remap={int(k):i for i,k in enumerate(used)}
            for model_arm in ('A','B'):
                model=SparseTabM().to('cuda');model.eval()
                model.load_state_dict(torch.load(PREV/f'fold{fold}_{model_arm}/epoch25_model.pt',map_location='cpu',weights_only=True)['state_dict'])
                for input_arm in ('A','B'):
                    pp=predict_all(model,'TabM',x[input_arm][used],'cuda')
                    if model_arm==input_arm:
                        ref=np.load(PREV/f'fold{fold}_{model_arm}/epoch25_prob.npy')[used]
                        err=float(np.max(np.abs(pp-ref)))
                        assert np.allclose(pp,ref,rtol=2e-6,atol=2e-6) and np.array_equal(pp.argmax(1),ref.argmax(1))
                        replays.append({'fold':fold,'arm':model_arm,'max_probability_difference':err})
                    for i in take:
                        v=pp[remap[int(idx[i])]]
                        crossed.append({'row_position':int(pos[i]),'fold':fold,'truth':int(y[i]),
                            'model':model_arm,'input':input_arm,'prediction':int(v.argmax()),
                            'p_S_minus_p_M':float(v[2]-v[1])})
                del model
    cf=pd.DataFrame(crossed);assert len(cf)==682*4
    cf.to_parquet(OUT/'crossed_frozen_predictions.parquet',index=False)
    summary={}
    for key,z in cf.groupby(['model','input']):summary['model'+key[0]+'_input'+key[1]]=errors(z.truth.to_numpy(),z.prediction.to_numpy())
    support_summary={}
    for arm,z in support.groupby('arm'):
        s=z[z.truth.eq(2)];e=s[s.prediction.ne(2)]
        support_summary[arm]={'fit_role_rows':len(z),'unique_original_rows':int(z.row_position.nunique()),
            'S_role_rows':len(s),'S_unique_original_rows':int(s.row_position.nunique()),
            'S_error_role_rows':len(e),'S_error_unique_original_rows':int(e.row_position.nunique()),
            'S_error_role_rows_in_mixed_input':int(e.same_input_has_both_labels.sum()),
            'S_error_roots':int(e.root.nunique()),
            'S_correct_role_rows':int(s.prediction.eq(2).sum())}
    report={'version':'V125','status':'frozen_diagnosis_complete_no_training','classifier_fits_new':0,
        'optimizer_steps':0,'calibration_fits':0,'model_promoted':False,'latest_actual_training':'V124',
        'historical_bound_files_verified':len(bound),'full_official_rows':2056871,'ASA_rows':len(d),
        'input_chain':{'official_raw_and_label_identity':True,'old_text_matrix_unique_inputs_rebuilt':len(unique),
            'parsed_fact_matrix_rows_rebuilt':len(d),'record_metadata_matrix_rows_rebuilt':len(d),
            'raw_numeric_parameter_comparisons':sum(z['numeric_parameters_checked'] for z in chain),
            'unavailable_port_occurrences':sum(z['unavailable_ports_not_reconstructed'] for z in chain),
            'decoded_fields':fields,'numeric_mismatches':0,
            'limits':'Numeric parameters and cached fact encoding verified; no proof that arbitrary raw semantics, long-range binding or hidden context is sufficient.'},
        'crossed_header_682':summary,'checkpoint_replay':replays,'priority_support':support_summary,
        'all_fit_conflicts':conflict_report,
        'prior_qualification_gap':'V124 qualification counted full support but rendered only three roots/class and marked mechanism unresolved; it did not exhaustively qualify a raw-to-matrix mechanism.',
        'limits':['Repeatedly inspected development folds, no blind generalization evidence.',
            'Fit roles overlap across folds; unique original row counts are reported separately.',
            'Crossed inference measures fixed-model input sensitivity and trained-model difference, not independent causal estimates of optimizer randomness.',
            'No new classification capability is claimed.'],
        'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),TRACE,LADDER,OFFICIAL,VIEW,ENCODER,PREV/'delivery.json',PREV/'fit_support_probe.json',PREV/'capability_qualification.json']}}
    save(OUT/'diagnosis.json',report)
    print(json.dumps({k:report[k] for k in ['status','input_chain','crossed_header_682','priority_support','all_fit_conflicts']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
