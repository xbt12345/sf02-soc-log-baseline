"""Independent source/mass/decision audit and saved-input geometry witnesses.

No new official model execution. H2 identity/count checks use executor output;
numerical distance checks use actual CSR and already saved H1 only.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
from v135_runtime import load_data

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'artifacts/v156_conditional_representation_neighborhood_20261001'
OUT = ROOT/'artifacts/v156_independent_neighborhood_audit_20261001'
PREV = ROOT/'artifacts/v155_guarded_full_gradient_sam_20261001'
REF = ROOT/'artifacts/v153_independent_training_transfer_gap_20261001'
SPACES = ['actual_CSR_input', 'all16_H1', 'all16_V146_A_H2']
TIE = 1e-12
NUMERIC_BOUND = 5e-12


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def verify_bindings(values):
    for name, expected in values.items():
        p=Path(name)
        if not p.is_absolute():
            p=ROOT/p
        assert sha(p)==expected, name


def profile(rows, name, mask):
    selected=rows.loc[mask]
    result=[]
    for (space,role,cl,relation),g in selected.groupby(['space','query_role','truth','relation']):
        result.append(dict(cohort=name,space=space,role=role,truth=int(cl),relation=relation,
            original_role_rows=len(g),roots=int(g.root.nunique()),
            V146_A_errors=int(g.pred_baseline.ne(g.truth).sum()),
            V155_A_errors=int(g.pred_V155_A.ne(g.truth).sum()),
            V155_B_errors=int(g.pred_V155_B.ne(g.truth).sum())))
    return result


def main():
    assert (RUN/'audit.json').exists() and not OUT.exists()
    audit=read(RUN/'audit.json');seal=read(RUN/'run_seal.json')
    verify_bindings(audit['source_sha256']);verify_bindings(seal['source_sha256'])
    assert audit['seal_sha256']==sha(RUN/'run_seal.json')
    assert audit['feature_function_calls']==3
    assert audit['model_forward_calls']==audit['gradients']==audit['classifier_fits']==audit['updates']==0
    assert len(audit['feature_reports'])==3
    assert all(z['feature_function_calls']==1 and z['model_state_unchanged'] for z in audit['feature_reports'])
    original=pd.read_parquet(RUN/'original_reference.parquet')
    nodes=pd.read_parquet(RUN/'local_reference.parquet')
    previous=pd.read_parquet(REF/'all_original_classifier_gap_and_control_ledger.parquet')
    assert len(original)==112807 and original.row_position.is_unique
    assert np.array_equal(original.row_position,previous.row_position)
    for col in ['local','root','fold','truth','canonical_key']:
        assert np.array_equal(original[col],previous[col]),col
    assert len(nodes)==22546 and np.array_equal(nodes.local,np.arange(22546))
    assert original.groupby('local').root.nunique().eq(1).all()
    allrows=pd.read_parquet(RUN/'all_original_role_neighbors.parquet')
    assert len(allrows)==1015263
    assert not allrows.duplicated(['row_position','training_role','space']).any()
    roots=nodes.root.to_numpy();conditions=nodes.condition.to_numpy();witness_checks=0
    summary=[];numeric=[];maximum_gap=0.
    x,loaded=load_data()
    assert np.array_equal(loaded.row_position,original.row_position)
    raw=x.astype(np.float64);norm=np.sqrt(raw.multiply(raw).sum(1)).A1
    raw=(sparse.diags(1/np.where(norm==0,1,norm))@raw).tocsr()
    for fold in range(3):
        train=original[original.fold.ne(fold)]
        counts=np.zeros((22546,3),np.int64)
        np.add.at(counts,(train.local.to_numpy(),train.truth.to_numpy()),1)
        old=np.load(PREV/f'fold{fold}_zero_probability.npy')
        qa=np.load(PREV/f'fold{fold}_A/sealed_all_prob.npy')
        qb=np.load(PREV/f'fold{fold}_B/sealed_all_prob.npy')
        h1=np.load(ROOT/f'artifacts/v141_representation_evidence_20261001/fold{fold}_h1.npy').astype(np.float64).reshape(22546,-1)
        assert h1.shape==(22546,2048)
        hn=np.linalg.norm(h1,axis=1);h1/=np.where(hn==0,1,hn)[:,None]
        sample=set()
        for _,g in nodes.groupby('condition'):
            sample.update([int(g.local.iloc[0]),int(g.local.iloc[-1])])
        sample.update([226,1132,19885])
        for space in SPACES:
            g=pd.read_parquet(RUN/f'fold{fold}_{space}_local_neighbors.parquet')
            assert len(g)==22546 and np.array_equal(g.local,nodes.local)
            for col in ['root','fold','condition']:
                assert np.array_equal(g[col],nodes[col])
            part=allrows[(allrows.training_role==fold)&allrows.space.eq(space)].reset_index(drop=True)
            assert len(part)==112807 and np.array_equal(part.row_position,original.row_position)
            for col in original.columns:
                assert np.array_equal(part[col],original[col])
            loc=part.local.to_numpy()
            assert np.array_equal(part.pred_baseline,old[loc].argmax(1))
            assert np.array_equal(part.pred_V155_A,qa[loc].argmax(1))
            assert np.array_equal(part.pred_V155_B,qb[loc].argmax(1))
            assert np.array_equal(part.query_role,np.where(original.fold.eq(fold),'outer_HELD','legal_TRAIN'))
            for col in g.columns:
                assert np.array_equal(part[col].to_numpy(),g[col].to_numpy()[loc],equal_nan=part[col].dtype.kind=='f'),col
            for cl in [1,2]:
                mass=train[train.truth.eq(cl)]
                total=mass.groupby('condition').size()
                own=mass.groupby(['condition','root']).size()
                all_local=mass[['condition','root','local']].drop_duplicates()
                total_loc=all_local.groupby('condition').size()
                own_loc=all_local.groupby(['condition','root']).size()
                keys=pd.MultiIndex.from_frame(nodes[['condition','root']])
                avail=nodes.condition.map(total).fillna(0).to_numpy()-own.reindex(keys,fill_value=0).to_numpy()
                avail_loc=nodes.condition.map(total_loc).fillna(0).to_numpy()-own_loc.reindex(keys,fill_value=0).to_numpy()
                assert np.array_equal(g[f'c{cl}_available_original_rows'],avail)
                assert np.array_equal(g[f'c{cl}_available_locals'],avail_loc)
                supported=avail>0
                chosen=g[f'c{cl}_local'].to_numpy().astype(np.int64)
                assert np.isfinite(g[f'c{cl}_cosine']).to_numpy().tolist()==supported.tolist()
                assert np.all(chosen[~supported]==-1)
                assert np.all(g.loc[~supported,[f'c{cl}_tied_locals',f'c{cl}_tied_original_rows',f'c{cl}_tied_roots']].to_numpy()==0)
                take=chosen[supported]
                assert (roots[take]!=roots[supported]).all()
                assert (nodes.fold.to_numpy()[take]!=fold).all()
                assert (conditions[take]==conditions[supported]).all()
                assert (counts[take,cl]>0).all()
                assert (g.loc[supported,f'c{cl}_tied_original_rows'].to_numpy()>=counts[take,cl]).all()
                assert (g[f'c{cl}_tied_original_rows'].to_numpy()<=avail).all()
                assert (g[f'c{cl}_tied_locals'].to_numpy()<=avail_loc).all()
                witness_checks+=int(supported.sum())
                if space=='all16_V146_A_H2':
                    continue
                # Exhaustive legal candidate distances for a fixed deterministic
                # sample, no H2 function replay and no actual classifier.
                vectors=raw if space=='actual_CSR_input' else h1
                for query in sorted(sample):
                    cand=np.flatnonzero((conditions==conditions[query])&(roots!=roots[query])&(counts[:,cl]>0))
                    if not len(cand):
                        numeric.append(dict(fold=fold,space=space,query_local=query,neighbor_class=cl,candidates=0))
                        continue
                    sim=(vectors[cand]@vectors[query].T).toarray().ravel() if space=='actual_CSR_input' else vectors[cand]@vectors[query]
                    maximum=float(sim.max());stored=float(g.loc[query,f'c{cl}_cosine'])
                    gap=abs(maximum-stored);maximum_gap=max(maximum_gap,gap)
                    assert gap<=NUMERIC_BOUND,(fold,space,query,cl,gap)
                    definite=sim>=maximum-TIE+NUMERIC_BOUND
                    possible=sim>=maximum-TIE-NUMERIC_BOUND
                    reported=int(g.loc[query,f'c{cl}_tied_original_rows'])
                    assert int(counts[cand[definite],cl].sum())<=reported<=int(counts[cand[possible],cl].sum())
                    numeric.append(dict(fold=fold,space=space,query_local=query,neighbor_class=cl,
                        candidates=len(cand),maximum_distance_gap=gap,possible_tied_locals=int(possible.sum())))
            same=np.where(part.truth.eq(1),part.c1_cosine,part.c2_cosine)
            other=np.where(part.truth.eq(1),part.c2_cosine,part.c1_cosine)
            expected=np.select([np.isnan(same)&np.isnan(other),np.isnan(same),np.isnan(other),same>other+TIE,other>same+TIE],
                ['no_either_class','no_same_class','no_other_class','same_class_closer','other_class_closer'],default='class_tie')
            assert np.array_equal(expected,part.relation)
            assert int(part.zero_vector.sum())==0
            for (role,cl,relation),chunk in part.groupby(['query_role','truth','relation'],sort=True):
                summary.append(dict(fold=fold,space=space,role=role,truth=int(cl),relation=relation,original_rows=len(chunk),roots=int(chunk.root.nunique()),
                    baseline_errors=int(chunk.pred_baseline.ne(chunk.truth).sum()),V155_A_errors=int(chunk.pred_V155_A.ne(chunk.truth).sum()),V155_B_errors=int(chunk.pred_V155_B.ne(chunk.truth).sum())))
    assert summary==audit['summaries']
    enriched=allrows.merge(previous[['row_position','known_578_cohort','same_family_and_outer_fold_control_S']],on='row_position',validate='many_to_one')
    slices=[]
    for name,mask in [('all',np.ones(len(enriched),bool)),('hard578',enriched.known_578_cohort),('correct51',enriched.same_family_and_outer_fold_control_S)]:
        slices.extend(profile(enriched,name,mask))
    result=dict(status='all_roles_original_masses_source_exclusion_and_saved_decisions_recounted',original_rows=112807,
        role_space_rows=len(allrows),all_supported_class_witness_identity_checks=witness_checks,
        numerical_CSR_H1_witnesses=len(numeric),maximum_CSR_H1_cosine_gap=maximum_gap,
        feature_model_calls=0,official_classifier_forwards=0,gradients=0,fits=0,updates=0,
        limits=['H2 distances are from bounded executor function calls; no parent H2 replay was performed.',
                'Cosine geometry and same-class source availability do not constitute threat evidence or new classifications.',
                'CPU/GPU distance checks use a predeclared 5e-12 arithmetic bound; actual saved classification decisions remain exact.'],
        profiles=slices,source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__),RUN/'audit.json',RUN/'run_seal.json',RUN/'all_original_role_neighbors.parquet']})
    OUT.mkdir()
    pd.DataFrame(numeric).to_parquet(OUT/'bounded_CSR_H1_distance_witnesses.parquet',index=False)
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['profiles','source_sha256']},ensure_ascii=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
