"""No-fit exact-input collision audit for narrowly specified placeholder views."""
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy import sparse
from v89_common import ROOT, OUT, data, read, save, sha
from v75_views import byte_matrix, BYTE_FEATURES, matrix_hashes
from v75_corrective import stable

DEST=ROOT/'artifacts/v99_evidence_training_plan_20260928'
V92=ROOT/'artifacts/v92_evidence_training_20260928'
FRAGMENT=re.compile(r'((?:CRED|HOST|USER|ORG)-)([ \t]*)(?=<IDENTITY>)')
CLUSTER=re.compile(r'[ \t]*(?:(?:CRED|HOST|USER|ORG)-[ \t]*)*<IDENTITY>[ \t]*')


def normalize(text,mode):
    if mode=='literal_only':return FRAGMENT.sub(lambda m:m.group(2),text)
    if mode=='placeholder_cluster':return CLUSTER.sub(' <IDENTITY> ',text)
    raise ValueError(mode)


def main():
    assert not DEST.exists();DEST.mkdir()
    r,y,fid,_,_,_,_=data();isa=r.route.eq('asa').to_numpy();row=np.flatnonzero(isa)
    roles=pd.read_parquet(ROOT/'artifacts/v95_cross_component_training_20260928/full_format_roles.parquet',columns=['role']).role.to_numpy().astype(object)
    for name,fold in [('inner',1),('C',2),('H',0)]:roles[r.fold.eq(fold).to_numpy()]=name
    ids=np.load(V92/'ASA_input_ids.npy');x=sparse.load_npz(V92/'ASA_R0.npz')
    lookup=np.full(int(fid.max())+1,-1,np.int32);lookup[ids]=np.arange(len(ids));local=lookup[fid[row]]
    textdict=pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    # Use distinct source text and R0 pairs: byte bigrams can merge different strings.
    pairs=r.iloc[row][['new_text_id']].copy();pairs['R0']=fid[row]
    distinct=pairs.drop_duplicates().reset_index(drop=True)
    strings=[stable(textdict.loc[t]) for t in distinct.new_text_id]
    originals=x[lookup[distinct.R0.to_numpy()]]
    byte=byte_matrix(strings);difference=byte-originals[:,:BYTE_FEATURES]
    assert not difference.nnz or abs(difference.data).max()<1e-7
    pair_to_id={(int(t),int(f)):i for i,(t,f) in enumerate(zip(distinct.new_text_id,distinct.R0))}
    pair_rows=np.array([pair_to_id[(int(t),int(f))] for t,f in zip(pairs.new_text_id,pairs.R0)],np.int32)
    modes={};newgroups={}
    # Audit each observed role separately as well as all rows. All labels are development-only.
    def counts(group,mask):
        take=group[mask];truth=y[row][mask]
        cc=np.zeros((int(group.max())+1,3),np.int64);np.add.at(cc,(take,truth),1)
        return {'rows':len(truth),'unique_inputs':int((cc.sum(1)>0).sum()),
            'empirical_conflict_floor':int((cc.sum(1)-cc.max(1)).sum()),
            'conflict_groups':int(((cc>0).sum(1)>1).sum())}
    population={'AB':np.isin(roles[row],['A','B']),'V':roles[row]=='V',
        'inner':roles[row]=='inner','C':roles[row]=='C','H':roles[row]=='H','all':np.ones(len(row),bool)}
    baseline={name:counts(local,mask) for name,mask in population.items()}
    for mode in ['literal_only','placeholder_cluster']:
        texts=[normalize(t,mode) for t in strings]
        assert all(normalize(t,mode)==t for t in texts)
        changes=np.array([a!=b for a,b in zip(strings,texts)])
        xx=sparse.hstack([byte_matrix(texts),originals[:,BYTE_FEATURES:]],format='csr')
        assert (xx[:,BYTE_FEATURES:]-originals[:,BYTE_FEATURES:]).nnz==0
        hashes=matrix_hashes(xx);groups,unique=pd.factorize(hashes,sort=False);rowgroups=groups[pair_rows]
        # Explicitly verify every hash merge: no hash collision is accepted as input equality.
        order=np.argsort(groups,kind='stable')
        for a,b in zip(order[:-1],order[1:]):
            if groups[a]==groups[b]:assert (xx[a]-xx[b]).nnz==0
        newgroups[mode]=rowgroups
        cases=[]
        frame=pd.DataFrame({'group':rowgroups,'label':y[row],'row_position':row,'old_input':fid[row],'role':roles[row]})
        merged=frame.groupby('group').old_input.nunique()
        for g in merged[merged>1].index:
            cell=frame[frame.group==g]
            if cell.label.nunique()>1:
                cases.append({'group':int(g),'rows':len(cell),'old_input_count':int(cell.old_input.nunique()),
                    'labels':cell.label.value_counts().sort_index().to_dict(),
                    'representatives':cell.groupby(['old_input','label']).row_position.first().astype(int).tolist()})
        save(DEST/f'{mode}_mixed_merge_groups.json',cases)
        pd.DataFrame({'row_position':row,'old_R0':fid[row],'new_group':rowgroups}).to_parquet(DEST/f'{mode}_group_map.parquet',index=False)
        modes[mode]={'modified_rows':int(changes[pair_rows].sum()),'modified_text_input_pairs':int(changes.sum()),
            'idempotence_checked_pairs':len(texts),'facts_metadata_difference_nnz':0,
            'populations':{name:counts(rowgroups,mask) for name,mask in population.items()},
            'merged_mixed_label_groups':len(cases)}
    audit={'status':'normalization_feasibility_only_no_fit','source_sha256':sha(__file__),
        'official_ASA_rows':len(row),'actual_R0_inputs':len(ids),'distinct_text_input_pairs':len(strings),
        'baseline':baseline,'modes':modes,'classifier_fits':0,'calibration_fits':0,
        'scope':'Narrow text-only candidates; exact current byte+facts+metadata input collisions. No claim of semantic sufficiency, learned robustness or raw-field/sequence losslessness.'}
    save(DEST/'normalization_feasibility.json',audit);print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
