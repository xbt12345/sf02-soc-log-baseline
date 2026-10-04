"""No-fit interface isolation and naturally observed fine-control eligibility."""
import json,re
import numpy as np
import pandas as pd
from scipy import sparse
from run_v75 import ROOT,save,sha
from v75_views import BYTE_FEATURES,byte_matrix,matrix_hashes
from v110_layer_probes import LEDGER,PREV,check_registration

DEST=ROOT/'artifacts/v112_fine_supervision_review_20260929'
N1=ROOT/'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'
SRC=re.compile(r'\bDeny\s+(?:tcp|udp|icmp)\s+src\s+([A-Za-z][A-Za-z0-9_.-]*):')
DST=re.compile(r'\bdst\s+([A-Za-z][A-Za-z0-9_.-]*):')
FAMILY=re.compile(r'^(inside|outside|dmz)(?:[-_]\d+)?$',re.I)


def floor(d,key):
    tab=d.groupby([key,'truth'],dropna=False).size().unstack(fill_value=0)
    mixed=(tab>0).sum(1)>1
    return {'groups':len(tab),'mixed_groups':int(mixed.sum()),
       'mixed_rows':int(tab.loc[mixed].sum().sum()),
       'observed_min_errors':int((tab.sum(1)-tab.max(1)).sum())}


def eligibility(train,key):
    # Set members are unique (source, literal-pair) states; repeat events do not
    # turn into additional independent positive/negative pairs.
    cache={}
    for keys,g in train.groupby([key,'truth'],dropna=False):
        states=set(zip(g.root,g.interface_literal_pair))
        cache[keys]=states
    result=[]
    for v,y,r,n in train[[key,'truth','root','interface_literal_pair']].itertuples(index=False,name=None):
        pos=cache.get((v,y),set())
        neg=cache.get((v,3-int(y)),set())
        positive=any(rr!=r and nn!=n for rr,nn in pos)
        negative=any(rr!=r and nn==n for rr,nn in neg)
        result.append((positive,negative))
    return np.asarray(result,dtype=bool)


def main():
    check_registration()
    assert not DEST.exists() or not any(DEST.iterdir());DEST.mkdir(exist_ok=True)
    d=pd.read_parquet(LEDGER)
    pairs=PREV/'N2_text_pairs.parquet'
    text=pd.read_parquet(pairs)
    assert text.groupby('local').before.nunique().max()==1
    text_mapping_rows=len(text)
    text=text.drop_duplicates('local').sort_values('local').reset_index(drop=True)
    assert np.array_equal(text.local,np.arange(22546))
    fx=pd.read_parquet(ROOT/'artifacts/v75_four_arm_20260921_r2/projections.parquet',columns=['facts'])
    local_rows=d.drop_duplicates('local').set_index('local').sort_index()
    assert len(local_rows)==22546
    records=[];modified=[]
    for i,s in enumerate(text.before):
        f=json.loads(fx.facts.iat[int(local_rows.loc[i,'projection_id'])])
        a=SRC.search(s);b=DST.search(s,a.end()) if a else None
        aliases=[];spans=[]
        if a and b:
            for which,m in [('src',a),('dst',b)]:
                token=m[1];family=FAMILY.fullmatch(token)
                if not family or f.get(which+'_role')!=family[1].lower():break
                aliases.append(token);spans.append((m.start(1),m.end(1),which))
        eligible=len(aliases)==2
        out=s
        if eligible:
            for start,end,role in reversed(spans):
                out=out[:start]+'<'+role.upper()+'_INTERFACE_NAME>'+out[end:]
        modified.append(out)
        records.append({'local':i,'interface_parse_verified':eligible,
          'src_interface':aliases[0] if eligible else None,'dst_interface':aliases[1] if eligible else None,
          'interface_literal_pair':json.dumps(aliases,ensure_ascii=False) if eligible else None,
          'before':s,'isolated':out,
          'name_spans':json.dumps(spans) if eligible else '[]'})
    views=pd.DataFrame(records)
    x=sparse.load_npz(N1)
    base_text=byte_matrix(text.before.tolist())
    assert (x[:,:BYTE_FEATURES]!=base_text).nnz==0
    nx=sparse.hstack([byte_matrix(modified),x[:,BYTE_FEATURES:]],format='csr')
    assert (nx[:,BYTE_FEATURES:]!=x[:,BYTE_FEATURES:]).nnz==0
    nh=matrix_hashes(nx)
    views['isolated_full_input_hash']=nh
    views.to_parquet(DEST/'interface_views.parquet',index=False)
    z=d[['row_position','local','root','fold','truth','behavior','coarse','N1_TabM25']].merge(
        views[['local','interface_parse_verified','interface_literal_pair','isolated_full_input_hash']],
        on='local',how='left',validate='many_to_one',sort=False)
    z['original_full_input_hash']=d.N1_hash.to_numpy()
    assert np.array_equal(z.row_position,d.row_position)
    oldfloor=floor(z,'original_full_input_hash');newfloor=floor(z,'isolated_full_input_hash')
    old_new=z.groupby('isolated_full_input_hash').original_full_input_hash.nunique()
    collapsed=set(old_new[old_new>1].index)
    q=z[z.isolated_full_input_hash.isin(collapsed)]
    foldcross=z.groupby('isolated_full_input_hash').fold.nunique()
    cross=set(foldcross[foldcross>1].index)
    # Source-union count is a diagnostic only. We do not assign new folds here.
    parent={int(r):int(r) for r in z.root.unique()}
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    for _,g in z.groupby('isolated_full_input_hash'):
        rr=g.root.unique()
        if len(rr)>1:
            for r in rr[1:]:
                a=find(int(rr[0]));b=find(int(r));parent[max(a,b)]=min(a,b)
    z['candidate_joint_root']=[find(int(r)) for r in z.root]
    # Eligibility is not a training pair manifest. No held-out rows are eligible
    # partners for the corresponding fold, even though their labels are known.
    eligible_rows=[];stats=[]
    for k in (0,1,2):
        for resolution,key in [('exact_behavior','behavior'),('exact_remaining_input','isolated_full_input_hash')]:
            tr=z[(z.fold!=k)&z.interface_parse_verified&z[key].notna()].copy()
            p=eligibility(tr,key)
            tr['positive_cross_source_cross_name']=p[:,0]
            tr['negative_cross_source_same_name']=p[:,1]
            tr['both_controls']=p.all(1)
            for c,label in [(1,'M'),(2,'S')]:
                a=tr[tr.truth==c]
                stats.append({'fold':k,'resolution':resolution,'class':label,'available_train_rows':len(a),
                   'positive_rows':int(a.positive_cross_source_cross_name.sum()),
                   'negative_rows':int(a.negative_cross_source_same_name.sum()),
                   'both_rows':int(a.both_controls.sum()),
                   'both_roots':int(a.loc[a.both_controls,'root'].nunique()),
                   'both_unique_inputs':int(a.loc[a.both_controls,'local'].nunique())})
            tr['audit_fold']=k;tr['resolution']=resolution
            eligible_rows.append(tr[['row_position','local','root','truth','audit_fold','resolution',
               'positive_cross_source_cross_name','negative_cross_source_same_name','both_controls']])
    pd.concat(eligible_rows).to_parquet(DEST/'training_control_eligibility.parquet',index=False)
    # Local minimum-error sets may be unsuitable contrastive negatives.
    conflicts=z.groupby('isolated_full_input_hash').truth.nunique()
    mixed=set(conflicts[conflicts>1].index)
    z['isolated_input_conflict']=z.isolated_full_input_hash.isin(mixed)
    z.to_parquet(DEST/'interface_row_audit.parquet',index=False)
    casekey=z.loc[z.root==2868,'behavior'].iloc[0]
    case=z[(z.behavior==casekey)&(z.fold!=1)]
    summary={'status':'no_fit_fine_control_eligibility_and_projection_audit','classifier_fits':0,'calibration_fits':0,
       'source_sha256':sha(__file__),'input_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in [LEDGER,pairs,N1,ROOT/'artifacts/v75_four_arm_20260921_r2/projections.parquet']},
       'total_ASA_rows':len(z),'text_mapping_rows_before_unique_local':text_mapping_rows,
       'verified_interface_rows':int(z.interface_parse_verified.sum()),
       'unverified_rows':int((~z.interface_parse_verified).sum()),
       'verified_unique_inputs':int(views.interface_parse_verified.sum()),
       'original_floor':oldfloor,'isolated_floor':newfloor,
       'new_collapsed_full_input_groups':len(collapsed),'rows_in_collapsed_groups':len(q),
       'cross_current_fold_equal_input_groups':len(cross),'rows_in_cross_fold_groups':int(z.isolated_full_input_hash.isin(cross).sum()),
       'old_roots':int(z.root.nunique()),'joint_roots':int(z.candidate_joint_root.nunique()),
       'largest_old_root_rows':int(z.groupby('root').size().max()),'largest_joint_root_rows':int(z.groupby('candidate_joint_root').size().max()),
       'by_interface_and_class':z[z.interface_parse_verified].groupby(['interface_literal_pair','truth']).agg(rows=('row_position','size'),roots=('root','nunique')).reset_index().to_dict('records'),
       'train_fine_control_eligibility':stats,
       'root2868_fold1_train_same_behavior':case.groupby(['interface_literal_pair','truth']).agg(rows=('row_position','size'),roots=('root','nunique')).reset_index().to_dict('records'),
       'limits':['Interface family is checked against existing parsed roles, not independently certified network-zone equivalence.',
        'Isolation is an ablation candidate; original interface names and logs are preserved, no safe relabeling claim.',
        'Exact behavior equality does not imply all security-relevant facts match; eligibility numbers are candidate retrieval, not approved causal pairs.',
        'Equal remaining complete model input with different labels cannot support a deterministic separation loss.',
        'No source-held-out partner or synthetic label was added. Original folds must be rebuilt if equal-input edges cross them.',
        'Source components are isolation proxies, not known independent customer environments.']}
    summary['output_hashes']={p.name:sha(p) for p in DEST.glob('*.parquet')}
    save(DEST/'preflight.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['by_interface_and_class','input_hashes','output_hashes']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
