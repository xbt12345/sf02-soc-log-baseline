"""Runtime integrity, metric scopes, and role isolation shared by all four arms."""
from pathlib import Path
import numpy as np
import pandas as pd
from v61_common import read, sha, metrics, FIELDS


def load_data(path):
    path=Path(path);config=read(path/'configuration.json')
    for name in ['records.parquet','context.npz','data_audit.json']:
        assert sha(path/name)==config['data_bindings'][name],name
    assert sha(Path(__file__).with_name('v61_common.py'))==config['source_bindings']['training/v61_common.py']
    df=pd.read_parquet(path/'records.parquet');ctx=dict(np.load(path/'context.npz'))
    assert len(df)==config['input_rows'] and df.row_position.is_unique
    assert df.groupby('group').role.nunique().max()==1
    assert df.groupby('body_group').role.nunique().max()==1
    n=ctx['neighbors'];valid=n>=0
    assert ((df.role.to_numpy()[np.maximum(n,0)]==df.role.to_numpy()[:,None])|~valid).all()
    assert np.isfinite(ctx['stats']).all()
    return df,ctx,config


def score(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return {'ASA':metrics(y[y>0],p[y>0]),'all_including_normal_controls':metrics(y,p),
            'normal_controls_are_too_few_for_operational_FPR':True}


def predictions(path,df,idx,p):
    out=df.iloc[idx][['row_position','label','group']].copy()
    out[['p_B','p_M','p_S']]=p
    out.to_parquet(path,index=False)


def source_receipt(names):
    root=Path(__file__).parent
    return {n:sha(root/n) for n in names}
