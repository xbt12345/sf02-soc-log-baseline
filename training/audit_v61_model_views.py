"""Measure conflicts at model inputs, including unknown-category and neighbor caps."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from audit_v61_inputs import minimum_errors
from v61_common import FIELDS, fit_vocab, encode_facts, save, sha
from v61_runtime import load_data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);a=ap.parse_args()
    df,ctx,_=load_data(a.data);fit=df.role.eq('fit').to_numpy()
    facts=df[FIELDS].astype(str).to_numpy();encoded=encode_facts(facts,fit_vocab(facts[fit]))
    target=pd.factorize(pd.MultiIndex.from_frame(df[['text']+FIELDS]),sort=False)[0]
    # Finite typed IDs are categorical INPUT values, not entity IDs or labels.
    # The transformer tokenizer has separately passed full text round-trip tests.
    text_code=pd.factorize(df.text,sort=False)[0]
    b=[];c=[];d=[]
    for i in range(len(df)):
        bk=(int(text_code[i]),tuple(encoded[i]));b.append(bk)
        c.append((int(target[i]),ctx['stats'][i].tobytes()))
        ns=ctx['neighbors'][i];rs=ctx['relation'][i]
        states=tuple(sorted((tuple(encoded[j]),tuple(rs[k])) for k,j in enumerate(ns) if j>=0))
        d.append((bk,ctx['stats'][i].tobytes(),states))
    keys={name:pd.factorize(pd.Series(values,dtype=object),sort=False)[0] for name,values in [('B',b),('C_information_before_linear_vectorization',c),('D',d)]}
    report={}
    for role in ['fit','selection','evaluation']:
        mask=df.role.eq(role).to_numpy()
        report[role]={name:minimum_errors(key[mask],df.label.to_numpy()[mask]) for name,key in keys.items()}
        report[role]['rows_with_unknown_fact_categories']=int((encoded[mask]==1).any(1).sum())
    save(a.data/'model_view_ambiguity.json',{'by_role':report,
        'B_and_D_views':'Lossless tokenizer text plus exact categorical IDs; D additionally actual capped neighbor codes and full context statistics',
        'C_view_is_information_upper_bound_not_full_TFIDF_collision_audit':True,
        'label_conflict_minimum_is_retrospective_per_role_not_achievable_generalization':True,
        'not_used_to_change_training_or_epoch_choice':True,'source_sha256':sha(__file__)})
    print(report,flush=True)


if __name__=='__main__':main()
