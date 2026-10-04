"""Fixed-model feature removal diagnostic; not a new deployable representation."""
from pathlib import Path
import warnings
import joblib,numpy as np,pyarrow.parquet as pq
from v331_common import load_data,save,load,metrics,LABELS
from audit_v331_cloud_return import legacy_unique_matrix
from review_v331_round import manual_probabilities

root=Path(__file__).resolve().parents[1];prep=root/"artifacts/v331_ready_20260912_r2"
d=load_data(prep);manifest=pq.read_table(prep/"protocol_manifest.parquet")
results={}
for name in ["source_ad","source_waf","source_duo"]:
    folder=root/"artifacts/v331_cloud_return_20260912T122349Z/experiment/tasks"/name/"attempt_1"
    pos=np.flatnonzero(manifest[name].to_numpy()==3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore");v=joblib.load(folder/"vectorizer.joblib");model=joblib.load(folder/"model.joblib")
    x,inv,_=legacy_unique_matrix(d,pos,v);features=v.get_feature_names_out()
    selected=np.array([any(t.startswith("op_") for t in s.split()) for s in features])
    original=manual_probabilities(x,model)[inv]
    x.data[selected[x.indices]]=0;x.eliminate_zeros()
    after=manual_probabilities(x,model)[inv]
    words={}
    for token in ["failed","failure","denied","invalid_passcode","unknown_value","4625","reason","outcome","op_equals","op_backslash","op_pipe"]:
        j=v.vocabulary_.get(token)
        words[token]={"in_vocabulary":j is not None,"coefficients":model.coef_[:,j].tolist() if j is not None else None}
    results[name]={"scope":"All operator-bearing columns zeroed after TF-IDF, without renormalization. Includes potential payload operators; information-losing association diagnostic, not a semantics-preserving fix.",
        "removed_columns":int(selected.sum()),"before":metrics(d["y"][pos],original.argmax(1)),"after":metrics(d["y"][pos],after.argmax(1)),
        "class_flips":int((original.argmax(1)!=after.argmax(1)).sum()),"selected_word_coefficients":words}
    print(name,results[name]["before"]["confusion_matrix"],results[name]["after"]["confusion_matrix"],flush=True)
save(root/"evidence/2026-09-12/v331_result_review/v331_operator_ablation.json",results)
