"""Shared deterministic representation and metric helpers for the v3.3.1 batch."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
from run_v32_train import metrics, risk_threshold, alerts, new_vectorizer, available_memory

LABELS=["benign","malicious","suspicious"]
BUDGETS=[0.0001,0.001,0.01]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+".writing")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8388608),b""):h.update(b)
    return h.hexdigest()


def portable_content(root):
    root=Path(root);h=hashlib.sha256();rows=0
    cols=["row_position","group_id","label","text","product","format","repair_route","asa_template",
          "original_empty","filtered_empty","unknown_format","authentication_result_unknown","no_observable_auth_facts"]
    for batch in pq.ParquetFile(root/"prepared_corpus.parquet").iter_batches(batch_size=4096,columns=cols,use_threads=False):
        for row in batch.to_pylist():
            data=json.dumps([row[k] for k in cols],ensure_ascii=False,separators=(",",":")).encode("utf-8")
            h.update(struct.pack("<I",len(data)));h.update(data);rows+=1
    gm=pq.read_table(root/"group_manifest.parquet")
    groups_digest=hashlib.sha256(gm["group_id"].to_numpy().astype("<i4").tobytes()+
        gm["label_index"].to_numpy().astype("u1").tobytes()+
        gm["informative"].to_numpy().astype("u1").tobytes()).hexdigest()
    return {"rows":rows,"corpus_content_sha256":h.hexdigest(),"group_content_sha256":groups_digest,
            "scope":"Canonical row content, independent of Parquet library/compression metadata."}


def load_data(root):
    root=Path(root)
    gm=pq.read_table(root/"group_manifest.parquet")
    columns=["text","product","format","repair_route","original_empty","filtered_empty","unknown_format",
             "authentication_result_unknown","no_observable_auth_facts","asa_template"]
    table=pq.read_table(root/"prepared_corpus.parquet",columns=columns)
    enc=table["text"].combine_chunks().dictionary_encode()
    data={"texts":enc.dictionary.to_pylist(),"text_ids":enc.indices.to_numpy(),
          "y":gm["label_index"].to_numpy(),"g":gm["group_id"].to_numpy(),"info":gm["informative"].to_numpy()}
    for k in columns[1:]:
        if k in ["product","format","repair_route","asa_template"]:
            d=table[k].combine_chunks().dictionary_encode()
            # Null templates are only metadata; do not use in the predictor.
            data[k]=np.asarray(table[k].to_pylist(),dtype=object)
        else:data[k]=table[k].to_numpy()
    return data


def row_texts(data,positions,stage=""):
    total=len(positions)
    for i,pos in enumerate(positions):
        if stage and i and i%200000==0:
            print(json.dumps({"stage":stage,"rows":i,"total":total}),flush=True)
        yield data["texts"][int(data["text_ids"][pos])]


def unique_matrix(data,positions,vectorizer):
    ids,inverse=np.unique(data["text_ids"][positions],return_inverse=True)
    x=vectorizer.transform([data["texts"][int(i)] for i in ids])
    x.sort_indices()
    return x,inverse,ids


def group_metrics(y,p,g,info):
    keep=np.asarray(info,dtype=bool)
    if not keep.any():return {"scope":"nonempty group-equal","rows":0,"macro_f1":None}
    _,inv,cnt=np.unique(g[keep],return_inverse=True,return_counts=True)
    w=1/cnt[inv]
    cm=np.bincount(y[keep].astype(int)*3+p[keep],weights=w,minlength=9).reshape(3,3)
    den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*cm.diagonal(),den,out=np.zeros(3),where=den>0)
    return {"scope":"Each nonempty observed group has total weight 1; not incident independence.",
            "groups":len(cnt),"confusion_matrix":cm.tolist(),
            "macro_f1":float(f.mean()) if (cm.sum(1)>0).all() else None,
            "class_recall":[float(cm[i,i]/cm.sum(1)[i]) if cm.sum(1)[i] else None for i in range(3)]}


def score_metrics(y,p):
    result=metrics(y,p.argmax(1))
    ranking={}
    for c,name in enumerate(LABELS):
        binary=y==c
        supported=binary.any() and (~binary).any()
        ranking[name]={"average_precision":float(average_precision_score(binary,p[:,c])) if supported else None,
                       "roc_auc":float(roc_auc_score(binary,p[:,c])) if supported else None}
    result["one_vs_rest_ranking"]=ranking
    return result


def probability_table(positions,y,r,n,extra=None):
    import pyarrow as pa
    cols={"row_position":positions.astype(np.int32),"label_index":y,
          "R_pred":r.argmax(1).astype(np.int8),"N_pred":n.argmax(1).astype(np.int8)}
    cols.update({"R_p_"+c:r[:,i] for i,c in enumerate(LABELS)})
    cols.update({"N_p_"+c:n[:,i] for i,c in enumerate(LABELS)})
    cols.update(extra or {})
    return pa.table(cols)


def check_complete(folder,binding):
    folder=Path(folder);path=folder/"complete.json"
    if not path.exists():return None
    receipt=load(path)
    if receipt["binding"]!=binding:raise ValueError("Resume identity mismatch: "+str(folder))
    attempt=folder/receipt["attempt"]
    for name,value in receipt["files"].items():
        if sha(attempt/name)!=value:raise ValueError("Completed task damaged: "+str(attempt/name))
    return attempt

