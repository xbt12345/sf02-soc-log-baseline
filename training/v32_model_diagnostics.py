"""Frozen-model representation stress and exact vector-collision diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import v32_finalize as final
import soc_v3_prepare as base


def stress(run,vectorizer,model):
    pairs=json.loads((Path(run)/"model_stress_fixtures.json").read_text(encoding="utf-8"))
    results=[]
    for pair in pairs:
        a=final.prepare_record({"message_sanitized":pair["raw_a"]})["text"]
        b=final.prepare_record({"message_sanitized":pair["raw_b"]})["text"]
        x=vectorizer.transform([a,b]);p=model.predict_proba(x)
        item={"name":pair["name"],"kind":pair["kind"],"texts_equal":a==b,
          "vectors_equal":(x[0]-x[1]).nnz==0,"zero_vectors":(np.diff(x.indptr)==0).tolist(),
          "scores":p.tolist(),"max_score_change":float(np.max(np.abs(p[0]-p[1]))),
          "class_changed":bool(p[0].argmax()!=p[1].argmax())}
        results.append(item)
    invariants=[r for r in results if r["kind"]=="invariance"]
    passed=all(r["texts_equal"] and r["vectors_equal"] and r["max_score_change"]<1e-12 for r in invariants)
    return {"all_invariance_checks_passed":passed,"pairs":results,
      "semantic_pairs_lost_at_vectorization":[r["name"] for r in results if r["kind"]=="semantic_difference" and not r["texts_equal"] and r["vectors_equal"]],
      "scope":"Boundary counterexamples; no attack labels, fit, forced class flip or transfer claim. Semantic differences may be outside frozen vocabulary."}


def vector_collisions(run,mask,groups,y,vectorizer):
    seen={};collision={};zero=0;rows=0
    for batch in pq.ParquetFile(Path(run)/"prepared_corpus.parquet").iter_batches(batch_size=2048,columns=["row_position","text"],use_threads=False):
        pos=batch.column(0).to_numpy();keep=mask[pos]
        if not keep.any():continue
        texts=batch.column(1).to_pylist();pos=pos[keep]
        x=vectorizer.transform([s for s,k in zip(texts,keep) if k]);x.sort_indices()
        for i,p in enumerate(pos):
            rows+=1;a,b=x.indptr[i:i+2]
            if a==b:zero+=1;continue
            h=hashlib.sha256(x.indices[a:b].astype('<i4',copy=False).tobytes()+x.data[a:b].astype('<f4',copy=False).tobytes()).digest()
            g=int(groups[p]);label=int(y[p])
            if h not in seen:seen[h]=[g,[0,0,0]]
            entry=seen[h];entry[1][label]+=1
            if entry[0]!=g:
                collision.setdefault(h,set()).update([entry[0],g])
    items=[{"vector_sha256":h.hex(),"different_text_groups":len(gs),"class_counts":dict(zip(base.LABELS,seen[h][1]))} for h,gs in collision.items()]
    return {"outer_rows":rows,"zero_vector_rows":zero,"distinct_nonzero_vectors":len(seen),
      "nonzero_vectors_shared_by_different_normalized_text_groups":len(items),
      "rows_in_those_vectors":sum(sum(seen[h][1]) for h in collision),
      "mixed_label_vectors_in_those_collisions":sum(sum(c>0 for c in seen[h][1])>1 for h in collision),
      "examples":items[:50],"scope":"Exact sorted float32 sparse vectors within this frozen outer fold. Approximate collisions not counted; vector equality is not event identity."}
