"""Current root plan plus exact raw geometry/draft; no model execution."""
import hashlib,json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v166_direction_20261002'
ROOTPLAN=ROOT/'docs/V166_ROOT_ACTUAL_REVIEW_AND_V167_TRIAL_POINT_RESTORATION_PLAN_20261002.md'
DRAFT=ROOT/'training/review_policy/v167_trial_point_restoration_draft.json'
GEOMETRY=ROOT/'artifacts/v166_independent_covered_margin_geometry_review_20261002/review.json'
BOUNDARIES=ROOT/'training/review_policy/v166_observed_runtime_boundaries_v2.json'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists();draft=read(DRAFT);geometry=read(GEOMETRY);replay=ROOT/'artifacts/v166_observed_runtime_failure_replay_v2_20261002/replay.json';replayed=read(replay);check_bindings(replayed['source_sha256']);assert draft['execution_authority'] is False and draft['proposed_caps']['heads']==296 and draft['proposed_caps']['margin_gradients']==100 and draft['proposed_caps']['fits']==draft['proposed_caps']['permanent_updates']==0;assert geometry['roles'][1]['covered_blocked_functions']==5 and geometry['roles'][1]['unmeasured_blocked_functions']==0 and replayed['no_new_execution_authority'];assert len(replayed['cases'])==5
    source_binding=GEOMETRY.parent/'pre_review_bindings.json'
    if not source_binding.exists():source_binding=GEOMETRY.parent/'source_bindings.json'
    check_bindings(read(source_binding)['source_sha256'])
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']];catalog=read(mutable[2]);historic={r['id']:dict(r) for r in catalog['documents']};assert catalog['project']['authoritative_direction_id']=='v166-review' and catalog['project']['authoritative_delivery_id']=='v159-delivery'
    for row in catalog['documents']:assert sha(ROOT/row['path'])==row['sha256']
    old=ROOT/'artifacts/v166_results_20261002/review.md';retained=old.read_text(encoding='utf-8').split('## 既有证据限制完整保留\n\n',1)[1];original=ROOTPLAN.read_bytes();body=original+b'\n\n'+('## 原始机器草案、独立几何与可执行反例\n\n此处逐字保留原始JSON；草案没有运行权限，最新实际训练V164，完整质量交付V159。当前V167入口仍待完整资格与物理封存。\n\n').encode('utf-8')
    for source in [DRAFT,GEOMETRY,BOUNDARIES]:body+=(f'来源 `{source.relative_to(ROOT).as_posix()}`；SHA256 `{sha(source)}`。\n\n```json\n'.encode('utf-8')+source.read_bytes()+b'\n```\n\n')
    body+=('五项实际反例回放已完成：`training/v166_observed_runtime_failure_replay_v2.py`，保存结果 `'+replay.relative_to(ROOT).as_posix()+'`；0官方调用。首版把有限目标检查误当已接受，因实际分类门槛优先返回拒绝而停止；原源码/日志/失败继续保存，v2按真实classification_guard拒绝回放，没有重跑候选或改门槛。\n\n## 发布前当前限制全部保留\n\n'+'\n'.join('- '+v for v in catalog['project']['known_limits'])+'\n\n## 既有证据限制完整保留\n\n'+retained).encode('utf-8');assert body.startswith(original) and all(p.read_bytes() in body for p in [DRAFT,GEOMETRY,BOUNDARIES]) and len(body)<=256*1024
    new=dict(id='v167-plan',title='V167方案',path=(OUT/'plan.md').relative_to(ROOT).as_posix(),sha256=hashlib.sha256(body).hexdigest(),category='review_evidence',keywords=['V167']);assert new['id'] not in historic;catalog['documents'].insert(0,new);project=catalog['project'];project['authoritative_direction_id']='v167-plan';project['current_summary']='V164三fit/5更新，分类未掌握；V166两过一拒，18784头/618导数，交付V159。';project['current_direction']=['V167草案，待资格封存。'];project['known_limits']=['目标未完成；全部旧限制见当前附录。']
    assert all(r==historic[r['id']] for r in catalog['documents'] if r['id'] in historic);encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024,f'Catalog limit before write: {len(encoded)}'
    OUT.mkdir()
    for path in mutable:
        target=OUT/'previous'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    (OUT/'plan.md').write_bytes(body)
    for source,name in [(DRAFT,'draft.json'),(GEOMETRY,'root_geometry.json'),(BOUNDARIES,'observed_runtime_boundaries.json'),(replay,'failure_replay.json')]:shutil.copyfile(source,OUT/name)
    lead='当前方向：[V166根实际结论与V167试探点修正方案](docs/V166_ROOT_ACTUAL_REVIEW_AND_V167_TRIAL_POINT_RESTORATION_PLAN_20261002.md)。V166已执行且两通过一拒绝；V167草案上限296头/100导数，入口资格与物理封存待完成，未调用官方模型。最新实际训练V164、完整质量交付V159均未完成总体目标。\n\n';old_readme=mutable[0].read_text(encoding='utf-8');obsolete='当前方向：[V165实际结果与覆盖优先后续方案]'
    lines=old_readme.splitlines();lines=[line for line in lines if not line.startswith(obsolete)];mutable[0].write_bytes((lead+'\n'.join(lines)+'\n').encode('utf-8'));mutable[1].write_bytes((mutable[1].read_text(encoding='utf-8')+'\n\n## V166独立几何及V167当前方案\n\n'+lead+'唯一当前direction=v167-plan，delivery=v159-delivery；原草案与V166实际记录均保留为历史。\n').encode('utf-8'));mutable[2].write_bytes(encoded);mutable[3].write_bytes(mutable[3].read_text(encoding='utf-8').replace('v166-review','v167-plan').encode('utf-8'))
    save(OUT/'publication.json',dict(status='V166_original_geometry_failure_constraints_and_V167_prospective_plan_published',catalog_bytes=len(encoded),historic_document_metadata_preserved=len(historic),authoritative_direction='v167-plan',authoritative_delivery='v159-delivery',root_plan_exact_byte_prefix=True,root_geometry_and_draft_exact_bytes_in_MCP_document=True,all_old_limits_retained=True,old_contradictory_current_direction_removed_from_README=True,official_calls=0,new_derivatives=0,fits=0,permanent_updates=0,V167_started=False,V167_execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),ROOTPLAN,DRAFT,GEOMETRY,source_binding,BOUNDARIES,replay,OUT/'plan.md',*mutable]}));print(json.dumps(dict(status='V167_prospective_direction_published',catalog_bytes=len(encoded),historical_records_preserved=len(historic),official_calls=0)))

if __name__=='__main__':main()
