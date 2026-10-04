"""Publish the root's prospective one-role plan, never activation authority."""
import hashlib,json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

OUT=ROOT/'artifacts/v167_direction_20261002'
PLAN=ROOT/'docs/V167_ROOT_RESULTS_AND_V168_DECISION_AWARE_FLOOR_PLAN_20261002.md'
DRAFT=ROOT/'training/review_policy/v168_decision_aware_floor_draft.json'
REGISTRY=ROOT/'training/review_policy/v167_observed_runtime_boundaries.json'
REPLAY=ROOT/'artifacts/v167_observed_runtime_failure_replay_20261002/replay.json'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists();draft=read(DRAFT);replay=read(REPLAY)
    check_bindings(draft['source_sha256']);check_bindings(replay['source_sha256'])
    previous_validation=ROOT/'artifacts/v167_results_20261002/validation.json';check_bindings(read(previous_validation)['source_sha256'])
    assert not draft['execution_authority'] and draft['execute_only_actual_failed_role']==1 and draft['complete_functions']==25
    assert draft['proposed_caps']['heads']==98 and draft['proposed_caps']['margin_gradients']==50 and draft['proposed_caps']['QP_solves']==draft['proposed_caps']['finite_proposals']==1 and draft['proposed_caps']['fits']==draft['proposed_caps']['permanent_updates']==0
    assert draft['future_cumulative_caps']['heads']==19178 and draft['future_cumulative_caps']['all_complete_parameter_derivatives']==768
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    catalog=read(mutable[2]);effective=load_catalog(ROOT,mutable[2],256*1024);historic={r['id']:r for r in effective['documents']};assert len(historic)==450 and effective['project']['authoritative_direction_id']=='v167-review'
    for r in historic.values():assert sha(ROOT/r['path'])==r['sha256']
    old=ROOT/'artifacts/v167_results_20261002/review.md';retained=old.read_text(encoding='utf-8').split('## 既有证据限制完整保留\n\n',1)[1];original=PLAN.read_bytes();body=original+b'\n\n'+('## 原始草案及实际失败约束\n\nV167已经实际执行并拒绝角色1；V168只为草案，入口资格、物理封存仍待完成，尚无新官方调用或拟合权限。\n\n').encode('utf-8')
    for p in [DRAFT,REGISTRY]:body+=(f'来源 `{p.relative_to(ROOT).as_posix()}`；SHA256 `{sha(p)}`。\n\n```json\n'.encode('utf-8')+p.read_bytes()+b'\n```\n\n')
    body+=('六项V167实际失败约束均已回放，0新官方调用：`'+REPLAY.relative_to(ROOT).as_posix()+'`。两条完全平手及两次硬停止保留；不得追认成功或延续第三修正。V167完整发布已验证450记录（447旧记录逐项保留）及23个本地MCP测试；首次状态文案断言失败日志与单独修复记录保留，不改变模型或验收门槛。\n\n## 发布前当前限制全部保留\n\n'+'\n'.join('- '+v for v in effective['project']['known_limits'])+'\n\n## 既有证据限制完整保留\n\n'+retained).encode('utf-8')
    assert body.startswith(original) and all(p.read_bytes() in body for p in [DRAFT,REGISTRY]) and len(body)<=256*1024
    new=dict(id='v168-plan',title='V168判决地板草案',path=(OUT/'plan.md').relative_to(ROOT).as_posix(),sha256=hashlib.sha256(body).hexdigest(),category='review_evidence',keywords=['V168']);assert new['id'] not in historic;catalog['documents'].insert(0,new)
    p=catalog['project'];p['authoritative_direction_id']='v168-plan';p['current_summary']='V164三fit/5更新，分类未掌握；V167两次仍新增2S平手错误，累计19080头/718导数。V168仅草案待资格封存，三目标未完成，交付V159。';p['current_direction']=['V168只对失败角色1做一次判决地板诊断；先资格及物理封存，无新官方调用/拟合权限。'];p['known_limits']=['三个目标未完成；全部旧限制与失败成本见当前附录。']
    encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024
    OUT.mkdir()
    for path in mutable:
        target=OUT/'previous'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    (OUT/'plan.md').write_bytes(body)
    for p,name in [(DRAFT,'draft.json'),(REGISTRY,'observed_runtime_boundaries.json'),(REPLAY,'failure_replay.json')]:shutil.copyfile(p,OUT/name)
    lead='当前方向：[V167实际结论与V168判决地板草案](docs/V167_ROOT_RESULTS_AND_V168_DECISION_AWARE_FLOOR_PLAN_20261002.md)。V167两次修正仍新增2条S平手错误，全恢复V164；V168仅角色1一次诊断，上限98头/50导数/1QP、0fit/更新，资格及物理封存尚待完成，尚无新官方调用权限。三个目标未完成；最新训练V164，完整质量交付V159。\n\n'
    mutable[0].write_bytes((lead+mutable[0].read_text(encoding='utf-8')).encode('utf-8'));mutable[1].write_bytes((mutable[1].read_text(encoding='utf-8')+'\n\n## V167根结论与V168当前草案\n\n'+lead+'唯一当前direction=v168-plan，delivery=v159-delivery；V167原实际报告/资格/封存/失败日志全部保留。\n').encode('utf-8'));mutable[2].write_bytes(encoded);mutable[3].write_bytes(mutable[3].read_text(encoding='utf-8').replace('v167-review','v168-plan').encode('utf-8'))
    after=load_catalog(ROOT,mutable[2],256*1024);now={r['id']:r for r in after['documents']};assert len(now)==451 and all(now[k]==v for k,v in historic.items())
    save(OUT/'publication.json',dict(status='V167_actual_result_and_V168_single_role_prospective_direction_published',catalog_bytes=len(encoded),historic_document_metadata_preserved=450,effective_documents=451,authoritative_direction='v168-plan',authoritative_delivery='v159-delivery',root_plan_exact_byte_prefix=True,draft_and_actual_failure_registry_exact_bytes_in_document=True,all_old_limits_retained=True,official_calls=0,new_derivatives=0,fits=0,permanent_updates=0,V168_started=False,V168_execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),PLAN,DRAFT,REGISTRY,REPLAY,previous_validation,OUT/'plan.md',*mutable]}));print(json.dumps(dict(status='V168_prospective_direction_published',historic_records_preserved=450,official_calls=0)))

if __name__=='__main__':main()
