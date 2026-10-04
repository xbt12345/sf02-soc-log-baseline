"""Exact root plan, machine draft and observed trajectory/budget, no new training."""
import hashlib,json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

OUT=ROOT/'artifacts/v168_direction_20261002'
PLAN=ROOT/'docs/V168_INDEPENDENT_RESULTS_AND_NEXT_TRAINING_DECISION_20261002.md'
DRAFT=ROOT/'training/review_policy/v169_learnable_prior_pair_draft.json'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists();assert sha(PLAN)=='6cb91a9b340f1b530ba14c224a4deb00de6769c06138da51e5c2b05d5c072f73';draft=read(DRAFT);check_bindings(draft['source_sha256']);assert not draft['execution_authority'] and not draft['new_fit_permission'] and draft['candidate']=='B' and draft['proposed_fit_cap']==6 and draft['proposed_permanent_update_cap']==120
    registry=ROOT/'training/review_policy/v168_observed_runtime_boundaries.json';replay=ROOT/'artifacts/v168_observed_runtime_boundaries_20261002/replay.json';trajectory=ROOT/'artifacts/v168_saved_accepted_learning_trajectory_review_20261002/review.json';budget=ROOT/'artifacts/v169_saved_prospective_budget_review_20261002/review.json';previous_validation=ROOT/'artifacts/v168_results_20261002/validation.json'
    for p in [replay,trajectory,budget,previous_validation]:check_bindings(read(p)['source_sha256'])
    assert len(read(replay)['cases'])==5 and read(trajectory)['role1_frozen_deep_S_error_rows_still_wrong']==813 and read(budget)['remaining_complete_derivatives']==1658
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']];catalog=read(mutable[2]);effective=load_catalog(ROOT,mutable[2],256*1024);historic={r['id']:r for r in effective['documents']};assert len(historic)==454 and effective['project']['authoritative_direction_id']=='v168-review' and effective['project']['authoritative_delivery_id']=='v159-delivery'
    for row in historic.values():assert sha(ROOT/row['path'])==row['sha256']
    original=PLAN.read_bytes();prior=ROOT/'artifacts/v168_results_20261002/review.md';retained=prior.read_text(encoding='utf-8').split('## 既有证据限制完整保留\n\n',1)[1];body=original+b'\n\n'+('## 原始草案、实际边界与预算核查\n\nV168已完成真实诊断及独立审查，但全部恢复V164，0fit/永久更新。V169只为下一训练设计，精确调用图、参数维度、工作集资格及根审查/物理封存尚未完成，没有新前向、梯度或拟合权限。\n\n').encode('utf-8')
    for p in [DRAFT,registry,trajectory,budget]:body+=(f'来源 `{p.relative_to(ROOT).as_posix()}`；SHA256 `{sha(p)}`。\n\n```json\n'.encode('utf-8')+p.read_bytes()+b'\n```\n\n')
    body+=('五项V168实际边界回放：`'+replay.relative_to(ROOT).as_posix()+'`。预算核查为条件性算术，不是证明方法不可行；必须先明确安全初始化提交的更新计数、B新增beta的完整梯度/重复策略、逐候选完整保护以及配对技术中止的解释，不能复用旧固定宽度检查遗漏新参数。\n\n## 发布前当前限制全部保留\n\n'+'\n'.join('- '+v for v in effective['project']['known_limits'])+'\n\n## 既有证据限制完整保留\n\n'+retained).encode('utf-8');assert body.startswith(original) and all(p.read_bytes() in body for p in [DRAFT,registry,trajectory,budget]) and len(body)<=256*1024
    new=dict(id='v169-plan',title='V169配对先验学习草案',path=(OUT/'plan.md').relative_to(ROOT).as_posix(),sha256=hashlib.sha256(body).hexdigest(),category='review_evidence',keywords=['V169']);assert new['id'] not in historic;catalog['documents'].insert(0,new);p=catalog['project'];p['authoritative_direction_id']='v169-plan';p['current_summary']='V168单次诊断通过，全恢复V164；累计19178头/768导数，分类未掌握，三目标未完成。V169仅配对先验学习草案，0新fit/调用，精确预算与资格待完成；交付V159。';p['current_direction']=['只处理训练侧分类：A固定先验，B共享可学习系数；共同修完整保护与工作集管理。先精确预算/资格/根审查/封存，无新模型调用或拟合权限。'];p['known_limits']=['三个目标及完整质量未完成；原训练V164、完整交付V159；所有旧限制和条件性预算核查见当前附录。'];encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024
    OUT.mkdir()
    for path in mutable:
        target=OUT/'previous'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    (OUT/'plan.md').write_bytes(body)
    for p,name in [(DRAFT,'draft.json'),(registry,'observed_runtime_boundaries.json'),(replay,'failure_replay.json'),(trajectory,'accepted_trajectory.json'),(budget,'prospective_budget_review.json')]:shutil.copyfile(p,OUT/name)
    lead='当前方向：[V168独立结果与V169配对训练设计](docs/V168_INDEPENDENT_RESULTS_AND_NEXT_TRAINING_DECISION_20261002.md)。唯一学习变化为B新增共享先验系数beta，两臂共同修完整保护账本与动态工作集；当前0新调用/fit，精确预算、资格、根审查与封存尚待完成。813条角色1低先验S错仍未翻类。分类未掌握，三个目标未完成，最新训练V164、完整质量交付V159。\n\n'
    mutable[0].write_bytes((lead+mutable[0].read_text(encoding='utf-8')).encode('utf-8'));mutable[1].write_bytes((mutable[1].read_text(encoding='utf-8')+'\n\n## V169当前配对学习设计\n\n'+lead+'唯一direction=v169-plan，delivery=v159-delivery；新设计最多6fit/120接受更新不是执行授权。原V168证据/费用/失败源码完整保留，预算核查及真实已接受轨迹已保存并嵌入当前MCP方向；不得改旧封存源码或自动追加拟合。\n').encode('utf-8'));mutable[2].write_bytes(encoded);mutable[3].write_bytes(mutable[3].read_text(encoding='utf-8').replace('v168-review','v169-plan').encode('utf-8'));after=load_catalog(ROOT,mutable[2],256*1024);now={r['id']:r for r in after['documents']};assert len(now)==455 and all(now[k]==v for k,v in historic.items())
    sources=[Path(__file__).resolve(),PLAN,DRAFT,registry,replay,trajectory,budget,previous_validation,prior,OUT/'plan.md',*mutable];save(OUT/'publication.json',dict(status='V168_actual_evidence_and_V169_prospective_learning_direction_published',historic_document_metadata_preserved=454,effective_documents=455,catalog_bytes=len(encoded),document_bytes=len(body),authoritative_direction='v169-plan',authoritative_delivery='v159-delivery',root_plan_exact_byte_prefix=True,all_original_json_exact_bytes_embedded=True,all_old_limits_retained=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,V169_execution_authority=False,source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in sources}));print(json.dumps(dict(status='V169_prospective_direction_published',historic_records_preserved=454,effective_documents=455,official_calls=0)))

if __name__=='__main__':main()
