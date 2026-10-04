"""Publish the final root plan and original reviews; no execution authority."""
import hashlib,json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v165_final_direction_20261002'
ROOTPLAN=ROOT/'docs/V165_ACTUAL_RESULTS_AND_COVERAGE_FIRST_NEXT_PLAN_20261002.md'
DRAFT=ROOT/'training/review_policy/v166_coverage_first_diagnostic_draft.json'
BOUNDARIES=ROOT/'training/review_policy/v165_observed_boundaries.json'
ACTUAL=ROOT/'artifacts/v165_independent_actual_decision_floor_review_20261002/review.json'
COVERAGE=ROOT/'artifacts/v165_independent_blocker_coverage_review_20261002/review.json'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists();draft=read(DRAFT);boundary=read(BOUNDARIES);actual=read(ACTUAL);coverage=read(COVERAGE)
    assert draft['execution_authority'] is False and draft['new_fitting_permission'] is False and draft['proposed_caps']['heads']==246 and draft['proposed_caps']['margin_gradients']==66
    assert draft['prior_actual_costs']['heads']==18538 and draft['prior_actual_costs']['all_complete_parameter_derivatives']==552
    assert actual['new_heads']==180 and actual['new_fits']==actual['permanent_updates']==0 and not actual['supports_new_short_training_registration']
    assert [r['fresh_unmeasured_functions'] for r in coverage['roles']]==[20,9,4] and [r['current_union_if_fresh_functions_added'] for r in coverage['roles']]==[24,25,14]
    assert boundary['no_new_execution_authority'] is True
    check_bindings(read(ACTUAL.parent/'pre_review_bindings.json')['source_sha256']);check_bindings(read(COVERAGE.parent/'source_bindings.json')['source_sha256'])
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']];catalog=read(mutable[2]);historic={r['id']:dict(r) for r in catalog['documents']}
    assert catalog['project']['authoritative_delivery_id']=='v159-delivery' and catalog['project']['authoritative_direction_id']=='v165-review'
    for row in catalog['documents']:assert sha(ROOT/row['path'])==row['sha256']
    previous=ROOT/'artifacts/v165_results_20261002/review.md';retained=previous.read_text(encoding='utf-8').split('## 既有证据限制完整保留\n\n',1)[1]
    original=ROOTPLAN.read_bytes();body=original+b'\n\n'+('## 机器草案与新增反例：均无执行权限\n\n'+f'原始方案SHA256：{sha(ROOTPLAN)}。以下只保存草案和已观察反例，不增加官方调用或拟合许可。\n\n'+f'草案 `{DRAFT.relative_to(ROOT).as_posix()}`：\n\n```json\n'+DRAFT.read_text(encoding='utf-8')+'\n```\n\n'+f'反例 `{BOUNDARIES.relative_to(ROOT).as_posix()}`：\n\n```json\n'+BOUNDARIES.read_text(encoding='utf-8')+'\n```\n\n'+'## 当前旧限制逐项保留\n\n'+'\n'.join('- '+v for v in catalog['project']['known_limits'])+'\n\n## 既有证据限制完整保留\n\n'+retained).encode('utf-8')
    assert body.startswith(original) and len(body)<=256*1024
    plan_snapshot=OUT/'plan.md'
    additions=[dict(id='v166-plan',title='V166覆盖草案',path=plan_snapshot.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(body).hexdigest(),category='review_evidence',keywords=['V166']),dict(id='v165-root-review',title='V165根实际审查',path=ACTUAL.relative_to(ROOT).as_posix(),sha256=sha(ACTUAL),category='review_evidence',keywords=['V165']),dict(id='v165-coverage',title='V165函数覆盖',path=COVERAGE.relative_to(ROOT).as_posix(),sha256=sha(COVERAGE),category='review_evidence',keywords=['V165'])]
    for row in additions:assert row['id'] not in historic
    catalog['documents']=additions+catalog['documents'];project=catalog['project'];project['authoritative_direction_id']='v166-plan';project['current_summary']='V164三fit/5更新，分类未掌握。V165三候选拒绝；累计18538头/552导数。下一覆盖草案未授权，完整交付V159。';project['current_direction']=['V166草案：补测33函数，统一25；246头/66导数，尚无执行权限。'];project['known_limits'][-1]='旧限制完整保存在方向附录；历史元数据不变。'
    for row in catalog['documents']:
        if row['id'] in historic:assert row==historic[row['id']]
    encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024,f'Catalog capacity exceeded before any write: {len(encoded)}'
    OUT.mkdir()
    for path in mutable:
        target=OUT/'previous'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    plan_snapshot.write_bytes(body);shutil.copyfile(DRAFT,OUT/'draft.json');shutil.copyfile(BOUNDARIES,OUT/'observed_boundaries.json');mutable[2].write_bytes(encoded)
    lead='当前方向：[V165实际结果与覆盖优先后续方案](docs/V165_ACTUAL_RESULTS_AND_COVERAGE_FIRST_NEXT_PLAN_20261002.md)。33个已暴露未覆盖函数，拟统一25上限；246头/66导数仅为未授权草案，未启动V166。最新真实训练V164，最新完整质量交付V159；V165180调用诊断全部拒绝并恢复。\n\n'
    mutable[0].write_bytes((lead+mutable[0].read_text(encoding='utf-8')).encode('utf-8'));mutable[1].write_bytes((mutable[1].read_text(encoding='utf-8')+'\n\n## V165最终当前方向与原始独立证据\n\n'+lead+'direction=v166-plan，delivery=v159-delivery；根原始审查v165-root-review、完整函数覆盖v165-coverage。旧方向与全部历史元数据保留，新草案没有执行或拟合权限。\n').encode('utf-8'))
    mutable[3].write_bytes(mutable[3].read_text(encoding='utf-8').replace('v165-review','v166-plan').encode('utf-8'))
    save(OUT/'publication.json',dict(status='V165_final_root_direction_and_two_original_reviews_published',authoritative_direction='v166-plan',authoritative_delivery='v159-delivery',latest_actual_training='V164',latest_complete_quality_delivery='V159',catalog_bytes=len(encoded),historic_document_metadata_preserved=len(historic),new_document_ids=[r['id'] for r in additions],new_official_calls=0,new_derivatives=0,new_fits=0,permanent_updates=0,V166_started=False,V166_execution_authority=False,root_plan_bytes_preserved_as_exact_prefix=True,previous_known_limits_all_preserved=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOTPLAN,DRAFT,BOUNDARIES,ACTUAL,COVERAGE,plan_snapshot,*mutable]}));print(json.dumps(dict(status='V165_final_direction_published',catalog_bytes=len(encoded),historic_records_preserved=len(historic),new_official_calls=0,V166_started=False)))

if __name__=='__main__':main()
