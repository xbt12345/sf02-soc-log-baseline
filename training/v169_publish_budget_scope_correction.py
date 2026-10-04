"""Preserve the first direction and publish a source-backed budget scope correction."""
import argparse,json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

OUT=ROOT/'artifacts/v169_budget_scope_direction_20261002'
OLD=ROOT/'artifacts/v168_direction_20261002'
PLAN=ROOT/'docs/V168_INDEPENDENT_RESULTS_AND_NEXT_TRAINING_DECISION_20261002.md'
DRAFT=ROOT/'training/review_policy/v169_learnable_prior_pair_draft.json'
LOG=ROOT/'artifacts/v169_budget_scope_direction_MCP_tests_original_console_20261002.txt'
ID='v169-plan-v2'

def save(path,value):
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def publish():
    assert not OUT.exists()
    check_bindings(read(OLD/'validation.json')['source_sha256'])
    draft=read(DRAFT);check_bindings(draft['source_sha256'])
    assert not draft['execution_authority'] and not draft['new_fit_permission']
    v159=ROOT/'training/review_policy/v159_boundary_execution_contract_v4.json'
    v160=ROOT/'training/review_policy/v160_fixed_endpoint_diagnostic_contract.json'
    v164=ROOT/'training/review_policy/v164_short_trajectory_prospective_budget.json'
    p159,p160,p164=map(read,[v159,v160,v164])
    assert p159['total_cumulative_caps']['full_class_gradients']==2420
    assert p160['protocol']=='V160-three-fixed-B-endpoint-active-margin-finite-diagnostic-v1'
    assert p160['activation_entries']==['training/v160_fixed_endpoint_diagnostic_v3.py']
    assert p160['prior_technical_head_cap']==78226 and p160['new_caps']['heads']==3948
    assert p160['new_technical_head_cap']==78226+3948==82174
    assert p160['prior_technical_class_gradient_cap']==2420 and p160['new_caps']['class_gradients']==6
    assert p160['new_technical_class_gradient_cap']==2420+6==2426
    assert p160['new_margin_gradient_cap']==p160['new_caps']['margin_gradients']==144
    assert p160['new_caps']['fits']==p160['new_caps']['permanent_updates']==0
    assert p164['technical_complete_derivative_cap']==2426 and p164['new_caps']['fits']==3
    original_review=ROOT/'artifacts/v169_saved_prospective_budget_review_20261002/review.json'
    original=read(original_review);check_bindings(original['source_sha256'])
    assert original['naive_all120_points_total_derivatives']==12480
    actual=ROOT/'artifacts/v168_results_20261002/actual_result_summary.json'
    actual_cost=read(actual)
    assert actual_cost['cumulative_heads']==19178 and actual_cost['cumulative_all_complete_derivatives']==768
    files=[Path(__file__).resolve(),PLAN,DRAFT,v159,v160,v164,
      ROOT/'training/v160_seal_fixed_endpoint_diagnostic.py',ROOT/'training/v168_decision_floor_execution_review.py',
      ROOT/'docs/V159_FIXED_NUMERIC_POLICY_AND_TECHNICAL_BUDGET_CANDIDATE.md',
      ROOT/'docs/V160_FIXED_ENDPOINT_FINITE_DIAGNOSTIC_PLAN_20261002.md',ROOT/'docs/EXPERIMENT_REVIEW_RULES.md',
      original_review,actual,OLD/'plan.md',OLD/'publication.json',OLD/'validation.json']
    scope=dict(status='V169_original_budget_scope_checked_not_a_permanent_project_cap',
      original_V159_scope='six_fit_current_input_boundary_numeric_execution_contract_v4',
      V159_cumulative_complete_class_gradient_cap=2420,
      V160_scope=p160['protocol'],V160_allowed_entries=p160['activation_entries'],
      V160_head_cap_arithmetic='78226+3948=82174',
      V160_class_gradient_cap_arithmetic='2420+6=2426',
      V160_margin_gradient_cap_separately_registered=144,
      V164_later_explicit_all_derivative_cap=2426,
      original_2426_is_not_automatically_all_derivative_project_lifetime_cap=True,
      old_trial_limits_remain_binding_for_their_own_entries=True,
      previous_1658_remainder_is_only_conditional_on_reusing_old2426_ceiling=True,
      conditional_remainder=2426-768,conditional_naive_dense_derivatives=12480,
      V169_exact_callgraph_not_implemented=True,V169_new_budget_not_registered=True,
      no_new_budget_or_training_authority=True,no_beta_failure_or_effect_claim=True,
      historical_costs_not_reset=dict(heads=19178,complete_derivatives=768,fits_since_V159=9,updates_since_V159=170),
      still_required=['Actual new entry callgraph including bootstrap, terminal repeats, refusal and restoration',
        'Prospectively register bounded identical arm schedules and proportionate resource/cumulative cost bounds before execution',
        'B complete1060833 parameter identity and derivative/repeat policy',
        'Permanent full row protection ledger and current-point working-set derivatives',
        'Define bootstrap update counting within20 per fit and120 total',
        'Predeclare paired endpoint inconclusive after unmatched technical termination',
        'Qualification, independent root review and physical seal'],
      official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,
      execution_authority=False,new_fit_permission=False,
      source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    scope_bytes=(json.dumps(scope,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    correction=('\n\n## 预算适用范围纠正（当前解释）\n\n'
      '旧审查把82174/2426称为继承累计上限；这些数值不能自动成为V169或整个项目的永久上限。'
      '2426最初是V159完整类别梯度2420加V160诊断类别梯度6，V160另列144间隔导数；'
      'V164后来明确采用2426全部导数技术上限，但各原契约只约束其自身入口和注册范围。'
      '旧上限若继续沿用，剩余1658及朴素满程12480的条件算术成立；它不证明beta方法失败。'
      'V169须由实际新入口前瞻计算并登记相称的新预算，历史19178头/768导数、9fit/170更新保留。'
      '当前不扩大预算，不赋训练权，0新官方调用。精确调用图、参数身份、保护账本、初始化更新计数和不匹配终点处理仍待落实。\n\n'
      '下面完整保留旧审查与初版方向作为历史证据；其关于永久继承预算的解释以本节为准。\n\n```json\n').encode('utf-8')
    body=PLAN.read_bytes()+correction+scope_bytes+b'\n```\n\n'+('## 初版方向与原始审查完整保留\n\n').encode('utf-8')+(OLD/'plan.md').read_bytes()
    assert len(body)<=256*1024 and body.startswith(PLAN.read_bytes())
    mutables=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    catalog=read(mutables[2]);effective=load_catalog(ROOT,mutables[2],256*1024)
    history={r['id']:r for r in effective['documents']};assert len(history)==455 and effective['project']['authoritative_direction_id']=='v169-plan'
    for row in history.values():assert sha(ROOT/row['path'])==row['sha256']
    assert ID not in history
    import hashlib
    catalog['documents'].insert(0,dict(id=ID,title='V169配对先验学习草案与预算范围纠正',path=(OUT/'plan.md').relative_to(ROOT).as_posix(),sha256=hashlib.sha256(body).hexdigest(),category='review_evidence',keywords=['V169','预算范围']))
    project=catalog['project'];project['authoritative_direction_id']=ID
    project['current_summary']='V169配对先验学习草案已发布；旧82174/2426是注册试验范围预算，不自动成为永久项目上限。新调用图/预算未实现登记，0新官方调用/fit。历史19178头/768导数保留，最新训练V164、完整交付V159，分类未掌握，三目标未完成。'
    project['current_direction']=['只处理训练侧分类：A固定先验，候选B共享可学习系数；共同修完整保护账本与动态工作集。实际新入口须前瞻计算和登记相称预算、资格、根审查及封存；当前无新模型调用或拟合权限。']
    project['known_limits']=['V169原始草案未改；旧预算审查1658仅是沿用旧2426上限的条件余量，不能认作V169已注册预算或beta失败。','精确调用图未实现，B完整参数1060833、保护账本、20/120更新计数和技术停止匹配终点仍待落实。','最新训练V164、完整交付V159；三个目标及完整质量未完成。全部历史限制及原审查完整保留于当前方向附录。']
    encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024
    OUT.mkdir()
    for p in mutables:
        target=OUT/'previous'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    (OUT/'scope_review.json').write_bytes(scope_bytes);(OUT/'plan.md').write_bytes(body)
    lead='当前方向：[V169草案与原预算适用范围纠正](artifacts/v169_budget_scope_direction_20261002/plan.md)。旧82174/2426不自动成为项目永久上限；V169精确调用图和新预算仍未实现登记。0新官方调用/fit，历史成本保留，最新训练V164、完整交付V159，三目标未完成。\n\n'
    mutables[0].write_bytes((lead+mutables[0].read_text(encoding='utf-8')).encode('utf-8'))
    mutables[1].write_bytes((mutables[1].read_text(encoding='utf-8')+'\n\n## V169预算范围纠正\n\n'+lead+'当前direction=v169-plan-v2，初版v169-plan及全部旧记录保持原元数据和哈希。新预算须按实际新入口前瞻登记，不能将旧诊断额度自动继承为永久上限或自动授权。\n').encode('utf-8'))
    mutables[2].write_bytes(encoded)
    mutables[3].write_bytes(mutables[3].read_text(encoding='utf-8').replace('v169-plan','v169-plan-v2').encode('utf-8'))
    now={r['id']:r for r in load_catalog(ROOT,mutables[2],256*1024)['documents']}
    assert len(now)==456 and all(now[k]==v for k,v in history.items())
    save(OUT/'publication.json',dict(status='V169_budget_scope_correction_published_with_original_direction_preserved',
      current_direction=ID,authoritative_delivery='v159-delivery',historic_records_preserved=455,effective_documents=456,
      official_calls=0,new_fits=0,permanent_updates=0,execution_authority=False,
      source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [*files,*mutables,OUT/'plan.md',OUT/'scope_review.json']}))
    print(json.dumps(dict(status='V169_budget_scope_corrected_and_published',historic_records_preserved=455,effective_documents=456,official_calls=0)))

def verify():
    target=OUT/'validation.json';assert not target.exists()
    publication=read(OUT/'publication.json');check_bindings(publication['source_sha256'])
    scope=read(OUT/'scope_review.json');check_bindings(scope['source_sha256'])
    current=load_catalog(ROOT,ROOT/'mcp_readonly/catalog.json',256*1024)
    previous=load_catalog(ROOT,OUT/'previous/mcp_readonly/catalog.json',256*1024)
    old={r['id']:r for r in previous['documents']};now={r['id']:r for r in current['documents']}
    assert len(old)==455 and len(now)==456 and all(now[k]==v for k,v in old.items())
    for row in now.values():assert sha(ROOT/row['path'])==row['sha256'] and (ROOT/row['path']).stat().st_size<=256*1024
    assert current['project']['authoritative_direction_id']==ID and current['project']['authoritative_delivery_id']=='v159-delivery'
    body=(OUT/'plan.md').read_bytes()
    assert body.startswith(PLAN.read_bytes()) and (OLD/'plan.md').read_bytes() in body and (OUT/'scope_review.json').read_bytes() in body
    assert not read(DRAFT)['execution_authority'] and not read(DRAFT)['new_fit_permission']
    text=LOG.read_text(encoding='utf-8');assert 'Ran 23 tests' in text and text.rstrip().endswith('OK')
    paths=[Path(__file__).resolve(),OUT/'publication.json',OUT/'scope_review.json',OUT/'plan.md',LOG,ROOT/'mcp_readonly/catalog.json',ROOT/'mcp_readonly/tests/test_readonly_mcp.py',ROOT/'README.md',ROOT/'HANDOFF.md']
    save(target,dict(status='V169_budget_scope_current_direction_455_old_records_and23_local_MCP_tests_verified',
      current_direction=ID,effective_documents=456,historic_records_preserved=455,MCP_tests_passed=23,
      root_plan_and_draft_unchanged=True,old_budget_not_automatically_project_lifetime_cap=True,
      V169_callgraph_and_new_budget_still_unregistered=True,latest_actual_training='V164',latest_complete_delivery='V159',
      official_calls=0,fits=0,permanent_updates=0,execution_authority=False,quality_acceptance=False,classification_mastery=False,
      source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
    print(json.dumps(dict(status='V169_budget_scope_direction_verified',historic_records_preserved=455,MCP_tests=23,official_calls=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['publish','verify']);args=parser.parse_args()
    (publish if args.mode=='publish' else verify)()
