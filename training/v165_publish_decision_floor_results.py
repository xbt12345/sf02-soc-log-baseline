"""Publish real finite failures, retained V164 training and unchanged delivery."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v165_results_20261002'
REPORT=ROOT/'docs/V165_COMPLETE_DECISION_FLOOR_DIAGNOSTIC_RESULTS_20261002.md'
TRIAL=ROOT/'artifacts/v165_fixed_endpoint_decision_floor_diagnostic_20261002'
ROOTPLAN=ROOT/'docs/V164_RESULTS_AND_V165_DECISION_FLOOR_DIAGNOSTIC_PLAN_20261002.md'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists() and not REPORT.exists()
    own=ROOT/'artifacts/v165_saved_decision_floor_quality_review_v2_20261002/review.json';independent=ROOT/'artifacts/v165_independent_actual_decision_floor_review_20261002/review.json';ib=independent.parent/'pre_review_bindings.json'
    a,b=read(own),read(independent);check_bindings(a['source_sha256']);check_bindings(read(ib)['source_sha256'])
    assert b['status']=='all_three_actual_decision_floor_finite_probes_original_gold_guards_costs_and_full_restoration_verified'
    assert not a['all_actual_finite_guards_passed'] and not b['all_three_actual_finite_guards_passed'] and not b['supports_new_short_training_registration']
    diagnostics=[read(TRIAL/f'role{r}/diagnostic.json') for r in range(3)];assert all(d['exception'] is None and d['all_parameters_restored'] and not d['candidate']['accepted'] for d in diagnostics)
    assert sum(d['counts']['head_attempts'] for d in diagnostics)==b['new_heads']==180 and b['new_complete_derivatives']==b['new_fits']==b['permanent_updates']==0
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']];catalog=read(mutable[2]);historic={r['id']:dict(r) for r in catalog['documents']}
    assert catalog['project']['authoritative_delivery_id']=='v159-delivery'
    for row in catalog['documents']:assert sha(ROOT/row['path'])==row['sha256']
    previous=ROOT/'artifacts/v164_results_20261002/review.md';retained=previous.read_text(encoding='utf-8').split('## 既有证据限制完整保留\n\n',1)[1]
    lines=['# V165 判决边界保护：三个真实有限诊断结果','',
           '三个处理候选均因新增 M 错误被原保护门槛拒绝。改为只恢复正确判决边界确实减小了局部修正，但仍未保住原正确类别，因此该单因素机制尚不能进入训练。所有模型均恢复 V164 最后真实接受参数；本轮0新增拟合、0永久更新，最近真实训练仍 V164，最新完整三分类质量交付仍为未通过的 V159。','',
           '起点为角色0/1的 V164 新参数点1及角色2的点3；控制是相同参数点已保存的原置信间隔恢复0，处理是根端同 u/A/c/gM/gS、仅 b=0 的保存完整位移。真实步长1、原输入/频次/固定错误目标/完整原类别分母、原16eps/Armijo、逐类计数、累计新修复保护与旧能力检查均未改变。没有新增法向、QP求解或函数容量，24上限未改。','',
           '|角色|起点纯错误|原恢复控制纯错误|零地板处理纯错误|处理修复 M/S|处理新增错误 M/S|实际接受|',
           '|---|---:|---:|---:|---|---|---|']
    for item in a['roles']:
        role=item['role'];paired=item['paired_treatment_vs_V164'];control=item['paired_control_vs_V164'];repairs='/'.join(str(paired[c]['repairs_vs_previous_accepted']) for c in ['M','S']);new='/'.join(str(paired[c]['new_errors_vs_previous_accepted']) for c in ['M','S'])
        lines.append(f"|{role}|{item['pure_errors_before']}|{sum(v['pure_errors'] for v in control.values())}|{item['pure_errors_after']}|{repairs}|{new}|否|")
    lines+=['','处理相对原恢复控制的分类错误有减少，但三个候选仍分别新增76、32、14条 M 原行错误；候选中的收益未提交，不把它们当作永久修复。原置信间隔恢复并非充分根因：释放该额外目标仍不安全。下一项须依根计划研究真实阻挡覆盖、约束工作集与非线性修正，不能用损失下降、较小修正范数或原控制更差来覆盖拒绝。','',
            '全部原行均计分，包括最初未纳入固定纯正确保护的206条混合正确行；全部已经真实修复的原行仍累计保护。相对 V164 和相对原 V159 固定终点的修复/退化分列，角色1原有8条修复没有重复计为新收益。来源root仅为代理，候选的来源分布和读出必要界限不构成细行为或独立来源验收。','',
            '每个角色实际执行起点全重放、一个处理试探、finally恢复后全重放：66/48/66，共180次头与180次特征、3实际候选、0新目标/原类别/间隔导数、0拟合/永久更新、0正式QP。资格中的12次CPU保存向量回放单列，未记为官方梯度或训练。累计18538次头/特征、368原类别+32固定错误+152间隔=552完整参数导数；自V159以来9拟合、170永久更新均不变。','',
            '正式执行前完成两个新资格回放，物理封存10055个来源，合同 `training/review_policy/v165_fixed_endpoint_decision_floor_diagnostic_contract.json` 与入口 `training/v165_fixed_endpoint_decision_floor_diagnostic.py`。初始与恢复 q/logq 按既定8eps检查且argmax完全一致；保存的完整 endpoint.pt 张量与 V164 原终点逐位一致。拒绝候选的完整位移、临时参数身份、概率、原行、阻挡与真实调用全部保留。','',
            '独立原gold/有限条件/账本/完整恢复审查 `artifacts/v165_independent_actual_decision_floor_review_20261002/review.json`，逐类CE/来源集中度/未修复间隔/当前读出界及控制处理自审 `artifacts/v165_saved_decision_floor_quality_review_v2_20261002/review.json`，均实际完成且审查0官方调用。自审v1误用了未登记的逐位概率相等要求而停止；保留原失败和源码，v2使用原8eps/严格argmax协议，未重跑诊断、修改候选或放宽分类门槛。','',
            '当前具体计划为 `docs/V164_RESULTS_AND_V165_DECISION_FLOOR_DIAGNOSTIC_PLAN_20261002.md` 第6节的失败分支；实际反例约束 `training/review_policy/v164_root_failure_constraints.json` 全部绑定。下一项新容量或拟合须前瞻登记；本轮结果没有任何追加拟合或模型晋升权限。训练侧分类掌握、细行为同类支持、跨来源稳定性，以及完整2056871行三分类和独立来源检验仍未完成。','',
            '## 既有证据限制完整保留','',retained]
    body='\n'.join(lines)+'\n';OUT.mkdir()
    for path in mutable:
        target=OUT/'previous'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    REPORT.write_bytes(body.encode('utf-8'));snapshot=OUT/'review.md';shutil.copyfile(REPORT,snapshot)
    plan_snapshot=OUT/'root_plan.md';shutil.copyfile(ROOTPLAN,plan_snapshot)
    catalog['documents'].insert(0,dict(id='v165-plan',title='V165根具体计划',path=plan_snapshot.relative_to(ROOT).as_posix(),sha256=sha(plan_snapshot),category='review_evidence',keywords=['V165','计划']))
    catalog['documents'].insert(0,dict(id='v165-review',title='V165有限诊断失败',path=snapshot.relative_to(ROOT).as_posix(),sha256=sha(snapshot),category='review_evidence',keywords=['V165']))
    project=catalog['project'];project['authoritative_direction_id']='v165-review';project['current_summary']='V164三fit/5更新，分类未掌握。V165三候选不安全，0fit/更新；累计18538头/552导数。完整交付V159。';project['current_direction']=['按V165根计划失败分支研究阻挡覆盖/工作集，另登记；不得续训。'];project['known_limits'][0]='V164训练分类未掌握，V165三候选新增M错误均拒绝；未晋升模型。';project['known_limits'][5]='V165已恢复V164完整张量，累计8条原修复保留；处理未提交。'
    for row in catalog['documents']:
        if row['id'] in historic:assert row==historic[row['id']]
    encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024
    summary=dict(status='all_three_real_decision_floor_finite_candidates_rejected_and_V164_endpoints_restored',roles=a['roles'],actual_new_heads=180,actual_new_features=180,actual_new_complete_derivatives=0,actual_finite_proposals=3,new_fits=0,permanent_updates=0,actual_new_official_QP_solves=0,preparation_CPU_QP_solves=12,cumulative_heads=18538,cumulative_features=18538,cumulative_full_original_class_gradients=368,cumulative_fixed_error_target_gradients=32,cumulative_margin_gradients=152,cumulative_all_complete_derivatives=552,cumulative_fits_since_V159=9,cumulative_updates_since_V159=170,all_actual_finite_guards_passed=False,supports_new_short_training_registration=False,all_restored_full_tensors_exact=True,all_restored_original_argmax_exact=True,restored_probabilities_use_original_8eps_policy=True,latest_actual_training='V164',latest_complete_quality_delivery='V159',quality_acceptance=False,root_goal_status='active',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),REPORT,own,independent,ib,ROOTPLAN,previous,ROOT/'training/review_policy/v164_root_failure_constraints.json']});save(OUT/'actual_result_summary.json',summary);mutable[2].write_bytes(encoded)
    lead='最新诊断：[V165三候选判决边界实际失败](docs/V165_COMPLETE_DECISION_FLOOR_DIAGNOSTIC_RESULTS_20261002.md)。新增M错76/32/14均拒绝，完整恢复V164终点；180头/特征、0fit/更新/导数，累计18538头/552导数。最新训练V164仍未掌握，完整质量交付V159。当前根计划：[V165有限对照与失败分支](docs/V164_RESULTS_AND_V165_DECISION_FLOOR_DIAGNOSTIC_PLAN_20261002.md)。\n\n'
    mutable[0].write_bytes((lead+mutable[0].read_text(encoding='utf-8')).encode('utf-8'));mutable[1].write_bytes((mutable[1].read_text(encoding='utf-8')+'\n\n## V165三候选实际不安全、全恢复\n\n'+lead+'delivery=v159-delivery，direction=v165-review；当前根计划v165-plan。0新fit诊断不替换V164最新实际训练，不授予新训练。\n').encode('utf-8'))
    mutable[3].write_bytes(mutable[3].read_text(encoding='utf-8').replace('v164-review','v165-review').encode('utf-8'))
    save(OUT/'publication.json',dict(status=summary['status'],catalog_bytes=len(encoded),historic_document_metadata_preserved=len(historic),authoritative_delivery='v159-delivery',authoritative_direction='v165-review',current_root_plan='v165-plan',official_calls_by_publication=0,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),*mutable]}));print(json.dumps(dict(status='V165_actual_decision_floor_failures_published',catalog_bytes=len(encoded),historic_records_preserved=len(historic))))

if __name__=='__main__':main()
