"""Publish complete coverage results without changing training or delivery."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v166_results_20261002'
REPORT=ROOT/'docs/V166_COMPLETE_COVERAGE_DIAGNOSTIC_RESULTS_20261002.md'
TRIAL=ROOT/'artifacts/v166_coverage_first_diagnostic_20261002'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists() and not REPORT.exists()
    own=ROOT/'artifacts/v166_saved_coverage_quality_review_20261002/review.json';independent=ROOT/'artifacts/v166_independent_actual_coverage_review_20261002/review.json';ib=independent.parent/'pre_review_bindings.json';blockers=ROOT/'artifacts/v166_saved_residual_blocker_identity_review_20261002/review.json';a,b,c=read(own),read(independent),read(blockers)
    check_bindings(a['source_sha256']);check_bindings(read(ib)['source_sha256']);check_bindings(c['source_sha256']);assert b['status']=='all_three_actual_coverage_math_fresh_gradients_original_gold_costs_and_restoration_verified';assert not a['all_actual_finite_guards_passed'] and not b['all_three_actual_finite_guards_passed'] and not b['supports_new_short_training_registration'];assert a['actual_new_heads']==b['new_heads']==246 and a['actual_new_margin_derivatives']==b['new_complete_margin_derivatives']==66 and a['actual_new_QP_solves']==b['actual_QP_solves']==3
    diagnostics=[read(TRIAL/f'role{r}/diagnostic.json') for r in range(3)];assert all(d['exception'] is None and d['all_parameters_restored'] and d['new_fits']==d['permanent_updates']==0 for d in diagnostics);assert [r['actual_candidate_accepted'] for r in a['roles']]==[True,False,True];assert c['roles'][1]['previously_measured_blocking_functions']==5 and c['roles'][1]['fresh_unmeasured_blocking_functions']==0
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']];catalog=read(mutable[2]);historic={r['id']:dict(r) for r in catalog['documents']};assert catalog['project']['authoritative_delivery_id']=='v159-delivery'
    for row in catalog['documents']:assert sha(ROOT/row['path'])==row['sha256']
    prior=ROOT/'artifacts/v165_final_direction_20261002/plan.md';retained=prior.read_text(encoding='utf-8').split('## 既有证据限制完整保留\n\n',1)[1]
    lines=['# V166 完整覆盖诊断：两个通过，一个被真实非线性退化拒绝','',
        '角色0和2的真实有限试探通过全部原门槛，分别修复356和8条M原行错误且没有M/S新增错误；角色1修复16条M错误但新增10条S错误，被拒绝。三个角色均恢复V164最后真实接受参数，0新增拟合、0永久更新。全角色门槛未通过，因此本轮不授予追加训练。分类掌握、细行为同类支持及跨来源稳定性仍未完成，完整质量交付仍为未通过的V159。','',
        '|角色|新函数/联合函数|起点纯错误|本次纯错误|修复 M/S|新增 M/S|实际有限接受|','|---|---|---:|---:|---|---|---|']
    for item,spec in zip(a['roles'],read(ROOT/'training/review_policy/v166_coverage_first_diagnostic_contract.json')['roles']):
        paired=item['paired_treatment_vs_V164'];repairs='/'.join(str(paired[k]['repairs_vs_previous_accepted']) for k in ['M','S']);new='/'.join(str(paired[k]['new_errors_vs_previous_accepted']) for k in ['M','S']);lines.append(f"|{item['role']}|{spec['fresh_functions']}/{spec['joint_functions']}|{item['pure_errors_before']}|{item['pure_errors_after']}|{repairs}|{new}|{'是' if item['actual_candidate_accepted'] else '否'}|")
    lines+=['',
        '本轮唯一实验变化是：在相同V164真实参数起点和V165位移控制下，把已暴露的33个新函数全部加入联合恢复，法向上限前瞻登记为25。完整输入身份包括规范CSR、顺序16意见、完整2048块及查询位置、真实类别和竞争类别；角色1的25函数是本轮最大实际联合矩阵。零判决地板、全参数原单位证书、严格M/S共同下降、完整原类别分母、原16eps/Armijo和全部实际分类门槛保持原值。','',
        '关键失败事实：角色1新增的10条S错误对应5个完整函数，这5个函数均已在25函数约束中，没有新遗漏函数。局部数学证书合格但真实候选仍错，后续必须分析已测函数的预测间隔与实际间隔之间的非线性残差；单纯扩大容量不能解释本次失败，也没有继续加约束、追加恢复或拟合权限。该事实不证明全局不可行或方法的充分根因。','',
        '上述修复是临时有限候选的分类结果，全部候选结束后恢复V164，未提交为训练状态。角色1先前8条真实修复全部保留，没有再次计算为新收益。三角色原频次共225614个重叠角色观测，不是225614个唯一事件；所有原行、混合冲突、未知字段、原类别分母及累计保护都保留，原206条混合正确行未事后新增冻结。完整部署保护和V138/V140/V142联合能力均通过，保护通过不等于OOF分类掌握或外部泛化。','',
        '实际成本：角色0/1/2分别106/66/74次头与特征、40/18/8次完整裕量导数、各1次联合QP和1个真实步长1候选；总246头/特征、66导数、3QP，实际SLSQP各2次迭代。没有新固定目标或原类别导数，没有新fit/永久更新。累计18784头/特征、368原类别+32固定错误+218裕量=618完整参数导数，自V159以来9拟合/170更新保持不变。','',
        '正式调用前完成25函数真实全参数CPU资格、六个旧结果逐位回放、33新函数身份资格、完整CPU参数生命周期与实际SciPy调用/异常观察资格；11188个物理来源封存后顺序0→1→2执行。生命周期资格中的输出来自已保存向量，法向为明确合成全宽数组，均没有官方调用；CPU峰值约1.97GB仅说明所测CPU资源范围，不是分类结论。两个早期资格源码SyntaxError及原日志继续保存并绑定，AST检查仅覆盖实际可执行依赖。','',
        '零步与恢复重放按既定8eps及严格argmax核查，完整endpoint.pt逐位等于V164真实起点。独立审查复算原gold、所有33函数身份与成对导数、全部原单位QP、完整原行分类和费用账本；审查额外3次CPU保存向量QP回放单列，0官方调用。自审补齐逐类CE、来源集中度、未修复间隔、当前固定读出必要界和完整恢复；root只是来源代理，不构成独立细行为或新来源验收。','',
        f'正式合同：`training/review_policy/v166_coverage_first_diagnostic_contract.json`；执行入口：`training/v166_coverage_first_diagnostic.py`。根独立审查：`{independent.relative_to(ROOT).as_posix()}`；逐行质量：`{own.relative_to(ROOT).as_posix()}`；已测残差阻塞身份：`{blockers.relative_to(ROOT).as_posix()}`。下一方法须单独做现实反例分析、资格、前瞻预算和物理封存。本轮没有自动追加训练、预算扩展、新seed或检查点选择权限。','',
        '## 发布前的当前限制完整保留','']
    lines += ['- '+value for value in catalog['project']['known_limits']]
    lines += ['','## 既有证据限制完整保留','',retained]
    body=('\n'.join(lines)+'\n').encode('utf-8');assert len(body)<=256*1024;new_report=OUT/'review.md'
    additions=[dict(id='v166-review',title='V166完整覆盖结果',path=new_report.relative_to(ROOT).as_posix(),sha256=__import__('hashlib').sha256(body).hexdigest(),category='review_evidence',keywords=['V166']),dict(id='v166-root-review',title='V166根实际审查',path=independent.relative_to(ROOT).as_posix(),sha256=sha(independent),category='review_evidence',keywords=['V166'])]
    assert all(r['id'] not in historic for r in additions);catalog['documents']=additions+catalog['documents'];project=catalog['project'];project['authoritative_direction_id']='v166-review';project['current_summary']='V164三fit/5更新，分类未掌握。V166两通过一拒绝；累计18784头/618导数，0fit/更新，完整交付V159。';project['current_direction']=['分析已测5函数的非线性残差；另登记，不追加训练。'];project['known_limits']=['三个目标均未完成；完整交付V159，全部旧限制保存在当前方向附录。']
    for row in catalog['documents']:
        if row['id'] in historic:assert row==historic[row['id']]
    encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024,f'Capacity {len(encoded)} exceeds unchanged 256KiB limit before write'
    OUT.mkdir()
    for path in mutable:
        target=OUT/'previous'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    REPORT.write_bytes(body);new_report.write_bytes(body);mutable[2].write_bytes(encoded)
    summary=dict(status='V166_two_finite_passes_one_known_function_nonlinear_failure_all_origins_restored',roles=a['roles'],actual_new_heads=246,actual_new_features=246,actual_new_complete_margin_derivatives=66,actual_finite_proposals=3,actual_QP_solves=3,actual_optimizer_iterations=6,new_fits=0,permanent_updates=0,cumulative_heads=18784,cumulative_features=18784,cumulative_full_original_class_gradients=368,cumulative_fixed_error_target_gradients=32,cumulative_margin_gradients=218,cumulative_all_complete_derivatives=618,cumulative_fits_since_V159=9,cumulative_updates_since_V159=170,all_actual_finite_guards_passed=False,supports_new_short_training_registration=False,all_restored_full_tensors_exact=True,all_restored_original_argmax_exact=True,latest_actual_training='V164',latest_complete_quality_delivery='V159',quality_acceptance=False,root_goal_status='active',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),REPORT,own,independent,ib,blockers,prior,TRIAL/'run_seal.json',TRIAL/'registration.json']});save(OUT/'actual_result_summary.json',summary)
    lead='最新实际诊断：[V166完整覆盖两通过一拒绝](docs/V166_COMPLETE_COVERAGE_DIAGNOSTIC_RESULTS_20261002.md)。临时修复M356/16/8，角色1新增10S且5函数均已测，拒绝并全部恢复V164；246头/特征、66导数、3QP、0fit/更新，累计18784头/618导数。训练分类仍未掌握，最新训练V164，完整质量交付V159。\n\n';mutable[0].write_bytes((lead+mutable[0].read_text(encoding='utf-8')).encode('utf-8'));mutable[1].write_bytes((mutable[1].read_text(encoding='utf-8')+'\n\n## V166完整覆盖实际结果\n\n'+lead+'direction=v166-review，delivery=v159-delivery；原v166-plan草案保留。下一步分析已测函数的非线性残差，另做资格与登记，当前不追加训练。\n').encode('utf-8'));mutable[3].write_bytes(mutable[3].read_text(encoding='utf-8').replace('v166-plan','v166-review').encode('utf-8'))
    save(OUT/'publication.json',dict(status=summary['status'],catalog_bytes=len(encoded),historic_document_metadata_preserved=len(historic),authoritative_delivery='v159-delivery',authoritative_direction='v166-review',all_previous_project_known_limits_retained_in_report=True,official_calls_by_publication=0,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),*mutable]}));print(json.dumps(dict(status='V166_complete_coverage_results_published',catalog_bytes=len(encoded),historic_records_preserved=len(historic),official_calls=0)))

if __name__=='__main__':main()
