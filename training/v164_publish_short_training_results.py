"""Publish completed, audited supervised trajectories without model promotion."""
import json, shutil
from pathlib import Path
from experiment_review import ROOT, read, sha, check_bindings

TRIAL=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
OUT=ROOT/'artifacts/v164_results_20261002'
REPORT=ROOT/'docs/V164_COMPLETE_SHORT_SUPERVISED_TRAINING_RESULTS_20261002.md'

def save(path,value):
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists() and not REPORT.exists()
    own=ROOT/'artifacts/v164_saved_training_quality_review_20261002/review.json'
    independent=ROOT/'artifacts/v164_independent_actual_short_trajectory_review_20261002/review.json'
    ib=independent.parent/'pre_review_bindings.json'
    a,b=read(own),read(independent)
    check_bindings(a['source_sha256']); check_bindings(read(ib)['source_sha256'])
    assert b['status']=='actual_short_trajectories_original_gold_gradients_functions_commits_and_costs_independently_verified'
    assert b['official_calls_by_this_review']==a['official_review_calls']==0
    fits=[read(TRIAL/f'role{r}/fit.json') for r in range(3)]
    heads=sum(f['counts']['head_attempts'] for f in fits)
    target=sum(f['counts']['gradient_attempts'] for f in fits)
    margin=sum(f['counts']['margin_attempts'] for f in fits)
    updates=sum(f['permanent_updates'] for f in fits)
    assert (heads,target,margin,updates)==(b['new_heads'],b['new_target_gradients'],b['new_margin_gradients'],b['permanent_updates'])
    assert b['new_fits']==3 and all(f['exception'] is None and f['endpoint_joint_TRAIN_retention']['passed'] for f in fits)
    caps=read(ROOT/'training/review_policy/v164_short_trajectory_prospective_budget.json')
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    catalog=read(mutable[2]); historic={r['id']:dict(r) for r in catalog['documents']}
    assert catalog['project']['authoritative_delivery_id']=='v159-delivery'
    for row in catalog['documents']: assert sha(ROOT/row['path'])==row['sha256']
    previous=ROOT/'artifacts/v163_results_20261002/review.md'
    retained=previous.read_text(encoding='utf-8').split('## 既有证据限制完整保留\n\n',1)[1]
    lines=['# V164 三角色真实短程监督训练结果','',
           '三个角色均执行了事前封存的真实短程，保存所有接受、拒绝和停止状态，并保留最后真实接受参数。训练侧分类尚未掌握，细行为同类支持与跨来源稳定性尚未验收。最新完整三分类训练质量交付仍为 V159；本轮监督训练记录不构成独立来源泛化或模型晋升。','',
           '|角色|真实接受更新|停止原因|纯错误：固定B→终点|完整错误：固定B→终点|相对固定B修复/新增错误|',
           '|---|---:|---|---|---|---|']
    pure_start=[3278,1960,1586]; full_start=[3300,2032,1698]
    for role,(fit,quality) in enumerate(zip(fits,a['roles'])):
        cls=quality['endpoint_classes']; repairs=sum(c['repairs_vs_fixed_endpoint'] for c in cls.values()); regressions=sum(c['new_errors_vs_fixed_endpoint_all_original_rows'] for c in cls.values())
        assert repairs==b['roles'][role]['repairs_vs_fixed_endpoint'] and regressions==b['roles'][role]['new_errors_vs_fixed_endpoint']
        lines.append(f"|{role}|{fit['permanent_updates']}|{fit['status']}|{pure_start[role]}→{fit['endpoint_OOF_stats']['pure_errors']}|{full_start[role]}→{fit['endpoint_OOF_stats']['total_errors']}|{repairs}/{regressions}|")
    lines+=['','每个角色最多10个真实接受更新，单个新参数点最多6次恢复及24个间隔函数；本轮实际遇到的停止按原合同保留，未增加容量、重置预算、挑选早期最佳参数或补跑。新点重新测量两类完整固定错误梯度及其复测，旧参数点的法向不作为新点缓存。训练目标始终为固定纯错误的原类别质量贡献，原类别完整分母不变。','',
            '固定保护包含原纯正确与此前保护能力；最初未纳入该固定保护的206条混合正确行仍完整计分（各角色106/6/94条）。每个新修复的原行，包括混合来源修复，进入累计正确保护。独立原gold审核逐行报告了全部原行新增错误，不用“受保护回退为0”替代全原行质量。','',
            '每个实际接受状态均保存完整 checkpoint、q/logq、原行、累计保护、四参数段变化与当前固定读出界限。固定读出条件下不能覆盖先验的数量只描述该参数状态；读出变化后重新计算，不视为永久容量证明，未被该界限阻挡也不证明已经可学。','',
            '|角色|头/特征|固定错误完整导数|间隔完整导数|真实候选|方向QP/恢复QP|恢复优化内部迭代|',
            '|---|---:|---:|---:|---:|---|---:|']
    for fit in fits:
        c=fit['counts']; lines.append(f"|{fit['role']}|{c['head_attempts']}|{c['gradient_attempts']}|{c['margin_attempts']}|{fit['finite_proposals']}|{fit['direction_QP_solves']}/{fit['restoration_QP_solves']}|{fit['restoration_optimizer_iterations']}|")
    cumulative_heads=17662+heads; cumulative_target=12+target; cumulative_margin=42+margin; cumulative_derivatives=368+cumulative_target+cumulative_margin
    lines+=['',f'本轮实际3拟合、{updates}永久更新、{heads}头/特征、{target}固定错误导数、{margin}间隔导数、0原类别导数。累计{cumulative_heads}头/特征、368原类别+{cumulative_target}固定错误+{cumulative_margin}间隔={cumulative_derivatives}完整参数导数；自V159以来9拟合、{165+updates}永久更新。注册上限6216头/特征、120固定错误导数、1296间隔导数、3拟合/30更新；未使用部分不是重启或额外训练权限。','',
           '入口 `training/v164_short_supervised_trajectory.py`、合同 `training/review_policy/v164_short_supervised_trajectory_contract.json` 与8686个物理来源的事前封存位于 `artifacts/v164_short_supervised_trajectory_20261002`。逐行分类质量自审、独立完整梯度/函数/参数/成本复算均实际通过，审查不增加官方模型调用。所有接受和拒绝候选、停止及真实控制台保留。','',
           '首次封存因20GiB空间门槛在正式注册前停止。随后83个历史数组原地无损NTFS压缩，逐文件前后SHA相同，释放758345728字节；没有删除数据。唯一正式注册由根侧完成，重复注册进程在目录已存在时退出，两项均0官方调用。','',
           '详细逐行类别CE、来源代理集中度、零召回来源组、未修复错误的间隔分布、每状态固定读出诊断见 `artifacts/v164_saved_training_quality_review_20261002/review.json` 与原训练目录。来源root仅为来源代理，不能冒称独立细行为。所有既有部署与联合TRAIN保护通过；完整分类掌握及最后五个真实不同掌握状态均以实际结果判断，未获豁免。','',
           '下一步必须针对实际停止证据区分计算容量截断与未找到合格有限方向，采用独立前瞻登记；本报告没有新增训练预算、容量放宽或完整模型交付权限。','',
           '## 既有证据限制完整保留','',retained]
    body='\n'.join(lines)+'\n'
    OUT.mkdir()
    for path in mutable:
        target_path=OUT/'previous'/path.relative_to(ROOT); target_path.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target_path)
    REPORT.write_bytes(body.encode('utf-8')); snapshot=OUT/'review.md'; shutil.copyfile(REPORT,snapshot)
    catalog['documents'].insert(0,dict(id='v164-review',title='V164真实短程',path=snapshot.relative_to(ROOT).as_posix(),sha256=sha(snapshot),category='review_evidence',keywords=['V164']))
    project=catalog['project']; project['authoritative_direction_id']='v164-review'
    project['current_summary']=f'V164三fit/{updates}更新，分类未掌握；旧能力保护通过。累计{cumulative_heads}头/{cumulative_derivatives}导数。完整交付仍V159。'
    project['current_direction']=['按实际停止证据登记下一单问题有限诊断。']
    project['known_limits'][0]='V159及V164训练分类掌握均失败；本轮不构成完整模型晋升。'
    project['known_limits'][5]='V164逐行质量含所有混合行变化；来源root只作代理。'
    for row in catalog['documents']:
        if row['id'] in historic: assert row==historic[row['id']]
    encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8'); assert len(encoded)<=256*1024
    summary=dict(status='three_actual_short_supervised_fits_reviewed_classification_not_mastered',roles=a['roles'],actual_new_heads=heads,actual_new_features=heads,actual_new_fixed_error_target_gradients=target,actual_new_margin_gradients=margin,actual_new_full_original_class_gradients=0,new_fits=3,permanent_updates=updates,cumulative_heads=cumulative_heads,cumulative_features=cumulative_heads,cumulative_full_original_class_gradients=368,cumulative_fixed_error_target_gradients=cumulative_target,cumulative_margin_gradients=cumulative_margin,cumulative_all_complete_derivatives=cumulative_derivatives,cumulative_fits_since_V159=9,cumulative_updates_since_V159=165+updates,all_joint_TRAIN_retention_passed=True,classification_mastered=a['training_side_all_roles_mastered'],quality_acceptance=False,root_goal_status='active',latest_complete_trained_quality_delivery='V159',no_model_promotion=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),REPORT,own,independent,ib,previous,ROOT/'training/review_policy/v164_short_trajectory_prospective_budget.json']})
    save(OUT/'actual_result_summary.json',summary); mutable[2].write_bytes(encoded)
    lead=f'最新实际：[V164三角色真实短程监督训练](docs/V164_COMPLETE_SHORT_SUPERVISED_TRAINING_RESULTS_20261002.md)。3拟合/{updates}永久更新，分类尚未掌握，旧能力保护通过；累计{cumulative_heads}头/{cumulative_derivatives}完整导数。下一针对实际停止证据另登记，完整交付仍V159。\n\n'
    mutable[0].write_bytes((lead+mutable[0].read_text(encoding='utf-8')).encode('utf-8'))
    mutable[1].write_bytes((mutable[1].read_text(encoding='utf-8')+'\n\n## V164真实短程完成、分类未掌握\n\n'+lead+'delivery=v159-delivery，direction=v164-review；全原行与累计修复保护实际复核，未晋升完整模型。\n').encode('utf-8'))
    mutable[3].write_bytes(mutable[3].read_text(encoding='utf-8').replace('v163-review','v164-review').encode('utf-8'))
    save(OUT/'publication.json',dict(status=summary['status'],catalog_bytes=len(encoded),historic_document_metadata_preserved=len(historic),authoritative_delivery='v159-delivery',authoritative_direction='v164-review',official_calls_by_publication=0,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),*mutable]}))
    print(json.dumps(dict(status='V164_actual_short_training_results_published',catalog_bytes=len(encoded),historic_records_preserved=len(historic))))

if __name__=='__main__': main()
