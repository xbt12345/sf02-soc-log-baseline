"""Publish all four actual candidates and preserve the bounded reader history."""
import hashlib,json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

OUT=ROOT/'artifacts/v167_results_20261002'
REPORT=ROOT/'docs/V167_COMPLETE_TRIAL_POINT_RESTORATION_RESULTS_20261002.md'
TRIAL=ROOT/'artifacts/v167_trial_point_restoration_diagnostic_20261002'
QUALITY=ROOT/'artifacts/v167_saved_trial_point_quality_review_20261002/review.json'
GEOMETRY=ROOT/'artifacts/v167_saved_remaining_argmax_geometry_review_20261002/review.json'
AUDIT=ROOT/'artifacts/v167_independent_actual_trial_point_review_v2_20261002/review.json'
ROOT_GEOMETRY=ROOT/'artifacts/v167_root_decision_tie_and_floor_review_20261002/review.json'
REPLAY=ROOT/'artifacts/v167_observed_runtime_failure_replay_20261002/replay.json'

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def main():
    assert not OUT.exists() and not REPORT.exists()
    a,b,c,d,replay=map(read,[QUALITY,AUDIT,GEOMETRY,ROOT_GEOMETRY,REPLAY])
    bindings_files=[QUALITY,GEOMETRY,REPLAY,AUDIT.parent/'pre_review_bindings.json',ROOT_GEOMETRY.parent/'pre_bindings.json']
    for p in bindings_files:check_bindings(read(p)['source_sha256'])
    assert a['actual_new_heads']==b['new_heads']==296 and a['actual_new_complete_margin_derivatives']==b['new_complete_margin_derivatives']==100
    assert a['actual_QP_solves']==b['actual_QP_solves']==2 and a['actual_finite_proposals']==4
    assert not a['all_actual_finite_guards_passed'] and not b['supports_new_short_training_registration'] and not a['classification_mastery']
    assert replay['status']=='V167_six_actual_runtime_constraints_replayed' and len(replay['cases'])==6
    assert b['status']=='V167_actual_trial_points_original_gold_full_gradients_finite_guards_costs_and_restoration_independently_verified'
    assert d['status']=='V167_exact_actual_argmax_tie_and_saved_trial_geometry_independently_confirmed' and d['official_heads']==d['official_derivatives']==0
    diagnostics=[read(TRIAL/f'role{r}/diagnostic.json') for r in range(3)]
    assert all(x['exception'] is None and x['all_parameters_restored'] and x['new_fits']==x['permanent_updates']==0 for x in diagnostics)
    assert [r['actual_finite_accepted'] for r in a['roles']]==[True,False,True]
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py','mcp_readonly/tests/test_catalog_archive.py']]
    catalog=read(mutable[2]);effective=load_catalog(ROOT,mutable[2],256*1024);archive=ROOT/catalog['historical_catalog']['path']
    historic={r['id']:dict(r) for r in effective['documents']};assert len(historic)==447 and effective['project']['authoritative_delivery_id']=='v159-delivery'
    assert len(read(archive)['documents'])==447 and sha(archive)==catalog['historical_catalog']['sha256']
    for row in historic.values():assert sha(ROOT/row['path'])==row['sha256'] and (ROOT/row['path']).stat().st_size<=256*1024
    prior=ROOT/'artifacts/v166_direction_20261002/plan.md';retained=prior.read_text(encoding='utf-8').split('## 既有证据限制完整保留\n\n',1)[1]
    lines=['# V167 试探点恢复：两次修正后仍被两条真实 S 平手错误拒绝','',
        '角色1的两次实际修正均修复4条M旧错、新增2条S错，纯错误1952→1950，均被原分类与累计保护拒绝。相对V166，恢复8条S但丢失12条M修复。第二次剩余两条S的概率及log概率完全相等，原argmax选择M；局部零地板合格不能追认分类安全。角色0和2仅重放V166原安全候选。所有参数恢复V164，0新拟合、0永久更新；全角色训练前提、分类掌握及整体目标均未通过。','',
        '|角色/候选|纯错误起点→候选|修复 M/S vs V164|新增 M/S vs V164|原实际接受|','|---|---:|---|---|---|']
    for item in a['roles']:
        for candidate in item['all_finite_candidates']:
            paired=candidate['paired_vs_V164'];repair='/'.join(str(paired[k]['repairs_vs_previous_accepted']) for k in ['M','S']);new='/'.join(str(paired[k]['new_errors_vs_previous_accepted']) for k in ['M','S'])
            lines.append(f"|{item['role']}/{candidate['proposal']}|{candidate['pure_errors_before']}→{candidate['pure_errors_after']}|{repair}|{new}|{'是（未提交）' if candidate['actual_candidate_accepted'] else '否'}|")
    lines+=['',
        '本轮唯一机制变化：角色1在真实V166候选试探点重新测量同一25个完整函数的全参数法向，每个点各两次测量，使用原点M/S类别梯度约束原位移加新修正。原2048块、规范CSR、顺序16意见、查询位置、真值/竞争类身份不变；未换目标、seed、步长、输入或模型容量。角色0和2没有新优化，其有限结果逐项重放V166安全控制，不计为新的分类收益。','',
        '每次实际候选都从V164真实原点应用完整位移，按完整OOF与部署原行、原gold及类别分母判定。第一次失败只有阻塞函数均已覆盖、原固定错误目标下降与Armijo独立合格、最大负间隔小于前次0.99倍时才允许已登记的第二次修正；这一继续门槛不替代分类验收。第二次无论局部残差如何减少都停止，没有第三次修正权限。','',
        '精确失败几何：两条原行都对应OOF local21985、truth=2(S)、rival=1(M)，同一已测完整函数。第一次局部预测log间隔2.168404344971009e-19，实际为-3.091487621453837e-08，q差-1.5457438107269184e-08。第二次预测、实际log间隔和q差都为精确0；q1=q2=0.49999999999448597，logq1=logq2=-0.6931471805709734，argmax=1。原单位局部残差证书仍合格，但实际两条保护错误没有消失。该反例只定位本次判决边界问题，不证明全局不可行，也不授权事后改变容差、标签、argmax或零地板。','',
        '两次剩余阻塞始终在已测25函数内，不是遗漏函数或容量不足。两次候选均相对V164修复4M、新增2S；相对V166恢复8S、丢失12M修复，不能将总错误减少或部分S恢复称为共同分类通过。既有8条真实修复保留，所有混合冲突与原行仍参与质量统计，原206条混合正确行未事后新增冻结。','',
        '实际费用：角色0/1/2分别66/164/66次头与特征调用，0/100/0次完整裕量导数，0/2/0次QP，1/2/1个真实步长1候选；总296头/特征、100导数、2实际QP、4有限候选，优化器内部共4次迭代。全部0fit/0永久更新。累计19080头/特征，368原类别+32固定错误目标+318裕量=718完整参数导数；自V159以来9fit/170更新不变，预算没有重置。','',
        '完整endpoint.pt逐位恢复V164，原q/logq按既定8eps重放，argmax严格一致。V138/V140/V142已验收训练能力和部署保护通过；其范围不等于OOF掌握、独立新来源或细行为同类支持。三角色225614个观测存在重叠，不能加为唯一事件收益。最新真实训练仍V164，最新2056871原行完整质量交付仍为未通过的V159，三个总目标及后续完整三分类/泛化审查继续未完成。','',
        '执行前完成身份、生命周期、对抗资格及14973个物理来源封存。早期身份资格KeyError及修复后的独立v2、完整原日志均保留；生命周期资格使用缓存输出和明确合成全宽法向，不能当实际模型分类证据。根三个全参数CPU非线性反例检验一修正可通过、两修正仍失败必须停止，0官方调用。实际入口与合同封存后未改动。','',
        '独立实际审计v2复核完整原gold/所有25函数两点成对导数/试探点身份/原点类别梯度/完整有限门槛/成本/恢复；额外2次保存向量CPU QP回放单列，0官方调用。根v1被原封存保留；v2只对GPU分块与CPU求和产生的极小诊断浮点差采用原8eps比较，分类布尔、原16eps及Armijo判定均未放宽。逐行质量还覆盖逐类CE、来源代理集中度、未修复间隔及固定读出必要界，均不证明独立来源或细行为验收。','',
        '根精确平手审查另做2个保存点CPU局部正地板预览（0官方模型调用），不是两个新非线性候选，也没有分类收益。后续只能在单独前瞻合同、资格与封存之后检验判决边界机制；本报告没有授予新官方调用、拟合、第三次修正或预算扩展。','',
        '只读目录将447条旧记录完整保存到单个哈希绑定历史索引，保留原ID、路径、哈希和全部元数据；主索引独立保存新记录。每个索引仍受256KiB上限约束，禁止嵌套索引、路径越界、哈希变化与重复ID。search/fetch/get_project_status三接口和只读约束不变。接口测试及历史内容校验为本地验证，不构成外部云连接验收。','',
        f'实际合同：`training/review_policy/v167_trial_point_restoration_contract.json`；执行入口：`training/v167_trial_point_restoration_diagnostic.py`。完整逐行质量：`{QUALITY.relative_to(ROOT).as_posix()}`；独立实际审计：`{AUDIT.relative_to(ROOT).as_posix()}`；精确失败几何：`{GEOMETRY.relative_to(ROOT).as_posix()}`；根平手审查：`{ROOT_GEOMETRY.relative_to(ROOT).as_posix()}`；六项实际失败约束回放：`{REPLAY.relative_to(ROOT).as_posix()}`。','',
        '## 发布前的当前限制完整保留','']
    lines+=['- '+v for v in effective['project']['known_limits']]
    lines+=['','## 既有证据限制完整保留','',retained]
    body=('\n'.join(lines)+'\n').encode('utf-8');assert len(body)<=256*1024
    saved_report=OUT/'review.md';additions=[dict(id='v167-review',title='V167两次修正实际失败',path=saved_report.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(body).hexdigest(),category='review_evidence',keywords=['V167']),dict(id='v167-root-review',title='V167根独立实际审计',path=AUDIT.relative_to(ROOT).as_posix(),sha256=sha(AUDIT),category='review_evidence',keywords=['V167']),dict(id='v167-tie-review',title='V167根精确平手审查',path=ROOT_GEOMETRY.relative_to(ROOT).as_posix(),sha256=sha(ROOT_GEOMETRY),category='review_evidence',keywords=['V167'])]
    assert all(r['id'] not in historic for r in additions);catalog['documents']=additions+catalog['documents'];p=catalog['project'];p['authoritative_direction_id']='v167-review';p['current_summary']='V167两次修正仍新增2S平手错误；296头/100导数/2QP，0fit/更新，全恢复V164。累计19080头/718导数；三目标未完成。';p['current_direction']=['保存精确argmax平手反例；后续判决边界机制须单独资格与封存，无追加训练权限。'];p['known_limits']=['三个目标与完整质量未完成；最新训练V164，完整交付V159；所有旧限制保存在当前结果附录。']
    encoded=(json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024
    OUT.mkdir()
    for path in mutable:
        target=OUT/'previous'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    REPORT.write_bytes(body);saved_report.write_bytes(body);mutable[2].write_bytes(encoded)
    assert sha(archive)==catalog['historical_catalog']['sha256'];after=load_catalog(ROOT,mutable[2],256*1024);now={r['id']:r for r in after['documents']};assert len(now)==450 and all(now[k]==v for k,v in historic.items())
    summary=dict(status='V167_two_trial_point_corrections_rejected_by_two_actual_S_argmax_ties_all_origins_restored',roles=a['roles'],actual_new_heads=296,actual_new_features=296,actual_new_complete_margin_derivatives=100,actual_finite_proposals=4,actual_QP_solves=2,actual_optimizer_iterations=4,new_fits=0,permanent_updates=0,cumulative_heads=19080,cumulative_features=19080,cumulative_full_original_class_gradients=368,cumulative_fixed_error_target_gradients=32,cumulative_margin_gradients=318,cumulative_all_complete_derivatives=718,cumulative_fits_since_V159=9,cumulative_updates_since_V159=170,all_actual_finite_guards_passed=False,supports_new_short_training_registration=False,all_restored_full_tensors_exact=True,all_restored_original_argmax_exact=True,latest_actual_training='V164',latest_complete_quality_delivery='V159',quality_acceptance=False,root_goal_status='active',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),REPORT,QUALITY,AUDIT,GEOMETRY,ROOT_GEOMETRY,REPLAY,*bindings_files,prior,TRIAL/'run_seal.json',TRIAL/'registration.json',archive]});save(OUT/'actual_result_summary.json',summary)
    lead='当前实际结论：[V167两次修正仍被两条S平手错误拒绝](docs/V167_COMPLETE_TRIAL_POINT_RESTORATION_RESULTS_20261002.md)。角色1修复4M、新增2S；0/2仅重放V166安全控制；296头/100导数/2QP、0fit/更新，全恢复V164，累计19080头/718导数。分类仍未掌握，最新训练V164、完整质量交付V159，三个总目标均未完成。下一判决边界机制须另做资格与物理封存。\n\n'
    previous_readme=mutable[0].read_text(encoding='utf-8');assert previous_readme.startswith('当前方向：');previous_readme=previous_readme.split('\n\n',1)[1];mutable[0].write_bytes((lead+previous_readme).encode('utf-8'))
    mutable[1].write_bytes((mutable[1].read_text(encoding='utf-8')+'\n\n## V167实际试探点完整结果\n\n'+lead+'direction=v167-review，delivery=v159-delivery；原v167-plan及原合同/资格/封存/失败日志不改。角色1两次硬停止，无第三次修正或新fit权限；根保存点CPU预览不计实际收益。447旧目录元数据和来源内容完整保留，新增3条只读证据。\n').encode('utf-8'))
    mutable[3].write_bytes(mutable[3].read_text(encoding='utf-8').replace('v167-plan','v167-review').encode('utf-8'))
    save(OUT/'publication.json',dict(status=summary['status'],catalog_bytes=len(encoded),historical_archive_bytes=archive.stat().st_size,historic_document_metadata_preserved=447,effective_documents=450,new_documents=3,authoritative_delivery='v159-delivery',authoritative_direction='v167-review',all_previous_project_known_limits_retained_in_report=True,official_calls_by_publication=0,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),*mutable,ROOT/'mcp_readonly/catalog_io.py',ROOT/'mcp_readonly/server.py',archive]}));print(json.dumps(dict(status='V167_complete_actual_results_published',catalog_bytes=len(encoded),historic_records_preserved=447,effective_documents=450,official_calls=0)))

if __name__=='__main__':main()
