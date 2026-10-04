"""Publish source-bound entry and compatible protection; official calls still0."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v159_execution_material_publication_20261002'

def main():
    assert not OUT.exists();OUT.mkdir()
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for p in mutable:
        dest=OUT/'previous_publication_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    specs=[
        ('v159-execution-retention-review','V159正式入口与不可区分混类保护精化','docs/V159_CONFLICT_RETENTION_AND_EXECUTION_SUPPLEMENT.md','current_review'),
        ('v159-execution-contract','V159真实入口v3源绑定及同函数有限预算','training/review_policy/v159_boundary_execution_contract.json','current_plan'),
        ('v159-input-preparation','V159完整输入人口和可兼容保护下界实际准备','artifacts/v159_boundary_input_preparation_20261002/qualification.json','current_evidence'),
        ('v159-entry-source-qualification','V159正式入口绑定完成但未封存激活','artifacts/v159_execution_contract_preparation_20261002/qualification.json','current_evidence'),
        ('v159-entry-chunk-gradient','V159真实入口toy空类分块全局原频次梯度','artifacts/v159_boundary_chunk_gradient_qualification_20261002/qualification.json','review_evidence'),
        ('v159-counted-head-gradient','V159实际同头v3可计数意见块CPU-CUDA梯度','artifacts/v159_boundary_torch_synthetic_qualification_v4_20261001/qualification.json','review_evidence'),
        ('v159-joint-input-conflict','V159无序16意见与初始全部正确守卫冲突','artifacts/v159_joint_input_retention_feasibility_audit_20261002/audit.json','review_evidence'),
        ('v159-counted-train-source','V159真实全原行梯度预飞行与六fit入口源码','training/v159_boundary_train.py','review_evidence'),
        ('v159-full-evaluate-source','V159六真实终点窗口与全2056871验收源码','training/v159_boundary_evaluate.py','review_evidence')]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+[ROOT/s[2] for s in specs]}
    (OUT/'pre_publication_bindings.json').write_text(json.dumps(dict(source_sha256=bound,status='before_source_and_retention_supplement_publication'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    prep=read(ROOT/specs[2][2]);assert prep['input_qualification_passed'] and prep['official_classifier_calls']==prep['official_gradients']==prep['official_fits']==0
    p=read(ROOT/specs[1][2]);assert p['candidate_module']=='training/v159_current_input_boundary_v3.py' and p['total_future_caps']['opinion_feature_blocks']==77832
    cat=read(mutable[2]);assert cat['project']['authoritative_delivery_id']=='v158-delivery' and cat['project']['authoritative_direction_id']=='v159-function-budget-review'
    for e in cat['documents']:assert sha(ROOT/e['path'])==e['sha256'],e['id']
    snapshots=[]
    for ident,title,name,category in specs:
        source=ROOT/name;dest=OUT/'physical_source_snapshots'/name
        if dest.suffix=='.py':dest=dest.with_suffix('.py.txt')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);assert sha(dest)==sha(source) and dest.stat().st_size<=256*1024
        assert not any(e['id']==ident for e in cat['documents'])
        cat['documents'].insert(0,dict(id=ident,title=title,path=dest.relative_to(ROOT).as_posix(),category=category,summary=title,sha256=sha(dest),keywords=['V159','入口','原频次','冲突','保护','未激活']))
        snapshots.append(dict(id=ident,source=name,snapshot=dest.relative_to(ROOT).as_posix(),sha256=sha(dest)))
    project=cat['project'];project['authoritative_direction_id']='v159-execution-retention-review'
    project['current_summary']='V158完成60次真实拟合及2056871原行验收，融合1200完整梯度/1200proposal/1200更新。ASA414M/1872S，完整质量失败、未晋升。V159唯一v3完整输入+16意见自由三类头源码/计数入口/固定预算与实际保存人口已绑定，真实入口toy空类分块梯度已通过。新OOF全部correct冻结会将下界22/4/26抬成22/72/110，正式注册前改为纯当前X正确保护；已验收部署225558及全部联合能力不变。未来6fit/2412类梯度/77832头与意见块；当前官方新函数/梯度/拟合/更新0，真实封存/零步/拟合仍未执行，完整目标active。'
    project['current_direction']=[
        '正式唯一candidate_module v159_current_input_boundary_v3.py，与已发布v2函数/参数/seed相同，仅意见块显式可计数；旧资格不直接激活。',
        '真实counted register/preflight/fit/evaluate入口与完整验收已source绑定，先独立审查实际源，再seal全部导入/运行库/数据和预飞行。',
        '全部已验收225558部署TRAIN正确及V138/V140/V142联合范围保持；B冻结初始正确纯当前X OOF组，混标原行仍全部原频率loss/质量和类源账本。',
        '注册训练门槛：纯当前X原行错0，混类总错<=22/6/28，纯OOF保护错0；完整可见tuple下界22/4/26、全correct人为下界22/72/110仍明确分列，概率尾差不是新行为证据。',
        '未来6fit最多2412完整类梯度、3600累计proposal、1200接受更新、77832头及意见块；真实入口toy2049行验证空类chunk除全局原行mass正确，不能用该toy代替官方预飞行。',
        '现存45当前NN及9旧N1不重跑；A/B同函数/数值/初始化，稳定logCE分报probclipCE，endpoint固定/最后五不凑满；所有完整N/M/S与来源条件不变。',
        '当前官方函数/特征/梯度/fit/更新0，runtime实际seal/zero-step尚未发生；三个目标仍未完成，不停在来源报告而继续有界执行准备。']
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024;mutable[2].write_bytes(raw)
    mutable[0].write_text('当前正式入口与兼容保护：[V159执行补充](docs/V159_CONFLICT_RETENTION_AND_EXECUTION_SUPPLEMENT.md)。完整函数/计数入口/来源人口及未来预算已绑定，已验收能力全保护；新OOF混类冻结矛盾已在正式注册前修正。官方新调用/拟合0，真实封存和预飞行仍待执行。\n\n'+mutable[0].read_text(encoding='utf-8'),encoding='utf-8')
    mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V159正式入口及保护冲突前置修正\n\n`docs/V159_CONFLICT_RETENTION_AND_EXECUTION_SUPPLEMENT.md`、`training/review_policy/v159_boundary_execution_contract.json` 已绑定真正v3头/计数训练/验收/输入准备源。完整X+无序意见下界22/4/26，全OOF初始correct冻结下界22/72/110，不相容；新OOF未验收，B只冻结纯当前X正确组。部署225558及全部已验收联合能力全保留。真实保存输入准备已exit0，选定保护与纯错0/混类<=22/6/28目标兼容；float意见split不称行为证据。实际toy入口2049行空类分块与完整原频次梯度误差<=5.6e-17。官方新函数/特征/梯度/fit/更新仍0，注册/运行库seal/真实零步未执行。MCP当前方向v159-execution-retention-review，最新真实v158-delivery。旧资格/原full OOF guard矛盾及所有失败文件不改。\n',encoding='utf-8')
    tests=mutable[3].read_text(encoding='utf-8').replace('v159-function-budget-review','v159-execution-retention-review')
    addition='''    def test_v159_execution_retention_resolves_pre_fit_conflict(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        contract=json.loads(readonly._document_text(entries['v159-execution-contract']))
        prep=json.loads(readonly._document_text(entries['v159-input-preparation']))
        self.assertEqual(contract['candidate_module'],'training/v159_current_input_boundary_v3.py')
        self.assertEqual(contract['total_future_caps']['opinion_feature_blocks'],77832)
        self.assertTrue(contract['OOF_retention_refinement']['all_accepted_deployment_correct_and_joint_scopes_unchanged'])
        self.assertTrue(prep['input_qualification_passed'])
        self.assertEqual(prep['official_classifier_calls'],0)
        self.assertEqual([r['scopes']['OOF']['unconstrained_minimum_original_errors'] for r in prep['roles']],[22,4,26])
        self.assertEqual([r['scopes']['OOF']['all_initial_correct_guard_constrained_minimum_errors'] for r in prep['roles']],[22,72,110])
        self.assertEqual(sum(r['scopes']['deployment']['protected_initial_correct_original_rows'] for r in prep['roles']),225558)

'''
    needle='    def test_non_loopback_binding_requires_token(self) -> None:\n';assert needle in tests;mutable[3].write_text(tests.replace(needle,addition+needle),encoding='utf-8')
    check_bindings(bound)
    (OUT/'publication.json').write_text(json.dumps(dict(status='V159_real_entry_and_compatible_retention_published_without_official_execution',catalog_bytes=len(raw),authoritative_delivery='v158-delivery',authoritative_direction='v159-execution-retention-review',documents=snapshots,
        official_classifier_calls=0,official_features_calls=0,official_gradients=0,official_fits=0,official_updates=0,official_runtime_sealed=False,official_zero_step_replay=False,quality_acceptance=False,goal_status='active',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+mutable}),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='actual_entry_source_materials_published_no_formal_calls',catalog_bytes=len(raw),official_new_fits=0)))

if __name__=='__main__':main()
