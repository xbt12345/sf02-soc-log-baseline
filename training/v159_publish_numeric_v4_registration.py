"""Publish qualified wired numeric v4 after actual registration, before preflight."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v159_numeric_v4_registration_publication_20261002'
def main():
    assert not OUT.exists();OUT.mkdir()
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for p in mutable:
        target=OUT/'previous_publication_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    specs=[('v159-numeric-v4-current-review','V159当前数值v4实际注册及累计成本78196合同','docs/V159_NUMERIC_V4_REGISTERED_STATUS.md'),
      ('v159-numeric-v4-contract','V159数值v4未来78072与历史124合计78196封存契约','training/review_policy/v159_boundary_execution_contract_v4.json'),
      ('v159-numeric-v4-entry-qualification','V159新数值v4完整入口接线实际资格','artifacts/v159_numeric_execution_contract_preparation_20261002/qualification.json'),
      ('v159-numeric-policy-source','V159固定8eps重复与16eps有限下降数值政策源码','training/v159_float64_repeat_policy_v2.py'),
      ('v159-nonzero-dimension-qualification','V159真实尺寸非零合成及饱和方向拒绝资格','artifacts/v159_nonzero_real_dimension_numeric_qualification_v2_20261002/qualification.json'),
      ('v159-batch-replay-qualification','V159实际合成排序分块及有限下降探针资格','artifacts/v159_numeric_batch_replay_qualification_v2_20261002/qualification.json'),
      ('v159-numeric-v4-protocol-cases','V159新数值协议20个实际正反例','artifacts/v159_protocol_synthetic_qualification_v3_20261002/qualification.json'),
      ('v159-independent-numeric-policy','V159独立真实尺寸保存非零状态数值审查','artifacts/v159_independent_numeric_policy_saved_state_audit_20261002/audit.json')]
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+[ROOT/s[2] for s in specs]}
    (OUT/'pre_publication_bindings.json').write_text(json.dumps(dict(source_sha256=bindings),indent=2)+'\n',encoding='utf-8')
    trial=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002';reg=read(trial/'registration.json');assert reg['official_classifier_calls']==reg['official_gradients']==reg['official_fits']==reg['official_updates']==0 and reg['seal_sha256']==sha(trial/'run_seal.json')
    plan=read(ROOT/specs[1][2]);assert plan['total_cumulative_caps']['classifier_forward_chunks']==78196 and plan['total_cumulative_caps']['full_class_gradients']==2420
    cat=read(mutable[2]);assert cat['project']['authoritative_delivery_id']=='v158-delivery'
    for entry in cat['documents']:assert sha(ROOT/entry['path'])==entry['sha256'],entry['id']
    published=[]
    for ident,title,name in specs:
        src=ROOT/name;dest=OUT/'physical_source_snapshots'/name
        if dest.suffix=='.py':dest=dest.with_suffix('.py.txt')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);assert sha(src)==sha(dest) and dest.stat().st_size<=256*1024 and all(e['id']!=ident for e in cat['documents'])
        cat['documents'].insert(0,dict(id=ident,title=title,path=dest.relative_to(ROOT).as_posix(),category='current_review' if ident==specs[0][0] else 'review_evidence',sha256=sha(dest),keywords=['V159','数值v4','非零资格','78196','2420']))
        published.append(dict(id=ident,path=dest.relative_to(ROOT).as_posix(),sha256=sha(dest)))
    p=cat['project'];p['authoritative_direction_id']='v159-numeric-v4-current-review'
    p['current_summary']='V158完成60次真实拟合及2056871完整原行验收，融合1200完整梯度/1200proposal/1200更新，质量失败、未晋升。V159真实旧预检失败和有限诊断累计124头/意见块、8完整类梯度、0fit/更新；新固定数值政策经过真实尺寸非零、排序分块、有限下降探针、保存真实向量及独立审查。新v4入口/评价/审查20案例实际通过，新注册物理seal已执行，完整新预检尚未执行。未来78072/2412，含历史累计78196/2420，6fit和训练2400类梯度不变。全目标active。'
    p['current_direction']=['当前真实入口train_v4/evaluate_v4和runtime_v4调用版本协议v159_v3，唯一分类函数仍current_input_boundary_v3。','固定8eps重复政策，无梯度绝对下限，逐行argmax和旧能力严格；方向无法分辨则停止，16eps有限下降余量不能以容差替代学习。','实际真实尺寸非零及不同排序分块资格通过；强饱和状态共同下降拒绝，固定B有限探针确实下降。20协议正反例及独立保存数组审查通过。','旧失败84/4与封存诊断40/4共124/8永久保留；新register已seal，新预检和6fits未执行。','未来新运行78072头/2412类梯度，含历史累计78196/2420；只增加有依据技术额度124/8，不增加训练6fit/2400梯度/3600proposal/1200更新。','最新实际训练和完整交付仍V158失败；训练学习、支持、迁移及完整三分类泛化目标active。']
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(raw)<=256*1024;mutable[2].write_bytes(raw)
    lead='当前新版本：[V159数值v4实际注册封存](docs/V159_NUMERIC_V4_REGISTERED_STATUS.md)。真实尺寸非零及分块资格已过，新register实际exit0；旧真实技术成本124头/8类梯度，新预检和6fits尚未执行。未来78072/2412、含历史累计78196/2420；最新训练V158质量失败。\n\n'
    mutable[0].write_text(lead+mutable[0].read_text(encoding='utf-8'),encoding='utf-8');mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V159新数值v4注册封存\n\n'+lead+'当前方向v159-numeric-v4-current-review，训练交付仍v158-delivery；旧成本不重置。\n',encoding='utf-8')
    tests=mutable[3].read_text(encoding='utf-8').replace("'v159-real-preflight-failure'","'v159-numeric-v4-current-review'").replace('"v159-real-preflight-failure"','"v159-numeric-v4-current-review"')
    extra='''    def test_v159_numeric_v4_conserves_history_and_training_caps(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        plan=json.loads(readonly._document_text(entries['v159-numeric-v4-contract']))
        cases=json.loads(readonly._document_text(entries['v159-numeric-v4-protocol-cases']))
        self.assertEqual(plan['historical_technical_cost']['classifier_forward_chunks'],124)
        self.assertEqual(plan['historical_technical_cost']['full_class_gradients'],8)
        self.assertEqual(plan['total_cumulative_caps']['classifier_forward_chunks'],78196)
        self.assertEqual(plan['total_cumulative_caps']['full_class_gradients'],2420)
        self.assertEqual(plan['total_future_caps']['fit_full_class_gradients'],2400)
        self.assertEqual(plan['total_future_caps']['fits'],6)
        self.assertEqual(plan['numeric_repeat_policy']['repeat_eps'],8)
        self.assertFalse(plan['numeric_repeat_policy']['Armijo_relaxation'])
        self.assertEqual(len(cases['cases']),20)
        self.assertTrue(all(cases['cases'].values()))

'''
    needle='    def test_non_loopback_binding_requires_token(self) -> None:\n';assert needle in tests;mutable[3].write_text(tests.replace(needle,extra+needle),encoding='utf-8')
    check_bindings(bindings)
    (OUT/'publication.json').write_text(json.dumps(dict(status='new_numeric_v4_registered_materials_published_before_new_preflight',catalog_bytes=len(raw),documents=published,actual_new_registration_seal=sha(trial/'run_seal.json'),historical_head_calls=124,historical_full_class_gradients=8,new_official_heads=0,new_official_gradients=0,new_official_fits=0,new_official_updates=0,quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+mutable}),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='numeric_v4_registered_published',catalog_bytes=len(raw))))
if __name__=='__main__':main()
