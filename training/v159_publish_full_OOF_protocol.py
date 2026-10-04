"""Publish final pre-call protocol/caps; actual delivery remains V158."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v159_full_OOF_protocol_publication_20261002'

def main():
    assert not OUT.exists();OUT.mkdir()
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for p in mutable:
        dest=OUT/'previous_publication_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    specs=[
        ('v159-full-oof-window-review','V159正式调用前全OOF窗口重放及78072预算','docs/V159_FULL_OOF_WINDOW_EXECUTION_SUPPLEMENT.md','current_review'),
        ('v159-current-execution-contract','V159当前真实v3入口及全OOF窗口封存合同','training/review_policy/v159_boundary_execution_contract_v3.json','current_plan'),
        ('v159-current-entry-qualification','V159当前审查协议与真实v3入口资格未执行','artifacts/v159_execution_protocol_contract_v3_20261002/qualification.json','current_evidence'),
        ('v159-protocol-rejection-cases','V159新有限协议15个实际正反例','artifacts/v159_protocol_synthetic_qualification_v2_20261002/qualification.json','review_evidence'),
        ('v159-csr-identity','V159当前CSR规范无重复无显式零身份','artifacts/v159_saved_CSR_identity_audit_20261002/audit.json','review_evidence'),
        ('v159-independent-targeted-plan','V159父独立三层验收与单候选目标','docs/V159_LAST_ROUND_DECISION_AND_TARGETED_PLAN_20261002.md','review_evidence'),
        ('v159-actual-oof-window-evaluate-source','V159真实完整OOF及部署末五窗口验收源码','training/v159_boundary_evaluate_v3.py','review_evidence')]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+[ROOT/s[2] for s in specs]}
    (OUT/'pre_publication_bindings.json').write_text(json.dumps(dict(status='bound_before_final_pre_call_protocol_publication',source_sha256=bound),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=read(ROOT/specs[1][2]);assert p['total_future_caps']['classifier_forward_chunks']==p['total_future_caps']['opinion_feature_blocks']==78072 and p['total_future_caps']['evaluation_classifier_forward_chunks']==720
    assert not (ROOT/'artifacts/v159_class_boundary_trial_20261002/run_seal.json').exists()
    cat=read(mutable[2]);assert cat['project']['authoritative_delivery_id']=='v158-delivery' and cat['project']['authoritative_direction_id']=='v159-execution-retention-review'
    for e in cat['documents']:assert sha(ROOT/e['path'])==e['sha256'],e['id']
    # Redundant summaries repeat an unchanged searchable title; preserve IDs,
    # titles, keywords, paths, content and hashes while freeing catalog space.
    trimmed=0
    for e in cat['documents']:
        if e.get('summary')==e.get('title'):del e['summary'];trimmed+=1
    published=[]
    for ident,title,name,category in specs:
        source=ROOT/name;dest=OUT/'physical_source_snapshots'/name
        if dest.suffix=='.py':dest=dest.with_suffix('.py.txt')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);assert sha(source)==sha(dest) and dest.stat().st_size<=256*1024
        assert not any(e['id']==ident for e in cat['documents'])
        cat['documents'].insert(0,dict(id=ident,title=title,path=dest.relative_to(ROOT).as_posix(),category=category,sha256=sha(dest),keywords=['V159','当前入口','完整OOF重放','协议','78072','未执行']))
        published.append(dict(id=ident,original=name,snapshot=dest.relative_to(ROOT).as_posix(),sha256=sha(dest)))
    project=cat['project'];project['authoritative_direction_id']='v159-full-oof-window-review'
    project['current_summary']='V158完成60次真实拟合及2056871原行验收，融合1200完整梯度/1200proposal/1200更新。ASA414M/1872S，完整质量失败、未晋升。V159唯一当前头v3、真实train/evaluate v3、版本化审查seal/require/checkpoint、完整CSR和兼容OOF/部署保护已绑定；新增最后五真实OOF无梯度重放，未来6fit/2412完整类梯度/78072头前向及意见块(336+77016+720)。15个协议反例/小正例实测通过；当前官方新函数/特征/梯度/fit/更新0，真实run seal/零步/fit未执行，目标active。'
    project['current_direction']=[
        '正式当前合同v159_boundary_execution_contract_v3.json，唯一头v159_current_input_boundary_v3.py、入口train_v3/evaluate_v3，版本化experiment_review_v159_v2保留所有历史源码。',
        '实际新protocol15个正反例通过，seal/require/endpoint真实接入；未封存/错entry/源变更/漏行/类牺牲/后验终点均拒绝。',
        'CSR规范、无重复/显式零，数值键与独立审查同义；已验收部署225558全正确及全部联合能力不变，B只保护纯当前X正确OOF组，混类仍原频loss/质量。',
        '每个最后五真实状态及终点同时真实重放全OOF和全部署；只读旧OOF账本不算模型重放，不足五状态不凑满，评价不计算梯度。',
        '注册前扩大真实评价覆盖，预算改78072 head/opinion、2412完整类梯度、6fit、3600累计proposal、1200接受；旧77832合同/快照不改，没有新增训练预算。',
        '独立复核后进行actual register/全physical seal/336头前向与12完整类梯度preflight，再固定六fit和全2056871验收；当前官方新函数/fit/梯度/更新0。',
        '三层验收分列，现外折为已查看开发；N无新监督不能申报新N学习。第一训练问题、第二支持、第三迁移及完整目标仍未关闭。']
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024;mutable[2].write_bytes(raw)
    mutable[0].write_text('当前正式协议和完整重放：[V159全OOF窗口执行补充](docs/V159_FULL_OOF_WINDOW_EXECUTION_SUPPLEMENT.md)。唯一头未改，真实最后五OOF验证已纳入当前v3入口；未来head/意见预算78072，fit6/完整类梯度2412。CSR与协议反例实测通过，官方新调用/fit0；最新真实V158质量失败。\n\n'+mutable[0].read_text(encoding='utf-8'),encoding='utf-8')
    mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V159当前全OOF窗口重放协议\n\n`docs/V159_FULL_OOF_WINDOW_EXECUTION_SUPPLEMENT.md` 和 `training/review_policy/v159_boundary_execution_contract_v3.json` 是当前登记源，真正train/evaluate/runtime v3调用 `experiment_review_v159_v2` 版本协议；旧模块/合同/保存输入准备不改。每个实际末五模型全OOF+部署均真正前向，预算改78072/78072头/意见块、2412完整类梯度、6fit、3600累计proposal不变。15协议合成案例和CSRcanonical/duplicates/zero实测通过。当前官方新函数/特征/梯度/fit/更新0，无actual run seal/preflight/fit。只读MCP方向v159-full-oof-window-review，真实交付仍v158-delivery，三个目标active。\n',encoding='utf-8')
    tests=mutable[3].read_text(encoding='utf-8').replace('v159-execution-retention-review','v159-full-oof-window-review')
    addition='''    def test_v159_current_plan_replays_actual_OOF_windows_before_calls(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        contract=json.loads(readonly._document_text(entries['v159-current-execution-contract']))
        proof=json.loads(readonly._document_text(entries['v159-protocol-rejection-cases']))
        caps=contract['total_future_caps']
        self.assertEqual((caps['classifier_forward_chunks'],caps['opinion_feature_blocks']),(78072,78072))
        self.assertEqual(caps['evaluation_classifier_forward_chunks'],720)
        self.assertEqual(caps['full_class_gradients'],2412)
        self.assertEqual(contract['activation_entries'],['training/v159_boundary_train_v3.py','training/v159_boundary_evaluate_v3.py'])
        self.assertTrue(contract['pre_registration_budget_refinement']['all_last5_actual_OOF_replay'])
        self.assertTrue(all(proof['cases'].values()))
        self.assertEqual(proof['official_classifier_calls'],0)

'''
    needle='    def test_non_loopback_binding_requires_token(self) -> None:\n';assert needle in tests;mutable[3].write_text(tests.replace(needle,addition+needle),encoding='utf-8')
    check_bindings(bound)
    (OUT/'publication.json').write_text(json.dumps(dict(status='current_full_actual_OOF_window_protocol_and_caps_published_before_official_calls',catalog_bytes=len(raw),redundant_title_summaries_removed=trimmed,authoritative_delivery='v158-delivery',authoritative_direction='v159-full-oof-window-review',documents=published,
        official_classifier_calls=0,official_features_calls=0,official_gradients=0,official_fits=0,official_updates=0,official_runtime_sealed=False,official_zero_step_replay=False,quality_acceptance=False,goal_status='active',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+mutable}),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='current_full_OOF_protocol_published_no_actual_official_calls',catalog_bytes=len(raw),duplicate_summaries_removed=trimmed)))

if __name__=='__main__':main()
