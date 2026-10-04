"""Publish completed independent review and actual OOF counterexamples."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v158_independent_oof_review_publication_20261001'
PARENT=ROOT/'artifacts/v158_independent_fusion_result_audit_v2_20261001'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists();OUT.mkdir()
    paths=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for p in paths:
        dest=OUT/'previous_publication_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    report=ROOT/'docs/V158_INDEPENDENT_FINAL_REVIEW_AND_NEXT_TRAINING_DECISION.md'
    case=ROOT/'training/review_policy/v158_oof_supervision_and_support_cases.json'
    replay=ROOT/'artifacts/v158_oof_supervision_case_replay_20261001/replay.json'
    bound_paths=[Path(__file__).resolve(),report,case,replay,PARENT/'audit.json',PARENT/'OOF_saved_checkpoint_learning.json',
        PARENT/'OOF_supervision_tradeoff_and_support_counterexamples.json',ROOT/'artifacts/v158_fusion_trial_20261001/final_delivery.json']
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in bound_paths}
    save(OUT/'pre_publication_bindings.json',dict(status='bound_before_independent_result_publication',source_sha256=bound))
    r=read(replay);assert r['OOF_all_classes_mastered'] is False and r['deployment_FIT_retained'] is True
    assert r['own_fits']==r['own_classifier_calls']==r['own_gradients']==0
    d=read(bound_paths[-1]);assert d['classifier_fits']==60 and d['latest_actual']=='V158' and not d['quality_acceptance']
    cat=read(paths[2]);assert cat['project']['authoritative_delivery_id']=='v158-delivery'
    assert cat['project']['authoritative_direction_id']=='v158-result-review'
    for entry in cat['documents']:assert sha(ROOT/entry['path'])==entry['sha256'],entry['id']
    snapshot=OUT/'independent_final_report_snapshot'/report.relative_to(ROOT);snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(report,snapshot)
    specs=[
        ('v158-independent-final-review','V158独立最终结果与下一训练资格要求',snapshot,'current_review'),
        ('v158-oof-class-support-replay','V158完整OOF类牺牲与支持粒度真实回放',replay,'review_evidence'),
        ('v158-independent-oof-checkpoints','V158父独立OOF固定检查点逐类学习',PARENT/'OOF_saved_checkpoint_learning.json','review_evidence'),
        ('v158-independent-oof-support-cases','V158父独立OOF类别交换及完整固定支持反例',PARENT/'OOF_supervision_tradeoff_and_support_counterexamples.json','review_evidence')]
    for ident,title,p,category in specs:
        assert p.stat().st_size<=256*1024 and not any(e['id']==ident for e in cat['documents'])
        cat['documents'].insert(0,dict(id=ident,title=title,path=p.relative_to(ROOT).as_posix(),category=category,summary=title,sha256=sha(p),keywords=['V158','OOF','监督','分类','支持','下一资格']))
    pr=cat['project'];pr['authoritative_direction_id']='v158-independent-final-review'
    pr['current_summary']='V158完成60次真实拟合及2056871原行验收，融合1200完整梯度/1200proposal/1200更新。候选ASA414M/1872S，匹配及完整质量失败、未晋升。部署FIT联合TRAIN保持，不代表OOF监督掌握：B角色0/1总CE降但S的CE和错同时升，角色2两类改善。全库148M/979S受固定凸族限制，余266M/893S有正确专家。旧配置停止；下一项只做完整输入、旧方法差异、逐类OOF和来源隔离资格审查，新正式拟合0，完整目标active。'
    pr['current_direction']=['V15860次登记拟合已完成且质量失败，不追加当前凸融合、不晋升A或早期状态。',
        '下一项只审查一个完整行为输入与交叉预测类别判别函数，核对真实输入冲突及V128/V110历史差异；尚无正式拟合资格。',
        '合法OOF逐类风险/分类和部署FIT能力分开验证，保留角色0/1退化、角色2改善及精细/目的端/粗支持反例。',
        '复用54现成合法OOF；若做元头内部留根验证，底层专家和全部守卫也必须排除验证根。仅排除元头梯度不能称独立。',
        '固定终点且不依内部标签选模时，OOF仅作训练诊断、现外折仅作已查看开发评价；未来额外人口/调用成本必须事前登记。',
        '保护既有225558正确TRAIN角色原行，第二第三及完整目标仍active；当前新拟合/模型调用/梯度/更新均0。']
    pr['known_limits'].append('V158部署FIT掌握不证明合法OOF各类已学会；未来独立元验证不能复用已见验证根标签的基模型/守卫。')
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024;paths[2].write_bytes(raw)
    top=paths[0].read_text(encoding='utf-8');needle='当前实际训练与完整验收（V158）：'
    assert needle in top
    insert='当前证据与下一资格：[V158独立最终结果](docs/V158_INDEPENDENT_FINAL_REVIEW_AND_NEXT_TRAINING_DECISION.md)。部署FIT保护通过，但B的OOF角色0/1在总CE下降时S损失及错误上升；角色2两类改善，全部保留。下一项仅做完整输入/历史方法差异/逐类OOF监督/来源隔离资格审查，新正式拟合0，不追加当前配置。\n\n'
    top=top.replace(needle,insert+needle,1);paths[0].write_text(top,encoding='utf-8')
    paths[1].write_text(paths[1].read_text(encoding='utf-8')+'''\n\n## V158独立最终补充与OOF监督反例\n\n`docs/V158_INDEPENDENT_FINAL_REVIEW_AND_NEXT_TRAINING_DECISION.md`已冻结并保存物理发布快照；旧进展文和正式交付哈希未改。新增父凭据 `OOF_saved_checkpoint_learning.json` 与 `OOF_supervision_tradeoff_and_support_counterexamples.json`，执行方 `training/v158_replay_oof_supervision_cases.py`已实际exit0：完整225614 OOF角色原行，六终点逐类重算一致；B角色0/1的S错+50/+30、S CE上升，角色2 M/S错-22/-38且两类CE下降。全部部署FIT保持只证明另一人口，不能覆盖OOF。

979共同失败S中精细同类支持36、目的端126、粗行为979；893个尚有正确专家的S残错中分别58/276/849，涵盖161来源；578困难S分别0/160/578。这些是支持粒度反例，不证明因果。新历史文件 `training/review_policy/v158_oof_supervision_and_support_cases.json`，0新模型/特征/梯度/拟合/更新，正式新拟合资格False。

只读MCP当前方向 `v158-independent-final-review`，最新实际交付仍 `v158-delivery`。下一设计要求尚未获得执行资格；禁止重跑54项现成人口或追加本凸配置。若新增元头内部来源验证，留根标签必须从底层特征生成、元梯度和全部守卫排除；现缓存未必满足更深验证。固定终点不依内部选模时，不虚称OOF独立验证或现外折盲测。完整目标active。\n''',encoding='utf-8')
    tests=paths[3].read_text(encoding='utf-8').replace('v158-result-review','v158-independent-final-review')
    # Also fetch and check the actual OOF qualification case, rather than
    # making MCP success depend only on a narrative status string.
    addition='''    def test_oof_supervision_is_not_overridden_by_fit_retention(self) -> None:
        catalog = readonly._load_catalog()
        entries = readonly._documents_by_id(catalog)
        evidence = json.loads(readonly._document_text(entries['v158-oof-class-support-replay']))
        self.assertEqual(evidence['original_OOF_role_rows'], 225614)
        self.assertTrue(evidence['deployment_FIT_retained'])
        self.assertFalse(evidence['OOF_all_classes_mastered'])
        self.assertFalse(evidence['new_formal_fit_qualification'])
        arms = [z for z in evidence['class_tradeoff_cases'] if z['arm'] == 'B']
        self.assertEqual([z['S_error_delta'] for z in arms], [50, 30, -38])
        self.assertTrue(all(z['total_CE_decreased'] for z in arms))
        self.assertLess(arms[2]['M_error_delta'], 0)
        self.assertEqual(evidence['own_fits'], 0)

'''
    needle='    def test_non_loopback_binding_requires_token(self) -> None:\n';assert needle in tests
    paths[3].write_text(tests.replace(needle,addition+needle),encoding='utf-8')
    check_bindings(bound)
    save(OUT/'publication.json',dict(status='V158_independent_final_and_actual_OOF_supervision_cases_published',
        catalog_bytes=len(raw),authoritative_delivery='v158-delivery',authoritative_direction='v158-independent-final-review',
        official_new_fits=0,official_new_classifier_calls=0,official_new_gradients=0,official_new_updates=0,
        quality_acceptance=False,new_formal_fit_qualification=False,goal_status='active',
        independent_report_original_path=report.relative_to(ROOT).as_posix(),independent_report_snapshot_path=snapshot.relative_to(ROOT).as_posix(),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),snapshot,case,replay]+paths}))
    print(dict(status='published_actual_OOF_gaps_not_new_training',catalog_bytes=len(raw),new_formal_fits=0))

if __name__=='__main__':main()
