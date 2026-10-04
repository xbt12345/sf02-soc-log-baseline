"""Publish the completed failed trial without changing sealed training bytes."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha

OUT=ROOT/'artifacts/v158_complete_result_publication_20261001'
TRIAL=ROOT/'artifacts/v158_fusion_trial_20261001'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists();OUT.mkdir()
    paths=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for p in paths:
        dest=OUT/'previous_publication_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    # Publication files must never be prospective model/data dependencies.
    seals=[ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001/run_seal.json',
        ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001/run_seal_v2.json',TRIAL/'run_seal.json']
    for seal in seals:
        bound=read(seal)['source_sha256'];assert not any(p.relative_to(ROOT).as_posix() in bound for p in paths)
    d=read(TRIAL/'final_delivery.json');q=read(TRIAL/'quality.json');assert d['classifier_fits']==60 and not d['quality_acceptance'] and not d['model_promoted']
    cat=read(paths[2]);assert cat['project']['authoritative_delivery_id']=='v155-delivery'
    snapshots=[]
    for entry in cat['documents']:
        p=ROOT/entry['path']
        if sha(p)!=entry['sha256']:
            assert entry['path'].startswith('docs/') and 'INDEPENDENT' in p.name,entry['path']
            dest=OUT/'independent_report_snapshots'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            snapshots.append(dict(id=entry['id'],original_path=entry['path'],previous_sha256=entry['sha256'],snapshot_path=dest.relative_to(ROOT).as_posix(),actual_sha256=sha(dest)))
            entry['path']=dest.relative_to(ROOT).as_posix();entry['sha256']=sha(dest)
    specs=[
        ('v158-result-review','V158完整60拟合、全任务验收失败与停止决定','docs/V158_COMPLETE_TRAINING_RESULTS_AND_DECISION.md','current_review'),
        ('v158-delivery','V158最新实际训练及完整任务交付','artifacts/v158_fusion_trial_20261001/final_delivery.json','execution_delivery'),
        ('v158-full-verification','V158六模型真实末五重放与完整原行验证','artifacts/v158_fusion_trial_20261001/verification.json','review_evidence'),
        ('v158-quality','V158完整三分类、来源与匹配门槛','artifacts/v158_fusion_trial_20261001/quality.json','review_evidence'),
        ('v158-complete-costs','V158实际60拟合与185522分类器分块调用','artifacts/v158_full_result_records_20261001/complete_actual_costs.json','review_evidence'),
        ('v158-transfer-diagnosis','V158零步与实际学习分解，零新模型求值','artifacts/v158_saved_result_transfer_audit_20261001/audit.json','review_evidence'),
        ('v158-case-replay','V158真实类别交换与固定凸专家限制反例回放','artifacts/v158_observed_fusion_case_replay_20261001/replay.json','review_evidence'),
        ('v158-bank-final-qualification','V158全部54实际监督及合法OOF装配核验','artifacts/v158_legal_fusion_bank_v2_20261001/qualification.json','review_evidence'),
        ('v158-fusion-contract','V158固定六融合真实执行契约','training/review_policy/v158_fusion_execution_contract.json','review_evidence'),
        ('v158-zero-update-audit','V158四项零接受更新真实参数与概率身份','artifacts/v158_zero_update_stage_identity_audit_20261001/audit.json','review_evidence'),
        ('v158-independent-current-final','V158父独立全部45链阶段复核','artifacts/v158_independent_completed_fit_audit_20261001_03/audit.json','review_evidence'),
        ('v158-independent-legacy-final','V158父独立九全任务N1真实拟合复核','artifacts/v158_independent_legacy_fit_audit_20261001/audit.json','review_evidence'),
        ('v158-independent-fusion-final','V158父独立六融合、原始gold与来源退化复核','artifacts/v158_independent_fusion_result_audit_v2_20261001/audit.json','review_evidence'),
        ('v158-independent-plan-progress','V158父独立原方法与监督生成进展边界','docs/V158_INDEPENDENT_TRAINING_DECISION_AND_PROGRESS.md','review_evidence')]
    specs.extend([
        ('v158-whole-bank-supplement','V158全库148M/979S凸族限制与起点学习分解','docs/V158_WHOLE_BANK_CAPACITY_AND_ORIGIN_SUPPLEMENT.md','review_evidence'),
        ('v158-whole-bank-case-replay','V158完整ASA固定17专家可表达限制真实回放','artifacts/v158_full_bank_capacity_case_replay_20261001/replay.json','review_evidence')])
    # Independent reports remain owned by their writing chat. Serve a physical
    # publication snapshot even if the source report changes afterwards.
    candidates=list(ROOT.joinpath('docs').glob('V158*INDEPENDENT*RESULT*.md'))
    for p in candidates:specs.append(('v158-independent-result-'+p.stem.lower(),'V158父独立最终结果补充',p.relative_to(ROOT).as_posix(),'review_evidence'))
    for ident,title,rel,category in specs:
        p=ROOT/rel;assert p.is_file() and not any(e['id']==ident for e in cat['documents'])
        if rel.startswith('docs/') and 'INDEPENDENT' in p.name:
            dest=OUT/'independent_report_snapshots'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            snapshots.append(dict(id=ident,original_path=rel,snapshot_path=dest.relative_to(ROOT).as_posix(),actual_sha256=sha(dest)));p=dest
        assert p.stat().st_size<=256*1024
        cat['documents'].insert(0,dict(id=ident,title=title,path=p.relative_to(ROOT).as_posix(),category=category,summary=title,sha256=sha(p),keywords=['V158','实际','验收','分类','来源']))
    pr=cat['project'];pr['authoritative_delivery_id']='v158-delivery';pr['authoritative_direction_id']='v158-result-review'
    pr['current_summary']='V158完成60次真实拟合（45当前完整链+9全任务N1+6融合），全部6终点/末五及2056871原行验收完成。融合1200完整梯度/1200proposal/1200更新；全阶段分类器分块调用185522。A的ASA M/S错424/1862，登记B为414/1872；B修复10M却新增10S（根29），总错2286持平，S逐类与多折失败。旧联合TRAIN保护通过，完整三分类/来源质量失败、未晋升。本配置停止，不追加拟合或确认；第二第三及完整目标active。'
    pr['current_direction']=['V158本配置60拟合已全部消耗，停止同配置追加，不挑A或中间点晋升。',
        '保留零步先验引起的变化和后续学习变化；原频次、未知/ICMP/混标、578困难与51正确对照全部计分。',
        '全ASA的148M/979S全部17专家真类间隔为负，无法由非负凸组合修复；其余266M/893S虽有正确专家，也不能从已看外层真值写路由。',
        '继续保护225558正确TRAIN角色记录并联合旧范围；完整目标仍active，当前没有新正式拟合登记。']
    pr['known_limits']=[s for s in pr['known_limits'] if not s.startswith('V158')]
    pr['known_limits'].extend(['V158仍是此前已查看的开发来源，未取得独立外部环境验收。',
        'V158正文32维CountSketch有损，A/B容量208/8640不同，旧OVA softmax不宣称校准。',
        'V158 N1没有每接受状态完整类别/来源轨迹；当前22/6/28保护不是追加N1分数后22/4/26完整可观测下限的学习证明。'])
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024;paths[2].write_bytes(raw)
    text=paths[0].read_text(encoding='utf-8');title,rest=text.split('\n',1)
    rest=rest.replace('当前执行（V158新监督生成）','历史执行快照（V158监督生成初期）')
    top='当前实际训练与完整验收（V158）：[60拟合结果与停止决定](docs/V158_COMPLETE_TRAINING_RESULTS_AND_DECISION.md)。全部60拟合完成，六终点及末五真实重放、2056871原行评分完成；A的ASA M/S错424/1862，B414/1872，匹配和全任务质量失败、未晋升。旧联合TRAIN能力保持。本配置停止，不追加拟合或确认；第二第三及完整目标active。零步和后续学习退化分别保留，历史执行快照不代表当前进度。'
    paths[0].write_text(title+'\n\n'+top+'\n\n'+rest.lstrip('\n'),encoding='utf-8')
    paths[1].write_text(paths[1].read_text(encoding='utf-8')+'\n\n## V158完整执行与失败验收（2026-10-01）\n\n全部45当前链、9全任务N1、6融合共60拟合已完成。训练/evaluation session73917（日志故障）、70636、85434、58397、57517均terminal；无本轮活动训练进程。实际融合1200梯度/1200proposal/1200更新，全阶段185522分类器分块调用；当前39,381更新、N1 885接受迭代另列。正式训练源码/方案/seal未改、第一完整拟合未重复。四个零接受阶段经模型张量/概率身份核验，允许不存在接受轨迹；装配原故障保留。\n\n最新实际完整交付 `artifacts/v158_fusion_trial_20261001/final_delivery.json`，对应 `verification.json`、`quality.json`、完整 `full_prediction_ledger.parquet`；报告 `docs/V158_COMPLETE_TRAINING_RESULTS_AND_DECISION.md`。A ASA424M/1862S，B414M/1872S，匹配S保护和多折失败、完整任务质量失败，未晋升。全部旧联合TRAIN保护通过，不能当成第二第三或完整目标完成。零步M1720/S1658到B修复1306M、新增214S；578困难仍全错、51正确对照错49（21零步+28后续净新增）。本配置停止，无额外同配置拟合、阈值/份额扫描或确认。\n\n只读MCP最新训练/任务交付 `v158-delivery`，当前结果 `v158-result-review`；旧V155及V158一拟合快照保留为历史。真实反例 `training/review_policy/v158_observed_fusion_cases.json` 和 `training/v158_replay_observed_cases.py` 已实际exit0，0新模型/梯度/拟合/更新。完整目标仍active；未来新的训练必须另有合法机制资格，不改本轮失败门槛、路由或封存字节。\n',encoding='utf-8')
    tests=paths[3].read_text(encoding='utf-8').replace('v158-nested-fusion-execution','v158-result-review')
    tests=tests.replace("self.assertIn('v155-delivery', status.source_ids)","self.assertIn('v158-delivery', status.source_ids)")
    tests=tests.replace("self.assertIn('V155六拟合完成', status.current_summary)","self.assertIn('V158完成60次真实拟合', status.current_summary)")
    tests=tests.replace("self.assertIn('1800梯度', status.current_summary)","self.assertIn('1200完整梯度', status.current_summary)")
    tests=tests.replace("self.assertIn('1806proposal', status.current_summary)","self.assertIn('1200proposal', status.current_summary)")
    tests=tests.replace("self.assertEqual(catalog['project']['authoritative_delivery_id'], 'v155-delivery')","self.assertEqual(catalog['project']['authoritative_delivery_id'], 'v158-delivery')")
    # The historical zero-fit design and V155 receipt retain their own checks.
    needle="        self.assertFalse(actual['quality_acceptance'])\n"
    addition="        newest = json.loads(readonly._document_text(readonly._documents_by_id(catalog)['v158-delivery']))\n        self.assertEqual(newest['classifier_fits'], 60)\n        self.assertEqual(newest['latest_actual'], 'V158')\n        self.assertFalse(newest['quality_acceptance'])\n        self.assertTrue(newest['TRAIN']['B']['all_roles_mastered'])\n        self.assertEqual(newest['original_pairs']['B_vs_A']['2']['new_errors'], 10)\n"
    assert needle in tests;tests=tests.replace(needle,needle+addition);paths[3].write_text(tests,encoding='utf-8')
    save(OUT/'independent_snapshot_manifest.json',dict(snapshots=snapshots))
    save(OUT/'publication.json',dict(status='actual_completed_60_fit_failed_quality_and_current_MCP_authority_published',
        authoritative_delivery='v158-delivery',authoritative_direction='v158-result-review',catalog_bytes=len(raw),
        quality_acceptance=False,model_promoted=False,goal_status='active',
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),TRIAL/'final_delivery.json',OUT/'independent_snapshot_manifest.json']+paths}))
    print(dict(status='published_completed_actual_V158_not_promoted',catalog_bytes=len(raw),independent_result_reports=len(candidates)))

if __name__=='__main__':main()
