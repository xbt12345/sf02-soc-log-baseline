"""Publish prospectively sealed trial and real partial compute, not quality."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha

OUT=ROOT/'artifacts/v158_execution_publication_20261001'
TRIAL=ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists();OUT.mkdir()
    paths=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for p in paths:
        dst=OUT/'previous_publication_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    cat=read(paths[2]);assert cat['project']['authoritative_delivery_id']=='v155-delivery'
    # Independent reports can change in their owning chat. Serve this actual
    # publication's physical snapshots instead of a stale hash on a mutable file.
    snapshots=[]
    for e in cat['documents']:
        p=ROOT/e['path']
        if p.is_file() and sha(p)!=e['sha256']:
            if not (e['path'].startswith('docs/') and 'INDEPENDENT' in p.name):raise ValueError('Unexpected catalog source drift '+e['path'])
            dest=OUT/'independent_reports_at_publication'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            snapshots.append(dict(id=e['id'],old_expected_sha256=e['sha256'],current_actual_sha256=sha(dest),original_path=e['path'],snapshot_path=dest.relative_to(ROOT).as_posix()))
            e['path']=dest.relative_to(ROOT).as_posix();e['sha256']=sha(dest)
    fits=[read(p) for p in TRIAL.rglob('fit.json')]
    actual=dict(status='V158_source_excluded_supervision_generation_in_progress_not_task_delivery',
        completed_new_fits=len(fits),completed_batch_gradients=sum(z.get('batch_gradients',0) for z in fits),
        completed_dense_full_gradients=sum(z.get('full_gradients',0) for z in fits),
        completed_updates=sum(z.get('accepted_updates',0) for z in fits),
        completed_classifier_forward_calls=sum(z.get('classifier_forward_calls',0) for z in fits),
        official_current_pipeline_fits_max=45,legacy_N1_fits_max=9,fusion_fits_max=6,total_fits_max=60,
        failed_logger_after_completed_fit_preserved=True,no_completed_fit_repeated=True,quality_acceptance=False,model_promoted=False,
        scope='Immutable publication snapshot; live fits may progress. Only completed receipts counted here, active attempts remain in classifier/gradient journals.',
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [TRIAL/'registration.json',TRIAL/'registration_v2.json',TRIAL/'run_seal.json',TRIAL/'run_seal_v2.json',TRIAL/'execution_failure.json']+list(TRIAL.rglob('fit.json'))})
    save(OUT/'actual_training_snapshot.json',actual);save(OUT/'independent_report_snapshot_manifest.json',dict(snapshots=snapshots))
    pr=cat['project'];pr['authoritative_direction_id']='v158-nested-fusion-execution'
    pr['current_summary']=f"V158正式留根监督生成已启动：本次发布时已完成{len(fits)}个新拟合、{actual['completed_batch_gradients']}batch梯度、{actual['completed_dense_full_gradients']}dense完整梯度、{actual['completed_updates']}更新；余下活动成本以逐次日志为准。17专家库总上限60拟合（45当前链+9完整N1+6融合），多入口和完整物理数据已封存，日志故障另版接续不重跑/不重置预算。尚无V158完整任务验收。已有任务交付V155六拟合完成：1800梯度/1806proposal/1200更新，全2056871行质量失败、未晋升；第二第三及完整目标仍active。"
    pr['current_direction']=['V158在合法原频次与完整根排除下生成54个监督阶段，旧51仅为已完成初资格。','完成实际新OOF来源核对后，另封存六个概率融合A/B入口；不自动追加确认。','初始化先验只保证合法FIT原点，OVA softmax不宣称校准，32维文本投影非无损。','保护联合TRAIN能力；以完整2056871行三分类、来源、小组和匹配实际结果判断质量。']
    pr['known_limits'].append('V158仍为此前已查看开发来源；新OOF没有创造新独立环境。已开始真实拟合，不是零拟合方案；正式任务质量尚未计算。')
    entries=[('v158-nested-fusion-execution','V158完整留根融合正式60拟合与执行边界',ROOT/'docs/V158_COMPLETE_NESTED_TRIAL_DESIGN_AND_EXECUTION.md','current_review'),
        ('v158-complete-plan-v2','V158正式60拟合机器方案及日志接续',ROOT/'training/review_policy/v158_complete_nested_trial_v2_plan.json','review_evidence'),
        ('v158-registration-v2','V158新入口封存、已消费1拟合与剩59',TRIAL/'registration_v2.json','review_evidence'),
        ('v158-actual-training-snapshot','V158发布时真实拟合成本与未验收边界',OUT/'actual_training_snapshot.json','review_evidence'),
        ('v158-logging-retry','V158完成拟合后的日志故障与成本保留',ROOT/'docs/V158_COMPLETED_FIT_LOGGING_TECHNICAL_RETRY.md','review_evidence'),
        ('v158-legacy-bank-qualification','V158真正全任务N1专家与合法初始化资格',ROOT/'artifacts/v158_legacy_expert_bank_qualification_20261001/qualification.json','review_evidence'),
        ('v158-independent-legacy-prior','V158父独立全原行角色与合法间隔先验复核',ROOT/'artifacts/v158_independent_legacy_role_and_prior_audit_20261001/audit.json','review_evidence')]
    for ident,title,p,category in entries:
        assert p.is_file() and not any(e['id']==ident for e in cat['documents'])
        cat['documents'].insert(0,dict(id=ident,title=title,path=p.relative_to(ROOT).as_posix(),category=category,summary=title,sha256=sha(p),keywords=['V158','当前','训练','留根','融合']))
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024;paths[2].write_bytes(raw)
    s=paths[0].read_text(encoding='utf-8');title,rest=s.split('\n',1);rest=rest.replace('当前方向（V157实测）','历史诊断（V157实测）')
    paths[0].write_text(title+'\n\n当前执行（V158新监督生成）：[正式60拟合与执行边界](docs/V158_COMPLETE_NESTED_TRIAL_DESIGN_AND_EXECUTION.md)；[日志错误及已完成拟合接续](docs/V158_COMPLETED_FIT_LOGGING_TECHNICAL_RETRY.md)。17专家库=45当前完整链+9全任务N1+6融合；新物理封存与真实训练已开始，完整任务质量尚未验收。原第一项100轮/6300更新完成后日志故障保留，新入口不重跑已完成项、不重置60预算。已有完整任务交付V155质量失败、未晋升；完整目标active。\n\n'+rest.lstrip('\n'),encoding='utf-8')
    paths[1].write_text(paths[1].read_text(encoding='utf-8')+'\n\n## V158实际启动与接续\n\n正式plan `training/review_policy/v158_complete_nested_trial_v2_plan.json`，实际目录 `artifacts/v158_current_pipeline_OOF_trial_20261001`。原第一项6300更新已完成并保留；原session73917已terminal1（日志重复stage），新入口只修日志，原seal/source未改、60拟合总预算不重置。新执行 `v158_nested_base_train_v2.py all-base`，完成45当前链后执行 `v158_legacy_nested_train_v2.py all-legacy` 九全任务N1。每条真实角色均排除inner/outer根，后六融合须实际OOF来源核对和独立入口封存。没有V158完整任务交付/质量通过，V155交付保留。当前专家库方向60实际成本，不再称51/0拟合；完整目标active。\n',encoding='utf-8')
    s=paths[3].read_text(encoding='utf-8').replace('v157-function-review','v158-nested-fusion-execution').replace('self.assertIn("V157", json.dumps(fetch_result.structured_content, ensure_ascii=False))','self.assertIn("V158", json.dumps(fetch_result.structured_content, ensure_ascii=False))')
    paths[3].write_text(s,encoding='utf-8')
    save(OUT/'publication.json',dict(status='V158_actual_partial_training_and_60_fit_execution_direction_published',catalog_bytes=len(raw),
        authoritative_task_delivery='v155-delivery',quality_acceptance=False,current_direction='v158-nested-fusion-execution',
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),OUT/'actual_training_snapshot.json']+paths}))
    print(dict(published=True,catalog_bytes=len(raw),completed_fits_at_snapshot=len(fits),quality_acceptance=False))

if __name__=='__main__':main()
