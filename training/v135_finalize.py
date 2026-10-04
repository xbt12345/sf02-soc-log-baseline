"""Publish verified actual execution, preserving primary delivery and old bindings."""
import datetime,json
from pathlib import Path
import pandas as pd
from experiment_review import check_bindings
from v135_runtime import ROOT,OUT,ARMS,read,save,sha,require_run_seal

def main():
    require_run_seal(ROOT/'training/v135_train.py')
    for path in [ROOT/'artifacts/v134_stable_learning_trial_20260930/run_seal.json',ROOT/'artifacts/v131_learning_trial_20260930/run_seal.json',OUT/'postflight_audit/source_receipt.json',OUT/'error_review/source_receipt.json']:
        check_bindings(read(path)['source_sha256'])
    primary=read(OUT/'delivery.json');verified=read(OUT/'verification.json');audit=read(OUT/'postflight_audit/audit.json')
    confirm=read(OUT/'confirmation_decision.json');quality=read(OUT/'quality.json')
    records=read(OUT/'error_review/TRAIN_error_transitions.json')['records']
    if primary['classifier_fits']!=12 or primary['optimizer_steps']!=71200 or not audit['factorial_prefix_valid']:raise ValueError('Incomplete or unmatched primary')
    if confirm['fits']!=0 or confirm['updates']!=0 or primary['quality_acceptance']:raise ValueError('This failed-run closeout cannot promote models')
    fits=[read(OUT/f'fold{fold}_{arm}/fit.json') for fold in range(3) for arm in ARMS]
    train={a:{'pure_M_errors':sum(z['endpoint_training']['pure_M_errors'] for z in fits if z['arm']==a),
              'pure_S_errors':sum(z['endpoint_training']['pure_S_errors'] for z in fits if z['arm']==a),
              'stable_roles':sum(z['fixed_window_mastered'] for z in fits if z['arm']==a),
              'no_correct_member_endpoint_rows':sum(z['endpoint_no_correct_member_rows'] for z in records if z['arm']==a)} for a in ARMS}
    eout=OUT/'error_review';save(eout/'output_receipt.json',{'output_sha256':{p.name:sha(p) for p in eout.iterdir() if p.is_file()}})
    aborted=read(ROOT/'artifacts/v134_stable_learning_trial_20260930/interruption_receipt.json')
    final=dict(primary);final.update(status='v135_completed_learning_and_quality_failed',
        as_of=datetime.datetime.now().isoformat(),training_execution_completed=True,training_mastery_accepted=False,
        validation_scope='实际12次主拟合、71200步；90,245,600次TRAIN原行轮次记录、末五轮60个模型及4,512,280次原行重放、36,098,240次配对比较、2,056,871官方原行独立真值计数。已反复查看的来源开发折，无独立盲测、官方提交或外部迁移验收。',
        endpoint_TRAIN=train,confirmation_fits=0,confirmation_updates=0,
        current_request_attempts={'new_primary_fits_completed':12,'new_primary_updates':71200,
            'aborted_prior_fits_started':aborted['fits_started'],'aborted_prior_fits_completed':aborted['fits_completed'],
            'aborted_prior_updates_logged_lower_bound':aborted['optimizer_steps_logged'],
            'total_fits_started':12+aborted['fits_started'],'total_fits_completed':12+aborted['fits_completed'],
            'total_updates_logged_lower_bound':71200+aborted['optimizer_steps_logged'],'diagnostic_optimizer_updates':0},
        primary_training_seconds=sum(z['seconds'] for z in fits),
        evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [OUT/'delivery.json',OUT/'quality.json',OUT/'verification.json',
            OUT/'postflight_audit/audit.json',OUT/'postflight_audit/source_receipt.json',OUT/'postflight_audit/output_receipt.json',
            eout/'TRAIN_error_transitions.json',eout/'HELD_repair_and_regression.csv',eout/'source_receipt.json',eout/'output_receipt.json',
            OUT/'confirmation_decision.json',ROOT/'artifacts/v134_stable_learning_trial_20260930/interruption_receipt.json',Path(__file__)]})
    destination=OUT/'final_delivery.json'
    if destination.exists():raise FileExistsError('Preserve delivered evidence')
    save(destination,final)
    cases={'version':'V135_observed_execution_and_outcome_cases','new_updates':0,'model_promoted':False,
        'source_sha256':sha(destination),'cases':[
            {'id':'same_seed_is_not_a_matched_trajectory','evidence':'V134 same initial state, first10 epochs 24 prediction differences before schedule change.','action':'Repeated real forward/backward checks and exact first80 original-row prefix gate; unsupported determinism switch alone is insufficient.'},
            {'id':'stable_wrong_is_not_mastery','evidence':'R_decay fold0 last5 pure errors 4M/38S every epoch.','action':'Require correct last5 actual model decisions, not merely fewer flips or a zero endpoint elsewhere.'},
            {'id':'one_zero_epoch_is_not_stability','evidence':'O_const fold1 epoch96 pure0, epochs97-100 pure8M/0S.','action':'Replay all five registered window models and protect both classes.'},
            {'id':'training_improvement_does_not_protect_held_M','evidence':'O_const versus R_const TRAIN pure77 to52, HELD M2114 to2270 and S1639 to1627.','action':'Keep all original population and full class/source gates; do not increase class weights from S-only success.'},
            {'id':'existing_correct_member_is_not_an_available_oracle','evidence':'Every one of R_decay 80 pure TRAIN endpoint mistakes has at least one correct member.','action':'Separate member representation, confidence and aggregation. Never choose members per-row using true labels; no posthoc voting promotion.'},
            {'id':'gradient_opposition_is_not_a_unique_cause','evidence':'Fold1 R_decay epoch100 original facts_direct M/S gradient cosine -0.9998683333.','action':'Record class risks and gradient norms too; opposing class gradients can reflect equilibrium and do not alone justify gradient surgery.'},
            {'id':'protected_slice_failures_still_count','evidence':'R_decay header682 has12 errors versus A0 zero; root2868 M1040 errors versus A0 48.','action':'Keep hard slice gates. Header errors alone do not prove date dependence; require semantic-preserving input interventions before such a causal claim.'}]}
    save(ROOT/'training/review_policy/v135_observed_cases.json',cases)
    doc=ROOT/'docs/V135_TRAINING_EXECUTION_AND_RESULTS.md';text=doc.read_text(encoding='utf-8')
    text=text.replace('当前状态：实际训练进行中，尚未完成质量验收，无模型晋升。','当前状态：12次主拟合、71,200次参数更新全部完成；训练掌握与来源外质量均未通过预登记验收，无模型晋升。')
    text=text.replace('主训练尚在运行；本文件不能作为新模型已改善或已掌握的证据。','主训练和独立复核已完成，但本文件不证明新模型达到全面掌握或可迁移要求。')
    text+='\n## 实际结果与停止决定\n\n'
    text+='本机RTX 4060 Laptop GPU执行，未使用平台训练。12个主拟合各完成100轮，合计71,200步，拟合计时合计约%.2f分钟（不含中止运行和独立验收）。确认阶段0拟合、0更新。V134中止另有3次启动、2次完成、至少20,693步日志；本次请求合计15次启动、14次完整拟合、至少91,893步日志，不能把中止成本隐去。\n\n' % (final['primary_training_seconds']/60)
    text+='| 配置 | TRAIN非混标M错 | TRAIN非混标S错 | 通过五轮稳定验收的角色 | 来源外ASA M错 | 来源外ASA S错 | 完整任务总错 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    text+='| 原A0参照 | — | — | — | 318 | 2074 | 2499 |\n'
    for a in ARMS:
        t=train[a];m=primary['metrics'][a];errs=sum(z['missed'] for z in m['full_task'].values())
        text+=f"| {a} | {t['pure_M_errors']} | {t['pure_S_errors']} | {t['stable_roles']}/3 | {m['ASA']['1']['missed']} | {m['ASA']['2']['missed']} | {errs} |\n"
    text+='\nTRAIN数字为三折合法训练角色原行计数，同一原行可进入两个角色，不是独立原行数量；混标最低22/6/28未删除。ASA留出人口为78,748条M、34,059条S。\n\n'
    text+='唯一候选R_decay：相对原A0修复8条M、新增1438条M，净增1430条M错；修复435条S、新增16条S，净减少419条S错。相对新R_const减少366条M错，却增加16条S错；因此配对S保护仍失败。相对上一轮V131 R修复308M/10S、新增100M/24S，M净减208、S净增14；这些进步不能替代与原A0比较。\n\n'
    full=primary['metrics']['R_decay']['full_task']
    text+='| 完整任务真实类别 | 原行支持 | 判对 | 漏判 | Recall | Precision | F1 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for c,name in [('0','正常'),('1','恶意'),('2','可疑')]:
        z=full[c];text+=f"| {name} | {z['support']} | {z['correct']} | {z['missed']} | {z['recall']:.6%} | {z['precision']:.6%} | {z['f1']:.6%} |\n"
    text+='\n不能以大多数正常记录的高准确率宣告通过。可疑来源平均召回11.5125%→16.2834%，零召回来源198→189，是实际局部收益；但233个S来源仍有189个零召回。时间头682行新增12错；来源2868的1184条M仍错1040，原A0仅错48。四臂均未通过恶意、总错、多折改善、完整逐类和受保护切片等门槛。非ASA冻结组件仍107错。\n\n'
    text+='### 第一性原理审查：已证实与尚未证实分开\n\n'
    text+='- **对照缺陷已修复，分类缺陷没有因此自动解决。** 66项绑定源检查保持一致，6组前80轮配对共36,098,240次原行比较逐位一致。旧/新执行源码分别保留；不得用计算修复冒充业务精度提升。\n'
    text+='- **固定80轮衰减不代表学习已经完成。** 第0折R到80轮仍有139次非混标错误，O仍62次；末期衰减能减少部分反复，也可能保留错误。O在第2折有收益，第0折反而增加S错；不能宣布衰减普遍无效，也不能继续按统一时点认定学完。\n'
    text+='- **辅助监督改变优化取舍，没有新增行为信息。** 原行/辅助两损失的同日程对照表明，TRAIN错误减少而来源外M增加；下一轮不能依据训练S归零继续盲增S权重。\n'
    text+='- **不能把残错统称为所有成员都缺能力。** R_decay的80次纯输入残错、O_decay的48次均存在正确成员；此标志已由实际模型重放验证。成员存在正确输出不等于组合判决正确，也不提供无需真值的选成员方法。旧直接投票失败仍保留，不事后改变本轮聚合。\n'
    text+='- **梯度相反只是诊断，不能直接认定唯一根因。** 第1折R_decay终点原行M/S在facts_direct上的梯度余弦约-0.999868；二分类竞争及目标平衡也会形成相反方向。需和真实错误、置信度及配对干预结合，不能据此直接加入PCGrad或新权重。\n'
    text+='- **保护切片失败仍未解决。** 12个头部S错误与2868大量M错误必须逐行保留。当前结果没有证明它们都是日期依赖，也没有证明缺少的上下文靠更强模型可补齐；须分别做合法输入对照与来源支持核查。\n\n'
    text+='### 验收证据与后续边界\n\n'
    text+='独立审核90,245,600次TRAIN原行×轮记录；60个末五轮实际模型重放4,512,280次原行概率、预测及成员失败标志；全部终点对照绑定的原始官方真值与2,056,871条原行计数。12个训练角色均未通过五轮稳定要求。无留出标签参与梯度，未改变种子、官方标签、输入、候选或门槛。执行测试与实际重放通过仅说明过程可信；学习和分类验收均失败。\n\n'
    text+='停止本轮扩训和晋升。后续先拆分“成员已能表达但组合仍错”与“持续学习/置信度取舍”，在合法TRAIN来源内做匹配验证，再核查来源2868缺支持的M及其正确对照；任何新日程、聚合或目标需另行登记，不能把本轮其他臂事后拼接为赢家。本轮不启动新的未登记训练。\n\n'
    text+='关键文件：`artifacts/v135_stable_learning_trial_20260930/final_delivery.json`、`quality.json`、`learning_qualification.json`、`postflight_audit/audit.json`、`postflight_audit/factorial_effects.csv`、`error_review/TRAIN_error_transitions.json`、`error_review/HELD_repair_and_regression.csv`及全部逐轮模型/原行/来源账本。新增反例保存于 `training/review_policy/v135_observed_cases.json`；真实V134前80轮不一致反例由 `test_v135_evidence_regression.py` 回放。\n'
    doc.write_text(text,encoding='utf-8')
    summary='V135完成12次拟合、71200次参数更新；训练与来源外质量失败，无模型晋升。R_const/R_decay/O_const/O_decay非混标TRAIN残错77/80/52/48；ASA候选R_decay M/S错1748/1655，对原A0净增1430M错、净减419S错。可疑来源均值召回11.51%到16.28%，零召回198到189，但头部12错及2868的1040M错未解决。V134数值非确定性中止证据保留，前80轮配对与60模型重放已通过，确认0拟合。'
    rp=ROOT/'README.md';s=rp.read_text(encoding='utf-8');start=s.index('当前实际执行（V135');end=s.index('\n\n',start)
    s=s[:start]+'当前实际执行结果（V135）：**'+summary+'** [完整结果、问题与停止决定](docs/V135_TRAINING_EXECUTION_AND_RESULTS.md)。'+s[end:];rp.write_text(s,encoding='utf-8')
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    catalog['project'].update(authoritative_delivery_id='v135-delivery',authoritative_direction_id='v135-review',current_summary=summary,
        current_direction=['本轮12个主拟合已完整结束，训练掌握与质量验收失败，不追加确认，不部署。','保存对照修复证据；下一设计先区分成员表达、组合置信度及未完成学习。','在合法TRAIN来源内设计匹配验证，核查2868缺支持M与原正确对照；保留头部及全部逐类保护。','新目标、日程、聚合须有新预登记；不事后晋升O或分折拼接。'],
        known_limits=['开发来源已反复查看，无独立盲测、比赛提交或真实网络迁移验收。','TRAIN重复角色计数不等于独立原行数；0错要求仅为本轮预登记诊断。','ASA重新训练，其他格式仍为已验证的冻结组件。','正确成员存在不等于存在无需标签的正确选择器；梯度相反也不证明唯一因果。','V134中止更新数为日志下界，全部成本与记录保留。'])
    for entry in catalog['documents']:
        if entry['id']=='v135-review':entry.update(summary=summary,sha256=sha(doc))
    for ident,path,title,category in [('v135-delivery',destination,'V135实际训练与独立验收交付','current_delivery'),
        ('v135-observed-cases',ROOT/'training/review_policy/v135_observed_cases.json','V135真实执行及质量反例','review_evidence'),
        ('v135-postflight',OUT/'postflight_audit/audit.json','V135实际模型及对照复核','review_evidence')]:
        catalog['documents'].insert(0,{'id':ident,'title':title,'path':path.relative_to(ROOT).as_posix(),
            'category':category,'summary':summary,'sha256':sha(path),'keywords':['V135','训练','M/S','实际结果','验收','反例']})
    save(catalog_path,catalog)
    save(OUT/'closeout_receipt.json',{'status':'verified_records_published','source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),destination,doc,rp,catalog_path,ROOT/'training/review_policy/v135_observed_cases.json']},'new_fits':0,'new_updates':0,'model_promoted':False})
    print(json.dumps({'status':final['status'],'fits':12,'updates':71200,'TRAIN':train,'confirmation':0},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
