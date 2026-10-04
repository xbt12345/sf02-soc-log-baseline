"""Publish only actual V155 full evaluation, preserving zero-step history."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
from v138_runtime import save

OUT=ROOT/'artifacts/v155_guarded_full_gradient_sam_20261001'
DOC=ROOT/'docs/V155_FULL_GRADIENT_NEIGHBORHOOD_RESULTS_AND_DECISION.md'
CASE=ROOT/'training/review_policy/v155_observed_neighborhood_cases.json'

def prepare():
    assert not DOC.exists() and not CASE.exists()
    d=read(OUT/'final_delivery.json');q=read(OUT/'quality.json');v=read(OUT/'verification.json')
    assert d['classifier_fits']==6 and v['official_rows']==2056871
    assert d['quality_sha256']==sha(OUT/'quality.json') and d['verification_sha256']==sha(OUT/'verification.json')
    paths=[OUT/n for n in ['final_delivery.json','quality.json','verification.json','learning_qualification.json','ASA_prediction_ledger.parquet','full_prediction_ledger.parquet','run_seal.json','fit_activation.json']]
    paths += [OUT/f'fold{f}_{a}'/n for f in range(3) for a in ['A','B'] for n in ['fit.json','proposals.jsonl','progress.json']]
    failed=[k for k,z in q['B']['gates'].items() if not z]
    save(CASE,dict(status='actual_six_fit_neighborhood_trial_classification_cases',failed_task_gates=failed,
        cases=[dict(id='V155-FIXED-PROXY-NOT-MOVING-RISK',action='Replay accepted B proposals: frozen-current-epsilon Armijo does not imply cross-iteration monotonic risk.'),
            dict(id='V155-TRAIN-GUARD-NOT-TRANSFER-QUALITY',action='Keep actual preserved TRAIN scope separate from all matched and full classification gates; no loss-based promotion.')],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    print(json.dumps(dict(cases_prepared=True,failed_task_gates=failed),ensure_ascii=False))

def publish():
    assert CASE.exists() and not DOC.exists() and not (OUT/'publication.json').exists()
    check_bindings(read(CASE)['source_sha256']);replay=read(OUT/'observed_cases_replay.json');check_bindings(replay['source_sha256'])
    d=read(OUT/'final_delivery.json');q=read(OUT/'quality.json');v=read(OUT/'verification.json')
    fits=[read(OUT/f'fold{f}_{a}/fit.json') for f in range(3) for a in ['A','B']]
    table='\n'.join(f"|{z['fold']}|{z['arm']}|{z['full_gradient_evaluations']}|{z['proposal_evaluations']}|{z['accepted_updates']}|{z['termination']}|{z['endpoint_stats']['M_errors']}/{z['endpoint_stats']['S_errors']}|{z['stable_last_five']}|" for z in fits)
    outer='\n'.join(f"|{a}|{d['ASA_errors'][a]['M']}|{d['ASA_errors'][a]['S']}|{sum(d['ASA_errors'][a].values())}|{sum(d['ASA_errors'][a].values())+107}|" for a in ['A','B'])
    metrics='\n'.join(f"|{a}|{c}|{q[a]['full_task']['B'][str(c)]['support']}|{q[a]['full_task']['B'][str(c)]['missed']}|{q[a]['full_task']['B'][str(c)]['false_called']}|{q[a]['full_task']['B'][str(c)]['precision']:.9f}|{q[a]['full_task']['B'][str(c)]['recall']:.9f}|{q[a]['full_task']['B'][str(c)]['f1']:.9f}|" for a in ['A','B'] for c in [0,1,2])
    failed={a:[k for k,z in q[a]['gates'].items() if not z] for a in ['A','B']}
    text=f'''# V155 完整梯度一阶邻域风险对照：实际结果与决定

2026-10-01。**六次登记拟合与完整2056871行验收已实际完成；匹配效果通过={d['matched_effect_passed']}，任务质量通过={d['quality_acceptance']}，模型晋升={d['model_promoted']}。** 训练侧已验收能力保持与迁移分类质量分别判断。第二、第三及完整任务仍未关闭；本轮没有创造新独立同类支持，也没有新环境泛化确认。

实际拟合梯度{d['full_gradient_evaluations']}、proposal {d['proposal_evaluations']}、接受更新{d['accepted_updates']}，拟合分类器分块前向{sum(z['classifier_forward_chunk_calls'] for z in fits)}；零步9完整梯度/108前向及注册1×1 dummy 1前向/梯度分别计量，不冒充拟合收益。评价实际模型分块前向{json.dumps(v['actual_evaluation_model_forward_chunk_calls'],ensure_ascii=False)}，评价新增梯度0。

## 单因素、真实终点与保留能力

[设计审查](V155_NEIGHBORHOOD_TRIAL_DESIGN_AND_IMPLEMENTATION_REVIEW.md)、[零步历史](V155_ZERO_STEP_REVIEW_AND_EXECUTION_STATUS.md)与机器方案均在更新前绑定。A普通完整原频次成员CE、B完整梯度一阶扰动点CE，同折V146 A共同初始化，既有第二层22528参数；ρ=.001×初始化范数整个fit固定，无扫描。两组共享归一化方向、步长和回溯，B使用gplus而忽略ε导数；proposal内ε固定，只验收该局部代理Armijo。B多一次梯度，不称等算力。全部合法TRAIN原行频次、未知、ICMP、冲突保留，无外折答案选点或自动追加。

|折|臂|完整梯度|proposal|接受更新|登记终止|TRAIN M/S错|末五状态稳定|
|---|---|---:|---:|---:|---|---|---|
{table}

终点及末五状态逐模型重放，固定ε代理/普通成员风险与日志重现；旧V138/V140/V142联合保护实际执行，A/B三role及五状态联合通过={all(d['TRAIN'][a]['all_roles_mastered'] for a in ['A','B'])}，纯TRAIN错{ {a:d['TRAIN'][a]['pure_errors'] for a in ['A','B']} }和初始正确原行新增错{ {a:d['TRAIN'][a]['all_original_correct_new_errors'] for a in ['A','B']} }。六模型全部22546实际数值输入绕过隐藏缓存重放，argmax一致、概率最大差{max(z['cache_free_max_gap'] for z in v['audits']):.12g}，冻结reference/head张量身份不变。窗口/模型核验只证明其覆盖的TRAIN与输入执行范围，不代替来源外验收。

## 原行分类质量

|臂|ASA M错|ASA S错|ASA总错|完整人口总错|
|---|---:|---:|---:|---:|
{outer}

完整人口2056871行独立真值逐行计分；ASA112807行由六注册模型真实重放，非ASA沿用V146冻结预测、独立重计107错，不能称重新运行其模型。已看开发折不能改称盲测或正式提交。A0原ASA M318/S2074，任务ASA总错上限2170，完整各类precision/recall/F1、来源/小组/未知及多折标准均未放宽。

|臂|类0=N/1=M/2=S|原行数|漏判|误报|precision|recall|F1|
|---|---|---:|---:|---:|---:|---:|---:|
{metrics}

匹配保护：`{json.dumps(q['matched_gates'],ensure_ascii=False)}`。

任务未通过项：A `{json.dumps(failed['A'],ensure_ascii=False)}`；B `{json.dumps(failed['B'],ensure_ascii=False)}`。

B对匹配A逐类别修复/新增错：`{json.dumps(d['original_pairs']['B_vs_A'],ensure_ascii=False)}`。

B对上轮V146 A逐类别修复/新增错：`{json.dumps(d['original_pairs']['B_vs_V146_A'],ensure_ascii=False)}`。

## 反例与边界

实际B接受日志有{replay['changed_epsilon_cross_iteration_proxy_rises']}次相邻迭代代理值上升，当前各自固定ε代理仍满足Armijo；重放脚本确认扰动身份改变。这直接阻止把局部代理验收写成移动邻域风险单调。普通loss、局部TRAIN正确、论文动机都不能覆盖实测分类门槛。反例文件 `training/review_policy/v155_observed_neighborhood_cases.json`，可执行 `training/v155_replay_observed_cases.py`；实际重放0模型前向/梯度/拟合/更新。

试验资格、分类质量及替换资格分开。本批只对这组共同初始化、固定半径和一阶适配产生证据，不宣布全部SAM无效、不挑折/种子重试、不从已看外层选择其他终点。同配置主门槛失败则不追加确认；即便局部效果通过，也没有权限关闭细行为同类支持、跨来源稳定性或完整任务。

产物 `artifacts/v155_guarded_full_gradient_sam_20261001/` 包含六fit、原行概率/预测、梯度/proposal日志、末五状态/终点、完整账本及质量拒绝原因。原V146/V154及V155封存源码、机器方案、零步报告保留。
'''
    DOC.write_text(text,encoding='utf-8')
    paths=[ROOT/n for n in ['mcp_readonly/catalog.json','README.md','HANDOFF.md','mcp_readonly/tests/test_readonly_mcp.py']]
    snapshot=OUT/'pre_results_publication_snapshot';snapshot.mkdir()
    for p in paths:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    save(snapshot/'manifest.json',dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    cp=paths[0];cat=read(cp);pr=cat['project'];pr['authoritative_delivery_id']='v155-delivery';pr['authoritative_direction_id']='v155-results-review'
    pr['current_summary']=f"最新实际V155六拟合完成：{d['full_gradient_evaluations']}梯度/{d['proposal_evaluations']}proposal/{d['accepted_updates']}更新；完整2056871行验收，匹配{d['matched_effect_passed']}、任务质量{d['quality_acceptance']}，未晋升。历史参考（下述V146及零步状态已被本次实际执行更新）："+pr['current_summary']
    pr['current_direction']=[f"最新实际V155六拟合及完整验收完成，匹配通过{d['matched_effect_passed']}、任务通过{d['quality_acceptance']}、晋升False。",
        'TRAIN能力保持与来源外分类分开；第二第三及full未关闭，无新增独立同类支持或新环境确认。',
        '固定ε局部Armijo不保证移动风险单调；保留全部逐类/来源修复退化及失败门槛，不用loss或论文覆盖。',
        '同配置主门槛失败则不追加确认、半径/seed/系数/终点扫描；后续新机制须依据真实残余错误审查。']
    entries=[('v155-results-review','V155六拟合完整原行验收与决定',DOC,'current_review'),('v155-delivery','V155最新实际六拟合交付',OUT/'final_delivery.json','delivery'),
        ('v155-full-quality','V155完整三分类与来源匹配质量',OUT/'quality.json','review_evidence'),('v155-actual-verification','V155实际终点窗口输入重放与计量',OUT/'verification.json','review_evidence'),
        ('v155-real-counterexamples','V155真实代理与分类反例重放',OUT/'observed_cases_replay.json','review_evidence'),('v155-independent-pretrain-review','V155父实际源码方案独立审查',ROOT/'artifacts/v155_independent_pretrain_review_20261001/review.json','review_evidence')]
    assert not {z['id'] for z in cat['documents']}.intersection(z[0] for z in entries)
    cat['documents']=[dict(id=i,title=t,path=str(p.relative_to(ROOT)).replace('\\','/'),category=c,summary=t,sha256=sha(p),keywords=['V155','当前','实际训练','完整质量','邻域','反例']) for i,t,p,c in entries]+cat['documents']
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024;cp.write_bytes(raw)
    rp=paths[1];s=rp.read_text(encoding='utf-8').replace('最新实际（V146）与当前方向（V155零步）','历史实际（V146）与V155零步时的方向',1).replace('V155新增实际：','V155零步历史：',1)
    lead=f"最新实际与当前方向（V155完整验收）：[六拟合完整验收与决定](docs/{DOC.name})；真实{d['full_gradient_evaluations']}梯度/{d['proposal_evaluations']}proposal/{d['accepted_updates']}更新，完整2056871行，匹配通过{d['matched_effect_passed']}、任务通过{d['quality_acceptance']}，未晋升。旧能力保持与迁移质量分别判定，第二第三及完整目标未关闭；本配置不追加确认。以下零步/旧轮段落是历史记录。\n\n"
    title,rest=s.split('\n',1);rp.write_text(title+'\n\n'+lead+rest.lstrip('\n'),encoding='utf-8')
    hp=paths[2];s=hp.read_text(encoding='utf-8')+f"\n\n## V155六拟合与完整质量已实际完成\n\n最新结果 docs/{DOC.name} / {OUT.relative_to(ROOT)}/final_delivery.json；实际{d['full_gradient_evaluations']}梯度/{d['proposal_evaluations']}proposal/{d['accepted_updates']}更新。六endpoint/末五状态/frozenε proxy/cache-free输入真实重放与旧联合保护，完整2056871原行质量已验。匹配通过{d['matched_effect_passed']}，任务通过{d['quality_acceptance']}，未晋升/未关闭第二第三/full。不可再按上段零步记录启动同六fit；source/plan/seal及历史均保留，无overwrite/resume/自动追加。反例文件与v155_replay_observed_cases.py actual重放；完整质量失败项见quality.json。当前停止该已执行配置的追加，继续工作须新证据/因素审查。\n";hp.write_text(s,encoding='utf-8')
    tp=paths[3];tp.write_text(tp.read_text(encoding='utf-8').replace('v155-preflight-review','v155-results-review'),encoding='utf-8')
    save(OUT/'publication.json',dict(status='actual_six_fit_full_quality_published',catalog_bytes=len(raw),classifier_fits=6,model_promoted=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),DOC,CASE,OUT/'observed_cases_replay.json']+paths}))
    print(json.dumps(dict(published=True,latest_actual='V155',catalog_bytes=len(raw),quality=d['quality_acceptance']),ensure_ascii=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','publish']);a=p.parse_args()
    prepare() if a.mode=='prepare' else publish()
