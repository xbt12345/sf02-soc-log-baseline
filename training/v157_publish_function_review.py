"""Publish actual function audit, failures, and absence of a qualified new trial."""
import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v157_functional_logit_audit_v2 import OUT,require,save

DOC=ROOT/'docs/V157_COMPLETE_FUNCTION_RESULTS_AND_TRAINING_DECISION.md'
CASE=ROOT/'training/review_policy/v157_actual_function_cases.json'

def main():
    require();assert not DOC.exists() and not CASE.exists()
    audit=read(OUT/'audit.json');verification=read(OUT/'v2_verification.json')
    check_bindings(audit['source_sha256']);check_bindings(verification['source_sha256'])
    failure=ROOT/'artifacts/v157_complete_function_logit_audit_20261001/failure.json'
    assert read(failure)['progress']['classifier_completed']==12
    assert audit['classifier_forward_calls']==36 and audit['cumulative_classifier_forward_calls']==48
    rows=pd.read_parquet(OUT/'all_original_role_function_summary.parquet')
    s=pd.read_parquet(OUT/'v2_fixed_cohort_function_summary.parquet')
    ag=s.groupby(['cohort','role','truth']).sum(numeric_only=True)
    hard=ag.loc[('hard578','outer_HELD',2)];strict=ag.loc[('strict51','outer_HELD',2)]
    assert [int(hard[k]) for k in ['original_rows','errors','unanimous_M_members','body_positive_S_margin','facts_positive_S_margin']]==[578,576,188,6,486]
    assert int(strict.original_rows)==51 and int(strict.errors)==0
    outer=rows[rows.query_role.eq('outer_HELD')]
    assert len(outer)==112807
    hypothetical={str(c):int(g.mean_logit_pred.ne(c).sum()) for c,g in outer.groupby('truth')}
    actual={str(c):int(g.pred.ne(c).sum()) for c,g in outer.groupby('truth')}
    assert actual=={'1':1904,'2':1629} and hypothetical=={'1':2008,'2':1639}
    # Fix the already registered sixteen new-M rows as a cohort in all three roles.
    gp=ROOT/'artifacts/v156_conditional_representation_neighborhood_20261001/all_original_role_neighbors.parquet'
    g=pd.read_parquet(gp,columns=['training_role','space','query_role','row_position','truth','pred_baseline','pred_V155_B'])
    marked=g[g.space.eq('all16_V146_A_H2')&g.query_role.eq('outer_HELD')&g.truth.eq(1)&g.pred_baseline.eq(1)&g.pred_V155_B.ne(1)]
    ids=set(marked.row_position);assert len(ids)==16
    selected=rows[rows.row_position.isin(ids)]
    assert len(selected)==48 and selected.pred.eq(1).all()
    selected.to_parquet(OUT/'new_M16_fixed_all_roles.parquet',index=False)
    transfers=[]
    for (f,cohort),g in s[s.cohort.isin(['hard578','strict51'])&s.truth.eq(2)].groupby(['fold','cohort']):
        for role in ['legal_TRAIN','outer_HELD']:
            z=g[g.role.eq(role)].iloc[0]
            transfers.append(f"|{f}|{cohort}|{role}|{int(z.original_rows)}|{int(z.errors)}|{z.body_S_minus_M_mean_mean:.6f}|{z.facts_S_minus_M_mean:.6f}|{z.bias_S_minus_M_mean_mean:.6f}|{z.p2_mean:.6f}|")
    population=[]
    for name in ['all_ASA','unknown_or_missing_port','ICMP','mixed_actual_local']:
        for cl in [1,2]:
            tr=ag.loc[(name,'legal_TRAIN',cl)];he=ag.loc[(name,'outer_HELD',cl)]
            population.append(f"|{name}|{cl}|{int(tr.original_rows)}|{int(tr.errors)}|{int(he.original_rows)}|{int(he.errors)}|")
    bindings=[OUT/'audit.json',OUT/'v2_verification.json',OUT/'v2_fixed_cohort_function_summary.parquet',OUT/'v2_all_field_fixed_cohort_summary.parquet',OUT/'new_M16_fixed_all_roles.parquet',failure,gp]
    save(CASE,dict(status='actual_complete_function_cases_not_new_training_or_model_authority',latest_actual_training='V155',
        successful_classifier_forward_calls=36,failed_classifier_forward_calls=12,cumulative_classifier_forward_calls=48,
        new_classifier_fits=0,new_gradients=0,new_updates=0,actual_outer_M_S_errors=actual,
        mean_logit_diagnostic_M_S_errors=hypothetical,hard578_outer_errors=576,hard578_unanimous_M_rows=188,
        hard578_facts_positive_S_margin_rows=486,strict51_real_errors=0,
        cases=['Actual mean-member probabilities are not mean logits; replacing rule would worsen both classes.',
            'Legal TRAIN mastery coexists with outer body margin reversal; continued TRAIN confidence is not transfer.',
            'Large body or direct-field components are additive algebra, not semantic cause or permission to remove fields.',
            'No new semantic or cross-source class teacher was created; eligibility, efficacy and promotion remain distinct.'],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in bindings}))
    DOC.write_text('''# V157 完整分类函数：实测、反例与训练决定

2026-10-01。**完整函数核查已完成，当前证据仍不足以登记新的正式训练。最新实际训练仍为 V155，质量未通过、未晋升；三个问题的完整目标继续开放。** 本轮复现固定 V146 A 三折共同起点，没有生成新分类器、标签或修复收益。

## 完整执行与两项实现纠正

原始设计、方案和源码在求值前封存3614个物理文件。首次执行完成首折12次完整forward（内部12次features），随后因局部变量all遮蔽内置函数，在状态检查处退出；源码、方案、seal、failure及实际计数全部保留。新v2只修汇总变量名和版本/输出/预算，AST检查无相关内置遮蔽；重新封存3619个物理文件后，一次完成36次forward/36次内部features。**累计48次forward/48次features，0拟合、梯度、更新；一次技术重试不是新科学方法。** 按入口每次分解的31次矩阵乘法另计，累计1488次，不冒充免费计算。两个register仅初始化、加载、核实依赖和非分类器softmax设置，没有分类器求值。

每折全部22546实际local、16成员、三类别输出；映射全部112807 ASA原行，合计338421角色行（225614合法TRAIN＋112807 outer）。保留混标、未知、ICMP、数值别名、原类频次、所有495 facts坐标，分成29组而非只挑有利字段。没有重算其他格式模型，完整2056871行质量继续以V155真实交付为准。

实际forward内捕获H2，不额外调用features；按原运算顺序body＋bias＋facts逐位重构全部logits，成员softmax后平均的概率与原保存起点逐位相同，模型张量身份不变、grad为空。字段分组求和最大差3.552713678800501e-15；概率差最大3.33e-16。

CPU独立重算保存logits的三分类softmax及原行映射，最大概率差3.33e-16，argmax一致。第一次CPU汇总使用了不存在的protocol键，ICMP切片误为空；全部原行概率复核不受影响。原结果保留，v2改为实际transport_protocol并断言2373条ICMP，单列纠正凭据。以v2_verification.json和v2_前缀汇总为有效派生结果。后续new_M16_fixed_all_roles.parquet将原登记16条新增M错固定到三个角色，48角色行均由V146 A正确分类；初版V2 cohort表的该切片是各角色即时负翻转，不能冒充固定TRAIN cohort。

## 全人口和原登记控制

下表是复现的V146 A判决，不是V157新候选。类1=M，类2=S；角色重复不增加独立样本。未知/缺端口切片与ICMP可以重叠。

|人口|类|TRAIN角色行|TRAIN错|outer原行|outer错|
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(population)+'''

578困难S在两个合法TRAIN角色1156行全部判对，所有成员均为S；outer仍576错，其中188原行16成员一致为M。outer仅6条body的平均S−M margin为正，facts直连却有486条为正。与之同时，全outer S的body正margin为32404/34059，不能将困难组描述扩大为全体分类失效。

|折|控制|角色|原行数|实际错|body平均S−M|facts平均S−M|bias平均S−M|真实pS均值|
|---|---|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(transfers)+'''

这些值说明冻结函数怎样合成输出。它们没有识别安全因果：表示与facts包含相关条件，改变某一通路会改变整体函数，不能由大分量直接授权删除字段、抑制body或拟合outer答案门控。TCP上http_status等分量还包含缺失哨兵，不能读成实际HTTP事件。29组完整数值汇总保留，包括零值，不挑最大字段作为“罪魁祸首”。

## 集成公式反例

真实mean-probability判决与mean-logit诊断在346条outer原行不同（M312、S34），TRAIN角色没有该差异。mean-logit会把M错误1904变成2008、S错误1629变成1639，总错误3533变3647；没有部署或作为新候选。困难578中8条会改变判决，严格正确51中14条也会改变成错；不能只报告局部修复。V155新增16M在基线均正确，其中8条mean-logit诊断会判错。这排除了把替换集成公式当本轮有支持修复机制。

## 有限诊断结束后的唯一决定

不追加V155同配置拟合；不重启V137–V140旧冻结读出12拟合、V146辅助或V155邻域预算。V90/V111已有不同端点/接口的函数分解，此次填补当前真实H2＋head＋495facts＋16概率成员缺口，不包装成首次发现或新监督。

**目前没有已具备资格的新正式训练候选。** 新证据把问题收窄为：合法TRAIN已经掌握的观察条件组合，为何在隔离来源中的完整body路径改变类别判断；现有代数分量、可解码字段和余弦关系没有提供合法的跨来源类别监督。旧读出重训、直接删端口、按困难名单增权、把近邻标签复制或扫阈值都缺少本轮新增依据。

下一项工作只形成并审查一项有实质区别的机制：必须明确改变哪一段分类函数或监督，指出合法TRAIN中可用的观察条件交互与类别支持，说明与历史失败方法的差异，并给一个原频次匹配对照、固定终点和有限预算。不能无限重复字段读出/距离/同一函数分解。若当前资料仍不能给出上述输入，明确保留不足，不把技术成功自动转成训练资格。试验资格不要求先证明最终有效；但需要可执行的新因素和可推翻假说。本报告不启动新fit，亦不把尚未解决的完整目标关闭。

新正式试验仍须保护V138/V140/V142已验收TRAIN能力；outer标签不得进入梯度、路由或选点。实际胜负按完整原行M/S保护、至少两折改善、小来源/已登记root与完整N/M/S的P/R/F1判定。A0门槛是项目约束，不是官方评分公式。完整来源外分类收益尚未新增。

父[V156独立训练决定](V156_INDEPENDENT_TRAINING_DECISION_AND_TARGETED_PLAN.md)保持原文件，只读发布；不把其中准备状态改成已执行。原始/修复方案、两次seal、失败12成本、36次完整输出、两次CPU复核及cohort纠正均保留在对应artifacts目录。
''',encoding='utf-8')
    paths=[ROOT/z for z in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    snap=OUT/'previous_publication_snapshot';snap.mkdir()
    for p in paths:
        t=snap/p.relative_to(ROOT);t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t)
    save(snap/'manifest.json',dict(source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
    cp=paths[2];cat=read(cp);pr=cat['project'];assert pr['authoritative_delivery_id']=='v155-delivery'
    pr['authoritative_direction_id']='v157-function-review'
    pr['current_summary']+=' V157完整函数实测完成：修复重放36次forward含36features，原失败12计入累计48；0新拟合/梯度/更新。全部495facts/16成员原概率精确重放，CPU复核及ICMP纠正保存；当前不足以登记新正式训练，完整目标active。'
    pr['current_direction']=['最新实际训练V155六拟合，质量失败、未晋升；已验收TRAIN能力保持。','V157一次完整函数诊断结束，累计48次前向/48内部features、0新拟合。','body/facts分量与平均logit均不能给新标签或安全因果；完整控制保留。','下一形成一项有实质新因素及合法TRAIN类别交互支持的有限机制；当前证据不足，不自动增加相同诊断或旧训练。']
    pr['known_limits'].append('V157只复现固定V146 A完整函数；不证明跨来源因果、不增加标签或模型收益。CPU初次ICMP空切片已另版纠正，原结果保存。')
    parent_doc=ROOT/'docs/V156_INDEPENDENT_TRAINING_DECISION_AND_TARGETED_PLAN.md'
    entries=[('v157-function-review','V157完整函数实测与当前训练不足',DOC,'current_review'),
        ('v157-function-design','V157完整函数零拟合设计',ROOT/'docs/V157_COMPLETE_FUNCTION_DIAGNOSTIC_DESIGN.md','review_evidence'),
        ('v157-function-retry','V157一次技术修复与累计48预算',ROOT/'docs/V157_COMPLETE_FUNCTION_DIAGNOSTIC_TECHNICAL_RETRY.md','review_evidence'),
        ('v157-function-audit','V157全部成员函数真实重放',OUT/'audit.json','review_evidence'),
        ('v157-function-verification','V157全部保存数组CPU与角色复核',OUT/'v2_verification.json','review_evidence'),
        ('v157-function-first-failure','V157原入口失败12次成本',failure,'review_evidence'),
        ('v157-cpu-cohort-correction','V157 CPU ICMP键纠正与原结果保存',OUT/'cpu_replay_cohort_correction.json','review_evidence'),
        ('v156-independent-training-decision','V156父独立研究与有限函数核查决定',parent_doc,'review_evidence')]
    for i,t,p,c in entries:
        assert not any(z['id']==i for z in cat['documents'])
        cat['documents'].insert(0,dict(id=i,title=t,path=p.relative_to(ROOT).as_posix(),category=c,summary=t,sha256=sha(p),keywords=['V157','当前','完整','函数','训练决定']))
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024;cp.write_bytes(raw)
    rp=paths[0];title,rest=rp.read_text(encoding='utf-8').split('\n',1);rest=rest.replace('当前方向（V156实测）','历史诊断（V156实测）')
    rp.write_text(title+'\n\n当前方向（V157实测）：[完整函数结果与训练决定](docs/'+DOC.name+')。修复版36次完整forward/内部features，失败12次计入累计48；0新拟合/梯度/更新。当前证据不足以登记新正式训练。最新实际训练仍V155，质量失败、未晋升；完整目标active。\n\n'+rest.lstrip('\n'),encoding='utf-8')
    hp=paths[1];hp.write_text(hp.read_text(encoding='utf-8')+'\n\n## V157完整函数实测与训练决定\n\n'+DOC.relative_to(ROOT).as_posix()+'；OUT '+OUT.relative_to(ROOT).as_posix()+'。原失败12forward原样保留，新v2封存3619物理文件、36forward成功，累计48/48features，0fits/grads/updates。16成员全部logits body+bias+495facts精确重构、原概率一致；CPU已复核、ICMP键另版纠正2373原行。TRAIN掌握不代表outer稳定，困难578仍错576，188条16成员一致判M。mean-logit会恶化M与S，分量不是安全因果，不删字段/复制邻近标签/用outer拟合。有限诊断结束；形成一项有实质差异、合法TRAIN类别交互支持的机制后另审有限方案。当前不足，无新fit，latest actual V155未晋升/full active。父V156独立决定报告已只读发布。\n',encoding='utf-8')
    tp=paths[3];tp.write_text(tp.read_text(encoding='utf-8').replace('v156-neighborhood-review','v157-function-review').replace('self.assertIn("V156", json.dumps(fetch_result.structured_content, ensure_ascii=False))','self.assertIn("V157", json.dumps(fetch_result.structured_content, ensure_ascii=False))'),encoding='utf-8')
    save(OUT/'publication.json',dict(status='actual_function_review_published_latest_training_V155_unchanged',catalog_bytes=len(raw),latest_actual_training='V155',quality_acceptance=False,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),DOC,CASE,parent_doc]+paths}))
    print(json.dumps(dict(published=True,current_direction='V157',latest_actual_training='V155',catalog_bytes=len(raw))))

if __name__=='__main__':main()
