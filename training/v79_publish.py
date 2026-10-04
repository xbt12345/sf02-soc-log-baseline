"""Publish actual experiment state, never reinterpret failed quality gates."""
import json
from pathlib import Path
from v79_execute import DEST,FOLDS
from run_v75 import ROOT,read,save,sha


def main():
    verification=read(DEST/'independent_verification.json');assert verification['all_checks_passed']
    reg=read(DEST/'development_regression.json');shadow=read(DEST/'P5_shadow.json');transfer=read(DEST/'P4_transfer.json')
    assert not reg['passed'] and not shadow['passed']
    baseline=read(ROOT/'artifacts/v78_boundary_20260922/denial_adapter_v4/scoring.json')['totals']['O_sgd']
    fits=verification['actual_classifier_fits'];valid=verification['valid_protocol_classifier_fits']
    summary=f'v7.9本轮新增{fits}次分类器拟合（{valid}次有效对照、1次空覆盖诊断）＋0次校准拟合。M/S辅助损失与扩训稳定性未通过；冻结开发回放总错4823→5947、M召回78.47%→76.46%，未达到提升目标，无新模型晋升。'
    table=[]
    for name in ['P1_A_ovr','P1_B_ovr','P1_C_ovr','P2_B_softmax','P2_B_ms']:
        scores=read(DEST/(name+'_scores.json')); fit=read(DEST/(name+'_fit.json'));c=scores['calibration'];h=scores['evaluation']
        table.append(f"| {name} | {fit['seconds']:.2f} | {c['errors']} | {c['cm'][1][1]}/{c['support'][1]} | {c['cm'][2][2]}/{c['support'][2]} | {h['errors']} | {h['cm'][1][1]}/{h['support'][1]} | {h['cm'][2][2]}/{h['support'][2]} |")
    alltable=[]
    for k,name in enumerate(['正常','恶意','可疑']):
        b=baseline;m=reg['metrics'];alltable.append(f"| {name} | {b['cm'][k][k]}/{b['support'][k]} | {m['cm'][k][k]}/{m['support'][k]} | {b['recall'][k]*100:.4f}% → {m['recall'][k]*100:.4f}% | {b['precision'][k]*100:.4f}% → {m['precision'][k]*100:.4f}% |")
    sourcetable=[]
    for f in transfer['source_folds']:
        m=f['metrics'];sourcetable.append(f"| H{f['H']} / C{f['C']} | {m['errors']} | {m['cm'][1][1]}/{m['support'][1]} | {m['cm'][2][2]}/{m['support'][2]} | {m['macro_f1']:.6f} |")
    report=f'''# v7.9 真实训练结果与停止决定

2026-09-27。{summary} 只用官方训练数据、本机Windows CPU（数值线程上限4）；没有操作赛事平台，没有引入外部训练数据，没有官方提交。此前反复查看的官方开发答案仍是开发回归，不是新盲测。

## 1. 本轮究竟完成了什么

按照[V79执行前方案](V79_MS_BOUNDARY_AND_UNKNOWN_FORMAT_PLAN.md)冻结源文件、输入散列、种子7901/7902、三组来源划分、正常采样目标、M/S损失系数和选模规则，再执行：P1三臂＋P2两项新损失5次；额外两组来源留出2次；3种单类网络格式撤除监督3次；CEF真实零M/整格式留出2次；第二种子1次；模拟扩训1次，共14次有效分类器拟合。另一次误用`cef`而非真实`cef_fields`的空覆盖拟合保留为失败记录，**实际运行总数15，不将它作为迁移证据**。没有阈值、偏置或概率校准拟合。

全部15次达到预设收敛检查。所有模型重载、三角色混淆矩阵与sklearn逐类指标独立复算、训练入样记录散列和完整201万行开发回归复算通过。实现核验通过不等于模型质量通过。

完全相同编码输入按字节比较确认后合并为457,566个特征单元，保留2,056,871条原记录的全部频率和不同原标签：正常1,899,723、恶意111,728、可疑45,420。合并没有重复降权；500条抽查特征差为0，三项目标与逐行展开损失误差最大4.44e-16。记录原件没有删除。

主拟合角色包含1,269,574条原记录，其中M67,448、S19,943全部使用；来源/正文/组件没有跨角色。32,596条native_flow-M分散在内部角色中，不将留出提前混入拟合。由于停止条件失败，本轮没有全量最终重训，因此不能声称这32,596条都进入同一个最终新模型。

## 2. 正常采样与M/S目标：真实成绩

固定66287维R0表示、alpha=1e-6、L-BFGS收敛条件。A采用全部正常原频率；B最多2048条正常/格式，M/S全保留，抽样行平均损失；C用与B完全相同的抽样，但以正入样概率和固定总体分母恢复A的风险估计。C是有限抽样估计，不能称与A最优解完全相同。

| 模型 | 拟合秒数 | 校准错误 | 校准M正确 | 校准S正确 | H错误 | H-M正确 | H-S正确 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

自然频率A与概率校正C在拟合和校准集混淆矩阵相同，H仅相差4条；C求解时间约为A的1/14.46。这是**当前表示/数据/目标下的效率收益**，不是分类收益，也不是所有后续模型都能提速14倍的保证。B的目标更符合这批逐类开发结果，校准错误少468条；不能因此称改变抽样后原目标不变。

联合softmax使校准总错少8条，但M多错16条；加M/S条件损失使校准少46条总错，却M多错38条、S精确率下降。H辅助损失比B多错18条，M少判对28条，S多判对10条。它改变了系数和联合目标，仍未摆脱类别取舍。

按校准侧冻结的逐类守门规则，A/C/softmax/辅助项均不合格，保留B-OVR作为诊断对照。开发答案没有参与这一选择。组件重采样400次后，辅助项H的M召回变化95%区间约[-0.236,-0.033]个百分点；总错误变化区间跨0。组件只是隔离代理，并非已验证的独立组织，不能用这个区间声称外部统计保证。

## 3. 迁移：能迁移哪些证据，哪些验证不成立

| 固定来源划分 | 总错误 | M正确 | S正确 | Macro-F1 |
|---|---:|---:|---:|---:|
{chr(10).join(sourcetable)}

第二种子主H仍错271条，M/S正确数不变。主校准侧最大的S失败仍存在（S仅8386/11851），不能用三个较好H的均值覆盖它。

native_flow、native_firewall、asa_protocol在官方train中都只有M。因此“保留该格式B/S但撤除M”和“移除整个格式”使用相同训练掩码，一次拟合共享两项成绩，不将其算成两种独立证据。B/S不存在处记NA；单类精确率100%不代表有正常反例支持。CEF额外提供不同掩码对照，但评价M仅12条、S不存在。

| 留出格式 | 保留已知规范事实时H-M正确 | 禁用专用事实时H-M正确 | 正常/可疑反例的限制 |
|---|---:|---:|---|
| native_flow | 6846/6970 | 320/6970 | 此格式train无B/S |
| native_firewall | 80/80 | 80/80 | 此格式train无B/S |
| asa_protocol | 4/4 | 4/4 | 仅1个H组件、4条M，不作泛化保证 |
| CEF整格式留出 | 12/12 | 12/12 | 已知规范时B0/96；禁用专用事实后B90/96，仍有6条B→M |

**关键反证：** 当前残余文本没有可靠替代native_flow的事实提取；CEF的专用事实在完全无格式监督时反而令96条正常全进S。因此适配器解析成功、能判中少量M，均不能单独证明未知格式安全。未知语法这里是关闭专用事实、保留通用残余及记录端口的压力测试，不是任意新语法能力证明。

## 4. 扩训和完整开发回放

模拟扩训加入C监督，H保持不变：错误271→283，修复20、破坏32；M正确23298→23272，S正确13357→13371。总错、M召回、S精确率及F1守门失败。没有以校准集加入训练后成绩更好作验收。

冻结选择完成后，先从无标签输入推理全部2,014,052条，再读取答案一次评分；推理中不读答案。共享重复记录计算与原逐行路径在8个位置共256条对照，特征差0、logit差<=1e-12；完整回放152秒左右。这不修改模型或输入语义。

对照为v7.8 O_sgd＋v4解析补丁，候选为本轮校准侧保留的B-OVR＋同补丁；二者都是诊断模型，不能冒称全量正式交付模型。

| 类别 | v7.8正确/真实数 | 本轮正确/真实数 | 召回率 | 精确率 |
|---|---:|---:|---:|---:|
{chr(10).join(alltable)}

总错误4823→5947（增加1124），Macro-F1 0.922732→0.904682，正常误报51→54。修复780条、新增1904条错误。预设总错误≤4340、M正确≥11729、M精确率≥86.67%、S召回≥95.68%、S精确率≥93.28%、正常误报≤34六项均失败，**不改门槛，不晋升，不进入P5全量最终拟合**。

逐个关键切片没有藏进总表：

- ASA：M5098/5098，但S→M增至2536，M精确率只有66.78%；M全对仍不是边界修复。
- VPC：M0/2664，全部进S；S1853/1853不能抵消S精确率41.02%。仍未解决。
- unsupported：M5642/6226，较参考6008净少366；修复150、新增516错，仍缺新语法的真实B/S反例。
- 载荷：S0/91→52/91，真实局部收益，但剩39条错误。
- ASA ACL：34条正常全部恢复正确；CEF正常净修复12条，仍有2条误报。
- Windows：M4/36，仍错32；S0/9；正常新增49条错误，是不能用局部收益掩盖的退化。
- 认证：M0/28、S0/1，仍未修复。

全部格式逐类正确数、混淆去向及修复/新增错误详见`artifacts/v79_execution_20260927/development_classwise.csv`。

## 5. 第一性原理：本轮能下什么结论

1. **求解不充分不是本轮失败的解释。** 五个主目标均收敛，频率合并和梯度检查通过；更低损失仍不保证开发分类更好。与SGD历史参考的差异含优化轨迹和正常抽样种子，不单因归为某一种正则化。
2. **改变类别目标没有补上独立行为证据。** 条件M/S监督降低一部分S错误，却稳定损伤M；不再原样扫描全局权重或偏置。
3. **统一事实有实际迁移价值，但某些事实/缺失状态也会形成新捷径。** native_flow事实保留/移除的强对照和CEF正常失败都要求跨格式正反例验收；不能简单关闭所有解析或把未知归S。
4. **冲突不能代表全部学习缺口。** 拟合侧完整编码M/S冲突只有2组、经验最低24错；这两组的有序残余也完全相同。有序分支不能凭空给这两组补标签区别。B拟合仍有491错，因此其余错误不能一概称不可区分；也未据此证明卷积容量一定有效。
5. **本轮没有实现有序模型/等价格式训练。** P3的条件缺少已核验的顺序丢失反例，按原计划未触发卷积拟合；10个无损封装往返检查只证明序列化保存原文，不算分类泛化收益。P0证据状态全面重构、任意未知格式支持仍未完成。不能将14个有效拟合说成全部问题已解决。

下一项应优先验证“可共享的同一行为证据在不同格式中的实际对齐”，以及ASA真实M/S反例的可观察区别；保留原件、事实与残余，逐字段给出证据位置。先用包含正常反例的CEF和ASA对照约束新适配，再用至少三种载体证实共享证据迁移；格式类支持缺失明确列出，不造标签补VPC。只有发现真实主客体/否定/字段绑定缺口，才启动预定有序分支及独立格式一致性对照。当前不能承诺它必然带来显著提升。

## 6. 多轮对抗性审查与结论

第一轮检查数值与信息：频率合并保留所有原记录质量、输入缓存对照0差异，OVR/softmax/辅助项逐行展开损失一致；不拿去重当降权。

第二轮检查选模与类别：校准总错小降不掩盖M召回或S精确率下降，所有类别计数和混淆保留；主候选冻结后才读取完整开发答案。

第三轮检查迁移证据：三个单类格式的退化协议共享同一拟合；CEF空路径排除，真实CEF补充执行；没有反例的格式不报安全保证。

第四轮检查扩训与实际效果：修复20却破坏32拦住最终扩训；完整回放六门槛全失败，不靠载荷/正常ACL局部修复发布模型。

第五轮检查容量与标签：24条完整编码冲突和467条额外拟合错误分开；“同有序残余”不等于原始行为真值相同，也不等于所有错误不可解。未训练的有序模型、格式增强、独立外部测试明确保持未完成。

第六轮检查执行记录：只修兼容性错误，不调目标或守门规则；保留原失败源与散列。一次误命名的空覆盖拟合计入实际15次，但不计入14次有效实验。失败结论同步到项目只读MCP，原正式模型和历史证据保留。

执行证据：`artifacts/v79_execution_20260927/registration.json`、`input_objective_audit.json`、`P1_selection.json`、`P2_selection.json`、`P4_transfer.json`、`P5_shadow.json`、`development_regression.json`、`independent_verification.json`及`evidence/2026-09-27/v79_execution/delivery.json`。
'''
    reportpath=ROOT/'docs/V79_EXECUTION_RESULTS.md';reportpath.write_text(report,encoding='utf-8')
    for rel,link in [('README.md','docs/V79_EXECUTION_RESULTS.md'),('docs/TRAINING_PLAN.md','V79_EXECUTION_RESULTS.md')]:
        path=ROOT/rel;text=path.read_text(encoding='utf-8');first,rest=text.split('\n',1)
        rest='\n'.join(line for line in rest.split('\n') if not line.startswith('当前执行（2026-09-27）：'))
        path.write_text(first+'\n\n当前执行（2026-09-27）：**'+summary+'** [本轮实际结果、反证与停止决定]('+link+')。以下均为历史阶段。\n'+rest,encoding='utf-8')
    ev=ROOT/'evidence/2026-09-27/v79_execution';ev.mkdir(parents=True,exist_ok=True)
    bound={}
    for path in DEST.rglob('*'):
        if path.is_file():bound[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ['v79_execute.py','v79_transfer.py','v79_input_audit.py','v79_replay_fast.py','v79_verify.py','v79_publish.py']:
        path=ROOT/'training'/name;bound[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ['docs/V79_EXECUTION_RESULTS.md','docs/V79_MS_BOUNDARY_AND_UNKNOWN_FORMAT_PLAN.md']:
        bound[name]=sha(ROOT/name)
    delivery={'status':'objective_and_transfer_round_completed_quality_not_promoted','quality_acceptance':False,'all_issues_solved':False,
        'actual_new_fits':fits,'actual_classifier_fits':fits,'actual_valid_protocol_fits':valid,'invalid_coverage_fits':1,'actual_calibration_fits':0,
        'new_full_data_final_fit':False,'full_input_replay_rows':2014052,'selected_by_calibration':'P1_B_ovr',
        'validation_scope':'Inspected official development: source/component isolation, carrier supervision withdrawal, generic-input pressure, shadow refit and complete frozen regression. No independent real unknown environment or arbitrary grammar validation.',
        'baseline':baseline,'candidate':reg['metrics'],'regression_gates':reg['gates'],'shadow_passed':shadow['passed'],
        'P3_ordered_branch_executed':False,'format_consistency_training_executed':False,'independent_checks_passed':True,
        'platform_used':False,'training_data':'Official only','artifact_sha256':bound}
    save(ev/'delivery.json',delivery)
    catalogpath=ROOT/'mcp_readonly/catalog.json';cat=read(catalogpath)
    for doc in cat['documents']:
        if doc['id'] in ['v78-delivery','v78-direction-review']:
            doc['category']='historical_evidence' if doc['id']=='v78-delivery' else 'historical_review'
        if doc['path'] in ['README.md','docs/TRAINING_PLAN.md']:
            doc['sha256']=sha(ROOT/doc['path'])
    newdocs=[{'id':'v79-direction-review','title':'v7.9 真实训练、迁移反证与停止决定','path':'docs/V79_EXECUTION_RESULTS.md','category':'current_review','summary':summary,'keywords':['当前','最新','下一步','方向','训练','恶意','可疑','泛化','v7.9']},
             {'id':'v79-delivery','title':'v7.9 十五次拟合及未晋升交付证据','path':'evidence/2026-09-27/v79_execution/delivery.json','category':'current_evidence','summary':'14有效＋1空覆盖，完整开发回放5947错、M召回76.46%；质量未通过。','keywords':['实际结果','训练','验证','v7.9']},
             {'id':'v79-verification','title':'v7.9 模型重载和逐类独立复算','path':'artifacts/v79_execution_20260927/independent_verification.json','category':'verification','summary':'实现核验通过，模型质量不通过；空CEF覆盖实验排除。','keywords':['核验','v7.9']}]
    for doc in newdocs:doc['sha256']=sha(ROOT/doc['path'])
    newids={doc['id'] for doc in newdocs}
    cat['documents']=newdocs+[doc for doc in cat['documents'] if doc['id'] not in newids];cat['project'].update({'as_of':'2026-09-27','current_summary':summary,
        'authoritative_delivery_id':'v79-delivery','authoritative_direction_id':'v79-direction-review',
        'current_direction':['停止当前辅助损失/全局权重和无稳定性扩训；保留已验证解析修复及原诊断参考。','优先核验跨格式真实事实对齐与ASA M/S可观察反例；必须带正常负例验证。','先证实顺序或字段绑定缺口再启动有序模型，VPC零M支持和稀缺切片保持未解决。'],
        'known_limits':['完整开发回放退化，六工程门槛全未通过，无新模型晋升。','有序分支、格式一致性训练、任意未知格式支持和独立外部测试尚未完成。','VPC-M0/2664、ASA2536条S→M、Windows52条正常误报，仍需解决。','15次真实拟合中1次空CEF覆盖不作为有效迁移证据。']})
    save(catalogpath,cat)
    print(json.dumps({'summary':summary,'bound_files':len(bound),'report':str(reportpath)},ensure_ascii=False))


if __name__=='__main__':main()
