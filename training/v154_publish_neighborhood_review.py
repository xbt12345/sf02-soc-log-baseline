"""Publish completed finite stress evidence; keep V146 the actual classifier."""
import json
from pathlib import Path
from experiment_review import read,sha,check_bindings
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/v154_directional_neighborhood_v2_20261001'
OLD=ROOT/'artifacts/v154_directional_neighborhood_20261001'
DOC=ROOT/'docs/V154_DIRECTIONAL_NEIGHBORHOOD_RESULTS_AND_DECISION.md'
CASE=ROOT/'training/review_policy/v154_observed_neighborhood_cases.json'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not DOC.exists() and not CASE.exists() and not (BASE/'publication.json').exists()
    a=read(BASE/'audit.json');check_bindings(a['source_sha256'])
    assert a['full_classifier_gradient_evaluations']==18 and a['classifier_forward_chunk_calls']==432
    assert a['classifier_fits']==a['persistent_parameter_updates']==0
    assert all(s['joint_TRAIN_retention_passed'] for s in a['outer_summaries'])
    assert read(BASE/'verification.json')['status']=='actual_probe_recount_passed'
    report_bindings=[BASE/'audit.json',BASE/'joint_TRAIN_retention_checks.json',BASE/'verification.json',OLD/'failure.json',
        ROOT/'docs/V153_TRAINING_TRANSFER_GAP_AND_NEXT_MECHANISM_REVIEW.md',
        ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/literal_scope_qualification.json',
        ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/scalar_score_order_capacity.json']
    cases=dict(status='finite_directional_loss_sensitivity_not_transfer_cause',latest_actual_classifier='V146',new_classifier_fits=0,
        new_persistent_updates=0,actual_full_gradients=18,actual_classifier_forward_chunk_calls=432,
        automatic_SAM_training_qualified=False,quality_acceptance=False,issue_solved=False,cases=[
            dict(id='V154-LOSS-NOT-CLASS-FAILURE',full_gradients=18,stress_points=18,base_points=6,
                all_stress_points_joint_TRAIN_protection_passed=True,pure_TRAIN_new_errors=0,
                action='Loss rise at these fixed points does not reopen mastered TRAIN classification or close source transfer. Keep both loss and decisions.'),
            dict(id='V154-HARD-REPAIR-COLLATERAL',A_plus_hard_repairs=2,B_plus_hard_repairs=4,
                A_plus_ASA_M_errors=2052,B_plus_ASA_M_errors=2006,A_base_ASA_M_errors=1904,B_base_ASA_M_errors=1912,
                action='Small hard-cohort repair with M deterioration cannot qualify a replacement or select this development point.'),
            dict(id='V154-NET-GAIN-STRICT-S-REGRESSION',A_minus_ASA_total_errors=3453,B_minus_ASA_total_errors=3491,
                A_base_ASA_total_errors=3533,B_base_ASA_total_errors=3541,minus_strict_correct_S_new_errors_per_arm=6,
                action='Lower ASA aggregate errors do not erase the observed S deterioration, establish causation or select a new checkpoint.'),
            dict(id='V154-RANDOM-FINITE-BOUNDARY',random_same_L2_ASA_decision_changes_per_arm=0,
                radii=.001,random_directions_per_endpoint=1,
                action='One random direction and two gradient directions at one relative radius cannot prove all neighborhoods flat or all SAM ineffective.'),
            dict(id='V154-INDEX-ONLY-EXECUTION-ERROR',attempt1_classifier_forwards=0,attempt1_classifier_gradients=0,
                index_reset_all_row_values_and_dtypes_equal=True,attempt1_source_and_seal_retained=True,
                action='Preserve original source/seal/failure; repair only index comparison in new v2. No classifier budget reset or overwritten evidence.'),
            dict(id='V153-INDEPENDENT-REVIEW-NOW-LINKED',scope='Parent reports saved-probability identity, narrow 51 controls, literal address scopes and scalar-score oracle capacity',
                action='Do not infer new forwards, calibration rules or classifiers from these read-only audits; parent-owned files remain unchanged.')],
        source_sha256={z.relative_to(ROOT).as_posix():sha(z) for z in report_bindings})
    save(CASE,cases)
    train_table=[]
    for r in a['endpoints']:
        pts=r['points'];base=pts[0]['TRAIN_member_CE'];ratios=[x['TRAIN_member_CE']/base for x in pts[1:]]
        train_table.append('|'+str(r['fold'])+'|'+r['arm']+'|'+f"{base:.9f}|{ratios[0]:.4f}|{ratios[1]:.4f}|{ratios[2]:.6f}|{r['base_plus_gradient_cosine']:.4f}|0|")
    outer_table=[]
    names={'base':'原终点','plus_TRAIN_gradient':'+完整 TRAIN 梯度','minus_TRAIN_gradient':'−完整 TRAIN 梯度','fixed_random':'固定随机'}
    for arm in ['A','B']:
        for point in names:
            subset=[s for s in a['outer_summaries'] if s['arm']==arm and s['point']==point]
            whole=next(s for s in subset if s['cohort']=='all_ASA');hard=next(s for s in subset if s['cohort']=='known_578');strict=next(s for s in subset if s['cohort']=='strict_correct_51')
            m,s=whole['outer_class_errors'][1:]
            outer_table.append(f"|{arm}|{names[point]}|{m}|{s}|{m+s}|{hard['outer_class_errors'][2]}|{strict['outer_class_errors'][2]}|{whole['outer_repairs']}/{whole['outer_new_errors']}|")
    DOC.write_text('''# V154 固定方向参数邻域实测与决定

2026-10-01。**六个 V146 实际终点在固定相对 L2 半径 0.001 下，完整 TRAIN 梯度方向的成员 CE 上升明显，同等 L2 的固定随机方向影响很小；所有探查点的 TRAIN 分类、冲突下限及旧联合正确保护仍通过。来源外有 M/S 代价，未证明邻域损失敏感性是迁移根因，也未授予 SAM 训练资格。** 最新实际分类器仍 V146。本轮实际 18 次完整分类梯度、432 次分类器分块前向、18 个功能式扰动点和 6 个原终点评估；0 新分类拟合、0 优化器更新、0 永久参数更新、0 模型晋升。第二、第三及完整任务保持未通过。

## 从未测过的前提到有限实测

[V133](V133_TARGETED_TRAINING_DESIGN_AND_RESEARCH.md)暂缓 SAM 的原因是尚未测本题参数邻域尖锐性，并非已证明 SAM 无效。本轮仅检验这个前提的一小部分。[SAM 原研究](https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/)提出邻域低损失目标；其他数据集收益没有移植成本题收益。方案在模型梯度前固定并绑定，没有扫描半径、按外折答案调整方向或选择点。

入口 `training/v154_directional_neighborhood_v2.py`、方案 `training/review_policy/v154_directional_neighborhood_plan_v2.json`。逐折逐臂读取 V146 A/B 六个实际 checkpoint；只对既有第二层 22528 个参数做 `torch.func.functional_call` 替代，不改模块存储。第一层、参考第二层、读出与 facts_direct 均冻结。保留全部16成员，使用原频次成员 CE，不换成集成概率 CE、不改变 canonical/local 数值。

每终点：原点完整合法 TRAIN 梯度两次，loss/质量/梯度逐元素精确重复；正梯度扰动点再算一次完整梯度。固定扰动为 `±ρg/||g||`、固定 Rademacher 随机向量，其中 `ρ=0.001||θ||`。随机种子17454+fold，A/B同折同随机方向，均等相对 L2 大小。每终点三个功能式扰动点加原点完整22546个实际local概率，共六终点18梯度、432分块前向。所有112807 ASA原行和每臂225614合法TRAIN角色原行计入，包括未知、ICMP、冲突及混标多数正确记录。根和外折不进入新输入。

执行前继承 V146 全部物理绑定，并绑定本版入口、方案、验证器、真实模型/数据/缓存、已载入源码、Python和Torch DLL，共3577项；每终点运行前复核。原点功能式概率与对应V146保存概率逐元素完全相同。每终点参数张量哈希前后相同，模型参数 `.grad` 保持空；没有持久优化状态、学习 checkpoint 或选模。

## 损失敏感性与 TRAIN 分类分开

下表倍率以该终点原始 TRAIN 成员 CE 为1，数字小不能省略原损失，也不能把倍率当绝对分类损害。

|折|臂|原 TRAIN 成员 CE|+梯度倍率|−梯度倍率|随机倍率|g与g+余弦|新错纯 TRAIN|
|---|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(train_table)+'''

全部24个原点/探查点的三折 TRAIN 分类均为 M0、S22/6/28，纯输入错0，原已正确 TRAIN 新错0。八组完整三折账本实际调用 `v142_retention_check.py`，联合V142/V140/V138旧保护，均通过；225558正确角色保护包括混标多数正确，不能误写成225202纯输入角色。探查带来概率/损失变化，但没有重新打开这个已验收的 TRAIN 分类范围。

正梯度点的梯度与原梯度余弦约0.79–0.90，显示方向变化；负方向在这个固定有限步长也可提高CE。它不是无穷小梯度下降步骤，不能称优化器失效或在本轮补做回溯搜索。

## 全 ASA 来源外的代价，不挑最佳探查点

以下使用各原行自己未参与分类拟合的那个折终点，全部112807原行；578及51只作已看开发数据的描述控制，不参与扰动方向、半径或规则选择。这里的“来源外”是既有根隔离折，根不是已确认真实环境域。六终点原点重放不是新的盲测。

|臂|点|ASA M错|ASA S错|ASA总错|原578中S错|严格51中S错|相对原点修复/新错|
|---|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(outer_table)+'''

正方向A/B只修复原578中的2/4，却使M错误上升148/94；不能以该群少错宣布修复或拟合资格。负方向总错下降80/50，但S新错14、严格正确51退化6；总量下降不能掩盖类别和正确控制代价。固定随机方向的ASA分类完全不变。所有点均远未达到既定完整任务类别要求，未保存替换 checkpoint，未据开发结果选方向。

这些点是参数压力诊断，**不是SAM训练轨迹、接受更新或新算法收益**。并未重放其他格式完整2056871行，不把ASA局部结果加常数冒充完整新模型验收。此轮可以拒绝“损失敏感就等于分类失稳/修复迁移”和“少数困难修复就够”的推断，不能排除所有SAM、所有参数邻域或所有新模型。

## 已完成的父 V153 资格证据同步

[父V153独立报告](V153_TRAINING_TRANSFER_GAP_AND_NEXT_MECHANISM_REVIEW.md)和实际字面范围/标量容量凭据已加入只读目录，父拥有的文件未改写。原578在两个合法TRAIN角色均0错、最低pS约0.991，自身outer仍576错；严格正确对照51与困难组同字面CGN→10/8，字面类别不能分开本组。当前固定分数即使看开发答案枚举全局阈值，在M错≤318时S最少2772；不能通过标量阈值或严格单调分数校准达成门槛。这是oracle容量诊断，不是新推理规则或盲测结果，也不排除全部多变量校准。

父这些审查没有新模型前向；本轮自己的432次实际分类器前向另计。字段可读性、根支持表、静态阈值不能重新包装成训练收益。旧pAUC、对比/原型与MLDG/MAG已有历史，不因本轮重新检索重启预算。

## 执行修复与证据边界

首版V154在首个probe的模型加载/前向/梯度之前，比较保留原非连续索引的DataFrame与Parquet回读RangeIndex而失败；实际92337行值与dtype在重置索引后完全相同。首版入口、方案、3571项seal、registration与failure.json原样保留。v2只修索引比较并创建新入口/方案/输出，不覆盖旧运行，不重复任何已算分类梯度；本轮总分类梯度仍18。两版注册各有一个用于解析runtime的1×1 dummy前向/梯度，合计2/2，明确不是分类器计算。

`v154_verify_neighborhood_v2.py`实际重算六终点保存梯度一致性、±扰动方向/范数，24份概率/TRAIN行账本与真实角色/频次、完整ASA错误及V142保护人口，terminal exit0。全部原点精确对应旧保存概率；联合保护也来自实际调用，不用自填布尔替代计数。

单半径三个方向只给有限压力点，不估计球内最大损失、Hessian或因果比例。[Dinh等一手研究](https://proceedings.mlr.press/v70/dinh17b.html)指出参数重参数化可改变尖锐性而不改变泛化；因此本L2并非参数化不变尺度。[Friendly-SAM作者研究](https://arxiv.org/abs/2403.12350)区分完整梯度和小批随机噪声分量；本完整原频次梯度诊断不能替代实际随机SAM试验的机制证明，也不宣称它在本题有效。

本轮固定预算已执行完，不追加半径、种子、方向或拟合。当前没有新正式分类拟合登记；邻域损失敏感这一前提获得有限支持，但它与真实迁移收益的关系仍未建立。继续新类别条件机制审查时，不回到已掌握TRAIN、已读对字段或本轮探查点选模；若后续获得具体新因素依据，才按[V152条件式计划](V152_NEXT_TRAINING_DECISION_AND_CONDITIONAL_PLAN.md)登记有界两臂三折、旧联合保护及全2056871逐类真实验收。完整目标active，不把诊断完成称为第二/第三或全任务完成。

原始产物：`artifacts/v154_directional_neighborhood_v2_20261001/`，每终点保存完整梯度、固定扰动向量、各点实际概率、TRAIN原行、执行计数与模型前后身份；合并outer/TRAIN账本、联合保护与audit独立保存。机器反例 `v154_observed_neighborhood_cases.json`；可运行原行回放 `v154_replay_observed_cases.py`。
''',encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';cat=read(cp);pr=cat['project'];assert pr['authoritative_delivery_id']=='v146-delivery'
    pr['authoritative_direction_id']='v154-neighborhood-review'
    pr['current_summary']+=' V154实际18完整梯度/432分类器分块前向，18固定扰动点旧TRAIN联合保护通过，0新分类拟合/永久更新；损失敏感性未证明迁移根因或SAM资格。'
    pr['current_direction']=[
        '最新实际分类器V146；V154是已完成固定方向压力诊断，18完整梯度/432前向，0新拟合/永久更新/晋升。',
        '所有探查点M0/S22/6/28、纯TRAIN0错、V142/V140/V138联合保护通过；概率损失变化不重开已验收TRAIN分类范围。',
        '正方向少量困难S修复伴随M退化；负方向总量减少伴随S及严格正确对照退化。开发点不选模型，不证明因果。',
        '父V153概率身份、严格51、字面范围、全局标量容量已同步参考；根不是已确认域，旧方法不重置预算。',
        '固定预算已完，无新SAM或正式分类训练登记。第二第三/full未通过；有新机制依据才依V152有界拟合、旧联合保护和全2056871验收。']
    pr['known_limits']+=['V154一半径三个固定方向不是最大球内尖锐性、曲率或因果证明；不替代随机SAM实训。',
        'V154全部ASA为已看开发折；无其他格式全量新模型验收，损失变化不等于分类掌握变化。']
    entries=[('v154-neighborhood-review','V154固定方向邻域实测与训练资格决定',DOC,'current_review'),
        ('v154-neighborhood-results','V154实际18梯度432前向原行压力结果',BASE/'audit.json','review_evidence'),
        ('v154-neighborhood-cases','V154真实压力反例与动作',CASE,'review_evidence'),
        ('v154-neighborhood-verification','V154保存梯度及原行概率独立计数复核',BASE/'verification.json','review_evidence'),
        ('v153-independent-mechanism-review','V153父TRAIN迁移差距与下一机制独立审查',ROOT/'docs/V153_TRAINING_TRANSFER_GAP_AND_NEXT_MECHANISM_REVIEW.md','current_review'),
        ('v153-literal-scope-results','V153父脱敏字面范围候选资格实测',ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/literal_scope_qualification.json','review_evidence'),
        ('v153-scalar-capacity-results','V153父当前固定标量排序oracle容量',ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/scalar_score_order_capacity.json','review_evidence')]
    assert not {x['id'] for x in cat['documents']}.intersection(x[0] for x in entries)
    cat['documents']=[dict(id=i,title=t,path=z.relative_to(ROOT).as_posix(),category=c,summary=pr['current_summary'],sha256=sha(z),keywords=['V154','V153','当前方向','TRAIN','SAM','迁移','固定压力']) for i,t,z,c in entries]+cat['documents'];save(cp,cat)
    rp=ROOT/'README.md';s=rp.read_text(encoding='utf-8').replace('当前方向（V153）','当前方向（V154）',1)
    s=s.replace('[V153迁移与增量谓词审查]','[V154固定邻域实际与决定](docs/V154_DIRECTIONAL_NEIGHBORHOOD_RESULTS_AND_DECISION.md)；[父V153独立迁移与排序决定](docs/V153_TRAINING_TRANSFER_GAP_AND_NEXT_MECHANISM_REVIEW.md)；[V153迁移与增量谓词审查]',1)
    s=s.replace('V153新增实际：','V154新增实际：18完整梯度、432分类器分块前向，18功能式扰动点＋6原点；全部旧TRAIN联合保护通过。损失方向敏感，分类仍M0/S22/6/28；outer有M/S代价，未证明迁移根因，不授予SAM资格。0新分类拟合/永久更新，无晋升；首版索引执行错误在模型计算前失败，旧source/seal保留并另v2修复。\n\nV153新增实际：',1);rp.write_text(s,encoding='utf-8')
    hp=ROOT/'HANDOFF.md';s=hp.read_text(encoding='utf-8');s+='''

## 最新续增：V154有限方向邻域实测

当前docs/V154_DIRECTIONAL_NEIGHBORHOOD_RESULTS_AND_DECISION.md，actual分类器仍V146。执行v154_directional_neighborhood_v2.py register/六probe/summarize及v154_verify_neighborhood_v2.py。18完整分类梯度、432分块前向、18功能式扰动点＋6原点；0分类拟合/永久更新。固定ρ=.001||θ||，±合法TRAIN原频次成员CE梯度，seed17454+fold单随机Rademacher，同折A/B同随机方向；旧第二层22528 only，不改head/facts/第一层。6原点概率与旧V146逐元素同，6梯度重复精确，全部模型tensorhash前后相同。

全部点旧联合V142/V140/V138保护通过，M0/S22/6/28、纯TRAIN0、原正确新错0。正梯度loss原倍率1.2196–4.3423、随机1.000267–1.000567，说明有限损失敏感但不等于分类失稳。outer A/B base1904/1629、1912/1629；plus2052/1625、2006/1623，hard错574/572；minus1810/1643、1848/1643，strict51新增错6各臂；random决策不变。全ASA112807/TRAIN225614原行保留。没有用开发点选方向/新checkpoint，未全2056871新模型验收，不证明SAM训练有效/失效，当前无新拟合登记。

首版OUT artifacts/v154_directional_neighborhood_20261001/因索引比较非连续vsRangeIndex失败在模型加载/计算前，92337值/dtypes确切等价；旧入口、方案、seal3571、failure均保留。v2新入口/plan/output仅修reset_index比较，seal3577完整物理绑定。每版1个1×1 dummy前向/梯度，合计2/2另计，分类预算未重置。新OUT artifacts/v154_directional_neighborhood_v2_20261001/存真实梯度/方向/概率/TRAIN/outer账本/联合保护/audit/verification。父V153独立报告及字面scope/scalar容量已入MCP只读目录，父文件未改。下一新机制仍遵V152条件式有界两臂三折、旧联合保护与全2056871逐类验收；第二第三/full active。
''';hp.write_text(s,encoding='utf-8')
    tp=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=tp.read_text(encoding='utf-8').replace('v153-mechanism-review','v154-neighborhood-review').replace('self.assertIn("V153", json.dumps(fetch_result.structured_content, ensure_ascii=False))','self.assertIn("V154", json.dumps(fetch_result.structured_content, ensure_ascii=False))');tp.write_text(s,encoding='utf-8')
    save(BASE/'publication.json',dict(status='completed_finite_directional_diagnostic_published',latest_actual_classifier='V146',
        actual_full_gradients=18,actual_classifier_forward_chunk_calls=432,new_classifier_fits=0,new_persistent_updates=0,
        source_sha256={z.relative_to(ROOT).as_posix():sha(z) for z in [Path(__file__),DOC,CASE,cp,rp,hp,tp]}))
    print(json.dumps(dict(published=True,current_direction='V154',latest_actual_classifier='V146',new_classifier_fits=0,new_persistent_updates=0,quality_acceptance=False),ensure_ascii=False))

if __name__=='__main__':main()
