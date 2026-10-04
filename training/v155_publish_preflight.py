"""Publish actual zero-step evidence, not the still-unexecuted six fits."""
import json
from pathlib import Path
from experiment_review import read,sha,check_bindings
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/v155_guarded_full_gradient_sam_20261001'
DOC=ROOT/'docs/V155_ZERO_STEP_REVIEW_AND_EXECUTION_STATUS.md'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not DOC.exists() and not (BASE/'preflight_publication.json').exists()
    a=read(BASE/'preflight.json');s=read(BASE/'run_seal.json');check_bindings(s['source_sha256']);check_bindings(read(BASE/'fit_activation.json')['source_sha256'])
    assert a['full_classifier_gradients']==9 and a['classifier_forward_chunk_calls']==108 and a['joint_TRAIN_retention']['passed']
    assert not list(BASE.glob('fold*_*/started.json'))
    DOC.write_text('''# V155：新机制审查、真实入口与零步验收已完成

2026-10-01。**已登记新的单因素有界完整梯度一阶邻域风险对照，并完成入口、3579项物理封存、真实零步验收；尚未执行六次分类拟合。** 本轮实际9次完整分类梯度、108次分类器分块前向，0拟合、0永久更新。最新完成分类器仍V146，第二/第三及完整2056871任务未通过。

## 从限制性结论到有界试验资格

V154的有限方向损失敏感性没有证明迁移因果，但项目规则区分试验资格、质量通过和替换资格；允许可证伪对照不要求先证明最终因果。原来“尚未证迁移根因”只能阻止收益/替换声明，不能成为循环禁止新目标的理由。本轮据已有实测登记一次明示限制的对照，不选V154扰动点当学习终点、不扩大已用旧链预算。

[设计与实现审查](V155_NEIGHBORHOOD_TRIAL_DESIGN_AND_IMPLEMENTATION_REVIEW.md)和机器方案 `training/review_policy/v155_guarded_full_gradient_sam_plan.json`限定A普通完整原频次成员CE、B一阶固定扰动点梯度；同折V146 A共同初始化、相同归一化方向/步长/回溯、仅既有第二层22528参数，头/第一层/reference/facts冻结。ρ=.001×初始化范数，每fit恒定。B忽略ε对θ导数；proposal内冻结ε，仅验收固定ε代理Armijo与未扰动真实TRAIN分类，不能声称移动ε目标单调或精确球风险。

每fit最多200外层尝试/200接受更新/600proposal；A至多200完整梯度，B至多400，六fit最多1800。B额外计算如实记录，不称等算力。只从合法TRAIN构造方向，全部原行、未知、ICMP、混标及原频次保留；无外折答案选择、半径/系数/种子扫描、伪标签、新独立同类支持或自动确认。

## 本轮实际执行

`v155_runtime.py`适用资源终点profile实际检查通过，9个禁止变体被拒绝。`test_v155_sam_objective.py`3项数值检查通过：真实完整质量/一阶扰动梯度、固定ε有限差分/guard、移动ε总导数与一阶近似的实际反例。它们不证明SOC分类收益。

`v155_train.py register`在任何分类器计算/更新前封存3579物理文件，包括全部V146输入/模型/缓存依赖、自己的训练器/目标/评价器、真实划分与初始半径、runtime/Python/Torch DLL。setup有一个1×1 dummy前向/梯度，另计，不是分类器。

`preflight`对三折实际V146 A初始化各算原点完整概率、base完整梯度两次及shifted完整梯度一次：合计9梯度、108分块前向。全部22546原点概率与原V146 A保存文件逐元素完全相同，base loss/梯度重复精确，原点与正扰动风险重现V154。模型前后参数张量身份相同，model.grad为空；没有参数更新。

|折|原点成员CE|固定正扰动成员CE|本fit固定L2半径|注册TRAIN M/S错|
|---|---:|---:|---:|---:|
|0|0.000786811885|0.000972530257|0.087011115874|0/22|
|1|0.000537114586|0.002213811380|0.081300336876|0/6|
|2|0.000911345563|0.001116336221|0.086078650912|0/28|

三折全部纯TRAIN错0，旧V142/V140/V138联合check通过。父 `v155_independent_guard_containment.py`实际证实旧5个范围都包含于同role V146 A初始正确225558行，真值一致，所以每proposal保护全部初始正确行可涵盖这些旧链；终点和窗口仍调用联合check。包含关系仅对本初始化/数据有效，pure225202不能误写成correct225558。

零步凭据、原点完整概率与物理seal被 `fit_activation.json`再次绑定；拟合入口必须复核seal和activation。当前始发fit目录为0、学习checkpoint为0。这个状态不把注册fit预算6写成已完成6，更不把9个诊断梯度写成训练收益。

## 独立参考与未通过项

[父V154独立邻域/损失审查](V154_INDEPENDENT_NEIGHBORHOOD_AND_LOSS_REVIEW.md)已同步只读目录，父原文件未修改。其24点损失拆分限定于固定实际local输出：经验熵占原终点成员CE71.2%–88.4%，canonical微小概率别名不能合并为精确输出下界。CE构成不等于参数梯度构成；新fit将分别记录逐类/pure/mixed原频次风险，不删除冲突。

下一实际阶段是登记的A/B×3折有界拟合，然后逐模型/五状态/冻结ε代理重放、缓存无关数值输入验证、旧联合保护和完整2056871独立真值逐行验收。当前没有新分类收益，未关闭任何未通过目标，未降低A0、匹配效果、来源/少数组或完整三类标准。完整任务active，已看的外折只称开发验收。

当前运行产物 `artifacts/v155_guarded_full_gradient_sam_20261001/`；实际审查 `artifacts/v155_pretraining_method_review_20261001/review.json`。注册source/plan已封存不覆写；拟合/评价状态以后另有真实凭据，不修改本零步历史。
''',encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';cat=read(cp);pr=cat['project'];assert pr['authoritative_delivery_id']=='v146-delivery'
    pr['authoritative_direction_id']='v155-preflight-review'
    pr['current_summary']+=' V155单因素完整梯度邻域风险6fit已登记未执行；3579项封存、9实际零步梯度/108分块前向、三折概率/梯度/旧联合保护通过，0拟合/永久更新。'
    pr['current_direction']=[
        '最新完成分类器仍V146，V155已审查/登记/封存并实际零步，0新分类拟合/永久更新，不冒充新训练交付。',
        '受限试验资格不要求先证迁移因果；新一阶完整梯度邻域因素只允许有界对照，不允许质量/替换越级。',
        'A完整原频次成员CE/B固定ρ完整梯度一阶邻域风险，同V146 A初始化，同22528第二层/归一化Armijo；固定ε代理不称移动球风险单调。',
        '拟合最多6，单fit200外尝试/200接受更新/600proposal，A200/B400完整梯度；9零步梯度另计，无半径/种子/头/权重扫描或旧链预算重置。',
        '下一登记6拟合及完整2056871真实验收；旧联合保护保持，第二第三/full未通过，数据/已看开发边界不变。']
    pr['known_limits']+=['V155是full-original-frequency一阶邻域适配，非随机SAM作者benchmark复现，B额外计算不称等算力。',
        'V155当前仅zero-step，9梯度108前向不产生新独立同类支持；未执行的fit预算不能写成实际拟合。']
    entries=[('v155-preflight-review','V155新机制真实入口与零步执行状态',DOC,'current_review'),
        ('v155-trial-design','V155有界完整梯度邻域风险设计实现审查',ROOT/'docs/V155_NEIGHBORHOOD_TRIAL_DESIGN_AND_IMPLEMENTATION_REVIEW.md','review_evidence'),
        ('v155-zero-step-results','V155实际9梯度108前向零步与联合保护',BASE/'preflight.json','review_evidence'),
        ('v155-registered-plan','V155机器方案六拟合预算未执行',ROOT/'training/review_policy/v155_guarded_full_gradient_sam_plan.json','review_evidence'),
        ('v155-guard-containment','V155父实际旧三链包含于初始正确行证据',ROOT/'artifacts/v155_independent_guard_containment_20261001/audit.json','review_evidence'),
        ('v154-independent-neighborhood-review','V154父独立邻域与损失资格审查',ROOT/'docs/V154_INDEPENDENT_NEIGHBORHOOD_AND_LOSS_REVIEW.md','review_evidence'),
        ('v154-independent-neighborhood-results','V154父3577绑定与真实原行重算',ROOT/'artifacts/v154_independent_fixed_neighborhood_v3_20261001/audit.json','review_evidence'),
        ('v154-independent-loss-floor','V154父24点真实local损失熵拆分',ROOT/'artifacts/v154_independent_ensemble_ce_floor_v2_20261001/audit.json','review_evidence')]
    assert not {z['id'] for z in cat['documents']}.intersection(z[0] for z in entries)
    cat['documents']=[dict(id=i,title=t,path=z.relative_to(ROOT).as_posix(),category=c,summary=t,sha256=sha(z),keywords=['V155','V154','当前方向','零步','一阶邻域','TRAIN','未执行拟合']) for i,t,z,c in entries]+cat['documents']
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024
    cp.write_bytes(raw)
    rp=ROOT/'README.md';s=rp.read_text(encoding='utf-8').replace('当前方向（V154）','当前方向（V155零步）',1)
    s=s.replace('[V154固定邻域实际与决定]','[V155实际零步与下一有界对照](docs/V155_ZERO_STEP_REVIEW_AND_EXECUTION_STATUS.md)；[V155设计实现审查](docs/V155_NEIGHBORHOOD_TRIAL_DESIGN_AND_IMPLEMENTATION_REVIEW.md)；[父V154独立邻域与损失审查](docs/V154_INDEPENDENT_NEIGHBORHOOD_AND_LOSS_REVIEW.md)；[V154固定邻域实际与决定]',1)
    s=s.replace('V154新增实际：','V155新增实际：新单因素有界完整梯度一阶邻域风险6fit登记未执行。真实入口/3579项物理seal/9实际零步完整梯度/108分块前向完成；3fold原点概率exact、梯度重复与V154风险重现、旧联合保护通过，0拟合/永久更新。A/B共同V146 A初始化/第二层/归一化Armijo，固定ε代理不能称移动球风险单调。受限试验资格不要求先证因果，但无质量/替换越级、无半径/seed/旧链预算重置。\n\nV154新增实际：',1);rp.write_text(s,encoding='utf-8')
    hp=ROOT/'HANDOFF.md';s=hp.read_text(encoding='utf-8');s+='''

## 最新续增：V155真实入口封存与零步，六拟合尚未执行

current docs/V155_ZERO_STEP_REVIEW_AND_EXECUTION_STATUS.md；formal设计 docs/V155_NEIGHBORHOOD_TRIAL_DESIGN_AND_IMPLEMENTATION_REVIEW.md / plan training/review_policy/v155_guarded_full_gradient_sam_plan.json。latest completed classifier V146，不误记V155拟合已完成。V155修正过窄资格解释：试验可检验新目标，不必先证迁移因果；仍不能用局部方向或低loss越级质量/替换。

新源 v155_sam_objective.py / v155_runtime.py / v155_train.py / v155_evaluate.py / test_v155_sam_objective.py。3数值method检查通过（fixed epsilon差分/质量/first-order≠moving epsilon exact derivative），9禁止变体拒绝。registered max6fit，A普通原freq成员CE，B完整梯度一阶邻域风险；同fold V146 A固定终点起点，仅22528既有第二层，固定ρ=.001初始化norm，不scan；共享V146 normalized Armijo，B gplus忽略epsilon导数，proposal冻当前epsilon L(theta_trial+epsilon)，不声明移动球risk单调。A200/B400 fullgrads每fit，200外尝试/200接受/600proposal，最大总1800，额外计算如实记；无自动repeat/confirmation/旧head budget重置。

OUT artifacts/v155_guarded_full_gradient_sam_20261001/；register封存3579physical（含所有原V146依赖＋自己的trainer/objective/evaluator/role/radius/runtime）；1×1 dummy 1forward/1grad另计。preflight实际9完整分类梯度/108分块forward、0fit/0永久更新。三fold q与V146 A saved probs逐元素exact，base gradients重复exact，base/plus成员CE重现V154，联合旧三链通过、modelhash前后同/grad None。fixed radius fold0 .0870111158736196/fold1 .08130033687625052/fold2 .08607865091229709。preflight和3zero_probability已bind fit_activation；代码/plan/risk已seal，不覆盖。

父guard containment actual artifacts/v155_independent_guard_containment_20261001/audit.json：旧5范围对应真值、role均包含V146 A初始正确225558行；每proposal维持all initial correct足以覆盖旧guards，但endpoint/window仍actual联合check。pure225202另报。父V154 independent geometry/loss报告与audits已同步MCP；经验熵按actual local限定、CEcomposition不等gradientcomposition、完整目标仍active。

下一执行 registered six fits：python training/v155_train.py fit --fold 0/1/2 --arm A/B（顺序fold0A/B→fold1A/B→fold2A/B，每fit复核seal/activation，无overwrite/resume）；全部实际完成后 python training/v155_evaluate.py。它重放last5/endpoint/frozen epsilon风险、cache-free22546输入、joint retention后score完整2056871；nonASA继承旧V146冻结preds独立计分，不称重新运行其模型。0新私有valid答案或外折选择；仍previously inspected development。尚未启动fit，不能只根据plan/zero step判已解决第二第三或全task。
''';hp.write_text(s,encoding='utf-8')
    tp=ROOT/'mcp_readonly/tests/test_readonly_mcp.py';s=tp.read_text(encoding='utf-8').replace('v154-neighborhood-review','v155-preflight-review').replace('self.assertIn("V154", json.dumps(fetch_result.structured_content, ensure_ascii=False))','self.assertIn("V155", json.dumps(fetch_result.structured_content, ensure_ascii=False))');tp.write_text(s,encoding='utf-8')
    save(BASE/'preflight_publication.json',dict(status='actual_preflight_published_six_fits_not_executed',latest_actual_classifier='V146',
        full_classifier_gradients=9,classifier_forward_chunk_calls=108,classifier_fits=0,updates=0,catalog_bytes=len(raw),
        source_sha256={z.relative_to(ROOT).as_posix():sha(z) for z in [Path(__file__),DOC,cp,rp,hp,tp]}))
    print(json.dumps(dict(published=True,current_direction='V155-preflight',latest_completed_classifier='V146',fits=0,updates=0,catalog_bytes=len(raw))))

if __name__=='__main__':main()
