"""Publish actual V140 failures and scoped evidence; no training or selection."""
from pathlib import Path
import pandas as pd
from v140_runtime import ROOT,OUT,OLD,read,save,sha,require_run_seal


def main():
    require_run_seal(ROOT/'training/v140_train.py')
    if (OUT/'publication_receipt.json').exists(): raise FileExistsError('Preserve publication')
    d=read(OUT/'final_delivery.json');q=read(OUT/'quality.json');diag=read(OUT/'readonly_diagnosis.json');scope=read(OUT/'additional_verified_TRAIN_scopes.json')
    assert not d['decision']['issue_solved'] and not d['quality_acceptance'] and not d['model_promoted']
    summary='V140完成6次末层拟合、1200次全角色梯度评估、584次接受更新；候选集成CE纯TRAIN错16→14，修复8但新增6，第一问题未解决。40879条旧保护通过，C第2角色92114条纯TRAIN追加范围保护。ASA候选M/S错2036/1623，完整2056871行质量失败、无模型晋升。原链两轮12拟合已用完，停止同冻结表示末层路线。'
    metrics=q['arms']['E']['full_task']['B'];metriclines=[]
    for cl,name in [('0','N'),('1','M'),('2','S')]:
        v=metrics[cl];metriclines.append(f"|{name}|{v['support']}|{v['missed']}|{v['precision']:.9f}|{v['recall']:.9f}|{v['f1']:.9f}|")
    report='''# V140：第一问题第二轮实际训练与失败处置

2026-10-01。本机RTX4060 CUDA执行。'''+summary+'''

## 实际训练与单问题验收

在同折V138 H_L终点上做成员平均CE的C、平均概率CE的E对照。冻结主体与输入，保留全部合法TRAIN原行及频次、混标行；仅更新7,677个原末层参数。E为预登记候选，没有看结果改成C，也没有选中间状态。

|角色|V138纯TRAIN错|C纯TRAIN错|E纯TRAIN错|E修复/新增|
|---|---:|---:|---:|---:|
|0|14|12|12|6/4|
|1|0|0|0|0/0|
|2|2|0|2|2/2|
|合计|16|12|14|8/6|

候选全部残错为S，14条独立原行、7输入。C总修复4、新增0，但第0角色仍错12；不后验切换候选。E第2角色修好了旧2错却新增另外2错，不能只看净数。原行真值从官方label_binary独立加载，不使用预测自带真值作为权威。

六次拟合各200次全角色梯度评估，实际接受更新C为96/96/99，E为97/98/98，总584。每次闭包全原行质量核验；12次零步诊断梯度单列。所有梯度有限，200是预算终点，不是已证明收敛或全局最优。两次未接受末试探被回滚，终点和末五个实际不同状态独立重放；全部缓存实际主体重放一致。152项运行依赖/源数据绑定和包版本记录在run_seal.json，原V138封存保持有效。

## 保护与已学会范围

v138_retention_check对两臂及各末五状态分开执行：第1角色40,879条已登记原行新增错0。E仍新增6条其他旧正确纯TRAIN行，未满足更广的修复保护要求，不能关闭第一问题。

C第2角色末五状态实际重放均纯M/S零错、完整M零错、S仅28条混标经验最低错，旧正确纯输入新增0；单独冻结其92,114条纯TRAIN原行，见additional_verified_TRAIN_scopes.json和fold2_C_verified_TRAIN_guard.parquet。这只是额外范围能力，未变更候选E、未晋升C、不是全问题或来源外成功。后续必须运行v140_retention_check.py，它同时检查旧V138范围；本轮实际E在新第2角色范围新增2错会被拒绝，C通过。

## 来源外与完整三分类

|ASA来源留出|A0|V135|V138|本轮C|本轮E|
|---|---:|---:|---:|---:|---:|
|M错/78,748|318|1748|2006|1982|2036|
|S错/34,059|2074|1655|1637|1635|1623|

E相对V138：M修复98、新增128；S修复20、新增6。相对匹配C：M修复90、新增144；S修复18、新增6。相对A0：M修复8、新增1726；S修复467、新增16。来源、折、未知/缺失参数与全部其他格式保留。S来源均值召回18.1494%，零召回184，但不覆盖M回退。

完整官方2,056,871条原行指标：

|类|人口|漏判|precision|recall|F1|
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(metriclines)+'''

完整候选共3766错，A0为2499错，非ASA冻结组件仍107错。ASA M、总错、全部类指标保护、至少两折改善、header682及root2868等门槛失败；quality.json保留所有门槛，不追认失败。开发来源已查看，不称盲测，无比赛提交或真实网络迁移验收。

## 新反证与解释边界

[TabM官方实现](https://github.com/yandex-research/tabm)有意训练成员CE、推理平均概率，原方法不是实现bug。[联合损失合谋论文](https://arxiv.org/abs/2301.11323)提供风险反证，本轮固定主体读出对照不等于复现其完整实验。

E第0角色整体集成CE下降0.00164494→0.00106531，但成员CE上升0.00241971→0.04067453；末层weight距起点L2约527.53，C仅29.88。原14条残错输入的平均成员CE约2.184→44.990，实际概率呈接近正确成员数量/16的阶梯；其中local19653有8个正确成员而平均真类概率0.499913，仍判错。E第2角色也修复旧错同时制造新错。

这支持“集成目标可下降却产生极端成员分工和回退”的局部反证；不证明全部原因就是合谋，不证明无解、表示不完整或数据标签错误。成员责任、同支持正确对照、两种CE与概率间隔的完整只读诊断保留在member_responsibility_diagnostics.parquet。

## 失败处置、预算与下一步

原V137/V138末层链累计两轮12主拟合、2400主梯度评估、1475接受更新，预算耗尽。V139锚定仍未训练，不能现在再花6次；旧C_margin未取得资格且未执行。不得追加同冻结表示CE/锚定/种子/阈值搜索、改名重置预算或降低门槛。

完整三问题顺序保持：①训练侧分类未通过，②细行为同类支持不足未解决，③跨来源修正稳定性未解决。当前只推进①；下一机制必须先获得新的独立证据，核查剩余合法TRAIN输入、原始文本/事实与冻结表示的具体差异，区分优化饱和、目标与表示限制，再决定新的可训练范围。不能凭无精确冲突就宣称可分，也不能用HELD错例训练新目标。已验证第1/第2角色能力继续保护，不将局部成功冒充第一问题关闭。

两项报告工程问题已记录并修复：只读诊断空类别块使布尔列类型变化，跳过空块并显式布尔索引；手动保护CLI误传两臂，重复键被正确拒绝后改用E单臂账本。均发生在六拟合完成后，0新增拟合/更新；原训练源码与封存未变。过程测试不充作模型质量验收。

证据目录artifacts/v140_ensemble_training_round2_20261001/；final_delivery.json、quality.json、verification.json、attribution.json、源/运行封存、全量账本、修复/新错/残错及额外范围保护均保留。
'''
    target=ROOT/'docs/V140_SINGLE_ISSUE_ROUND2_RESULTS.md';target.write_text(report,encoding='utf-8')
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    catalog['project'].update(as_of='2026-10-01',current_summary=summary,current_direction=[
        '当前仍在第一问题；E残14条且新增6错，C第0角色残12条，均不能关闭整个训练分类缺口。',
        '保护旧第1角色40879原行及新C第2角色92114原行，后续调用v140_retention_check联合回归。',
        '原末层链两轮12拟合已用完，停止同冻结表示清零/集成CE/锚定路线，不执行旧C_margin。',
        '先对剩余合法TRAIN文本、事实与冻结表示取得新独立证据，再决定第一问题的新可训练范围；第二/第三问题保持开放。'],
        authoritative_delivery_id='v140-delivery',authoritative_direction_id='v140-review')
    catalog['project']['known_limits']=[
        '已查看开发来源，不是盲测、比赛提交或真实网络迁移验收。','E集成CE下降但新增6条旧正确训练错；成员极化不是无解证明。',
        '训练范围能力不授予模型晋升；完整质量失败，任务参照保持。','同冻结表示末层预算已耗尽，V139锚定未训练不得重置预算。',
        '正常/其他格式为冻结组件，非ASA仍107错；第二和第三问题尚未验收。']
    for ident,path,title,category in [
        ('v140-review',target,'V140第一问题第二轮结果、失败与后续证据边界','current_direction'),
        ('v140-delivery',OUT/'final_delivery.json','V140六拟合实际交付与全量质量','execution_delivery'),
        ('v140-extra-train-scopes',OUT/'additional_verified_TRAIN_scopes.json','V140额外第2训练角色范围保护','review_evidence')]:
        catalog['documents'].append({'id':ident,'path':path.relative_to(ROOT).as_posix(),'title':title,'category':category,'summary':summary,
                                     'sha256':sha(path),'keywords':['V140','实际训练','单问题','末层','保护','失败']})
    save(catalog_path,catalog)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    text=text.replace('# SOC 日志威胁检测项目\n','# SOC 日志威胁检测项目\n\n最新实际与当前方向（V140）：**'+summary+'** [实际训练、失败证据和后续边界](docs/V140_SINGLE_ISSUE_ROUND2_RESULTS.md)。\n',1)
    text=text.replace('当前方向（V139，未训练）','历史研究方案（V139，未训练，顺序由V140对齐）',1).replace('最新实际执行（V138）','历史实际执行（V138）',1)
    readme.write_text(text,encoding='utf-8')
    handoff=ROOT/'HANDOFF.md';(OUT/'handoff_before_v140.md').write_bytes(handoff.read_bytes())
    handoff.write_text('''# SF02训练续接：V140实际第二轮已完成

2026-10-01。完整目标与用户授权保持：逐一解决①训练侧学会分类，②细行为同类支持，③修正量跨来源稳定，之后审查完整三分类及其他真实缺陷。当前仍在①，不能提前关闭或机械跳到③。

'''+summary+'''

先读AGENTS.md、docs/EXPERIMENT_REVIEW_RULES.md、docs/V140_SINGLE_ISSUE_ROUND2_RESULTS.md、docs/V140_PRETRAIN_SINGLE_ISSUE_DECISION.md。核对README/只读MCP及活跃进程；真实目录C:\\Users\\xiabutian\\Desktop\\人工智能算法挑战杯\\SF02。本地.venv-v61 CUDA可用，不需再次申请平台。

实际证据artifacts/v140_ensemble_training_round2_20261001/：final_delivery.json、quality.json、verification.json、attribution.json、run_seal.json、原行账本、6终点及末五不同状态、member_responsibility_diagnostics.parquet与readonly_diagnosis.json。旧V138封存/模型未改，新封存包含152依赖项。

保护旧V138第1角色40879原行及新增C第2角色92114原行。后续调用training/v140_retention_check.py（内含旧v138_retention_check），读取additional_verified_TRAIN_scopes.json。本轮E在旧范围通过，但在新C第2角色范围会新增2错而被拒绝；不得隐藏。

原末层链最多两轮12拟合已全部用完，累计2400主梯度评估、1475接受更新；不因改版本重置。不追加同冻结表示清零/锚定或换种子；V139锚定仍零拟合，旧C_margin资格未过。本轮固定200闭包/拟合，未使用草案600。

下一步仍只解决第一问题：先从剩余合法TRAIN原行和同支持正确对照，核查完整原文/事实、冻结表示与极化优化的具体缺陷，取得能改变可训练范围的新独立依据；再实施新的审查、训练器、源码/依赖/输入封存、零步重放与有界训练。未获得依据前不盲扩容量或只追加同路线计算。保留所有合法原行与频次，不用HELD答案改权重/目标/阈值；候选须完整2056871行独立真值评分，质量失败不晋升。原始A0参照与质量门槛保持。

TabM成员CE/推理平均是作者有意设计，不是实现bug。E为非凸、概率责任梯度可能饱和；本轮损失下降并伴6条新错及极端成员分工，已保留实际反例。无精确特征冲突不证明可分或语义完整，LP超时不证明无解。C第2角色范围成功不允许后验切换候选E或宣称整体成功。

当前目标保持active，未完成全部三问题；有界末层试验已完成，不等于整个任务完成。不要只重复规划，后续必须依据新证据继续实现和实际验证。
''',encoding='utf-8')
    save(OUT/'publication_receipt.json',{'status':'published_actual_V140','report_sha256':sha(target),'catalog_sha256':sha(catalog_path),
                                       'additional_scopes_sha256':sha(OUT/'additional_verified_TRAIN_scopes.json'),'diagnosis_sha256':sha(OUT/'readonly_diagnosis.json'),
                                       'source_sha256':sha(__file__),'new_fits_after_terminal':0,'cloud_MCP_verified':False})
    print(summary,flush=True)


if __name__=='__main__':main()
