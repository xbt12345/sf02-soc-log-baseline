"""Publish completed geometry with factual and classifier-invariance limits."""
import json,shutil
from pathlib import Path
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v156_conditional_neighborhood import OUT,require,save

DOC=ROOT/'docs/V156_CONDITIONAL_NEIGHBORHOOD_RESULTS_AND_NEXT_CHECK.md'
CASE=ROOT/'training/review_policy/v156_observed_neighborhood_cases.json'

def main():
    require();assert not DOC.exists() and not CASE.exists()
    audit=read(OUT/'audit.json');verify=read(OUT/'verification.json')
    check_bindings(audit['source_sha256']);check_bindings(verify['source_sha256'])
    parent=ROOT/'artifacts/v156_independent_geometry_counterexample_20261001/audit.json'
    cp_source=ROOT/'training/v156_independent_geometry_counterexample.py';counter=read(parent)
    assert counter['source_sha256']==sha(cp_source) and counter['complete_predictions_identical']
    assert [z['nearest_class'] for z in counter['cases']]==[2,1] and all(z['max_logit_change']==0 for z in counter['cases'])
    x=pd.read_parquet(OUT/'fixed_cohort_relation_summary.parquet');outer=x[x.role.eq('outer_HELD')]
    cohort=outer.groupby(['space','cohort','truth','relation'])[['original_rows','baseline_errors','V155_B_errors']].sum().reset_index()
    def count(space,name,relation):
        s=cohort[cohort.space.eq(space)&cohort.cohort.eq(name)&cohort.relation.eq(relation)];return int(s.original_rows.sum())
    assert [count(s,'hard578','same_class_closer') for s in audit_space()]==[48,112,16]
    assert [count(s,'strict51','other_class_closer') for s in audit_space()]==[41,28,23]
    witness=pd.read_parquet(OUT/'selected_factual_witnesses.parquet')
    hard_same=witness[witness.cohort.eq('hard578')&witness.neighbor_kind.eq('same_class')]
    assert not hard_same.whole_fine_behavior_identical.any()
    bindings=[OUT/'audit.json',OUT/'verification.json',OUT/'fixed_cohort_relation_summary.parquet',OUT/'selected_factual_witnesses.parquet',parent,cp_source]
    save(CASE,dict(status='actual_fixed_geometry_and_counterexamples_not_new_training_authority',
        hard578_same_class_closer_by_space=[48,112,16],strict51_other_class_closer_by_space=[41,28,23],
        fine_identical_nearest_same_class_hard_witnesses=0,new_classifier_fits=0,new_updates=0,
        cases=['Global H2 same-class relation improves while hard578 relation worsens; cannot claim global information loss.',
            'Strict correct S can have an other-class nearest witness; cannot replace classifier or synthesize labels.',
            'Nearest same-class hard witnesses differ in actual fields; geometry is not exact fine-behavior support.',
            'Coordinate/head inverse scaling changes cosine order while all logits remain exactly identical.'],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in bindings}))
    rows=[]
    for space in audit_space():
        for cl in [1,2]:
            part=cohort[cohort.space.eq(space)&cohort.cohort.eq('all_ASA')&cohort.truth.eq(cl)]
            def val(relation,col):return int(part.loc[part.relation.eq(relation),col].sum())
            rows.append(f"|{space}|{cl}|{val('same_class_closer','original_rows')}|{val('same_class_closer','baseline_errors')}|{val('other_class_closer','original_rows')}|{val('other_class_closer','baseline_errors')}|{val('class_tie','original_rows')}|{val('no_same_class','original_rows')}|")
    table='\n'.join(rows)
    DOC.write_text(f'''# V156 条件邻近关系：完整实测与下一项核查

2026-10-01。**完整诊断已实际完成，新增拟合/梯度/参数更新0。最新实际分类拟合仍V155，匹配与完整质量失败，未晋升。** 本轮实测填补了全人口表示关系缺口，没有生成类别标签或证明第二第三问题通过。

## 真实范围与计算

[设计](V156_CONDITIONAL_NEIGHBORHOOD_DIAGNOSTIC_DESIGN.md)与机器方案在features/距离审查前固定，3601物理文件封存。19个直接输入/模型/缓存/概率文件逐SHA复核；V146 A端点与九个V155保存概率在本次封存中直接绑定，未靠它们生成之前的旧seal覆盖。

固定V146 A（V155共同起点），三折×三个空间；112807 ASA原行、22546 actual local、18协议/typed状态条件。全部未知/ICMP/混标/具体参数及原频次保留；每条查询排自身整个root，候选仅合法TRAIN，全部16成员展平、float64余弦，不合并canonical别名。root是隔离代理，不是已验证真实环境。

实际3次SecondRepresentation.features函数求值（每折22546×16×128输出），0完整分类器forward/梯度/拟合/更新；模型张量身份不变、grad为空。九份空间角色输出共1015263行，每空间225614 TRAIN角色＋112807 outer原行。角色和空间重复不是新增独立样本。正常/其他格式未新运行，原完整2056871行分类质量仍由V155凭据确定。

原行身份、405828 query/class单元的合法来源/状态/原类可用质量已复核；输入和H1所选见证余弦重算269410个，最大差{verify['selected_cosine_max_gap']:.12g}。H2为事前封存入口真实features计算；本轮复核其所选root/状态/类支持，没有额外独立重算H2余弦或模型。无支持、零向量和并列状态保留；并列计全部原类质量和root并集，最小local仅作见证，不投票。

## 全部来源外原行关系

以下“错误”是复用V146 A实际保存概率，不是新近邻分类器。类1=M，类2=S。

|空间|类|同类更近原行|其中原分类错|反类更近原行|其中原分类错|几何并列原行|无同类原行|
|---|---|---:|---:|---:|---:|---:|---:|
{table}

H2中同类更近的S为32197，H1为31756；其中原分类错分别34和180。整体关系改善与困难组退化同时存在，不能挑困难切片宣称整层丢失类别信息。无同类S共238条，其中8错、230正确；M缺反类1548条及两类都缺8条均分类正确，不把缺支持写成万能错误根因。

## 已登记困难与正确对照

578困难S的“同类更近”原行为输入48、H1 112、H2 16；三空间原分类仍576错。51条严格正确S中，“反类更近”输入41、H1 28、H2 23，但原分类全部正确。V155新增16M中，输入全部同类更近，H1/H2各8同类更近、8反类更近，V146 A本来全部正确。因此近邻标签不能取代原分类或当新监督。

困难组所选同类见证没有任何一项完整13细事实相同；多数改变源/目的端口，也有范围等具体字段变化。18状态条件只是可比观察状态，不能称新增相同细行为支持，不能用同标签删去这些差异。并列最近同类没有多root质量，不表示其他候选不存在或组合泛化不可能。

## 分类器与度量的界线

实际forward还含facts@head_facts.T直接通路和head投影；H1/H2余弦不覆盖它们。[父矩阵反例](../artifacts/v156_independent_geometry_counterexample_20261001/audit.json)对非负隐藏坐标和head做逆缩放，保留facts通路，全部logit/预测bit-exact不变，最近类却从S变M。这是0官方数据/模型调用的数学反例，不宣称本SOC表示任意或任何对比目标无效。

“字段可解码”“固定表示同类近邻”“真实分类器使用条件”“跨来源分类收益”是不同证据。此轮没有证明丢失信息的因果、全分类器学不会、可生成带标签近邻或新训练资格。V145/V146辅助和V155预算均不因本轮改名重启；完整原频次、未知/冲突、旧TRAIN保护和全任务各类门槛保持。

下一项可执行核查是：在合法TRAIN及原已登记控制中，核对完整分类函数对实际可观察组成的使用，包括head/facts分量与对应原行反例；不由余弦顺序直接指定新目标，不把条件相近的端口改造成威胁标签。新机制形成后另审单因素有限方案，当前不启动第七fit、不自动选层/阈值/检查点。

产物 artifacts/v156_conditional_representation_neighborhood_20261001/：机器方案/真实seal、原行/local参考、九邻近表、完整角色表、固定cohort汇总、事实差异见证、复核与反例。父[V155独立验收](V155_INDEPENDENT_RESULT_REVIEW_AND_NEXT_DECISION.md)保持原文件与质量失败结论。
''',encoding='utf-8')
    paths=[ROOT/z for z in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    snap=OUT/'previous_publication_snapshot';snap.mkdir()
    for p in paths:
        target=snap/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    save(snap/'manifest.json',dict(source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
    cp=paths[2];cat=read(cp);pr=cat['project'];assert pr['authoritative_delivery_id']=='v155-delivery'
    pr['authoritative_direction_id']='v156-neighborhood-review'
    pr['current_summary']+=' V156全人口条件邻近实测：3真实features求值、0分类器前向/梯度/拟合/更新，1015263空间角色行；几何不含完整head/facts通路，无新标签或新训练资格，完整目标未关闭。'
    pr['current_direction']=['最新实际分类器V155六拟合，完整质量失败、未晋升，停止该配置追加；旧TRAIN能力保持。',
        'V156已实际完成三空间三折全部原行条件邻近和来源/质量/余弦见证复核；3features功能求值，0新拟合。',
        '全人口与困难/正确反例同时保留；几何不替代完整分类器，不证明信息丢失、不生成标签或自动授予新训练。',
        '下一核查实际完整分类函数对可观察类别条件的使用；旧辅助/邻域预算不重启，完整目标仍active。']
    pr['known_limits'].append('V156仅固定表示条件几何；H2余弦未额外独立重算，完整分类还含head及facts通路。')
    entries=[('v156-neighborhood-review','V156条件邻近真实结果与下一项核查',DOC,'current_review'),
        ('v156-neighborhood-design','V156零拟合条件邻近设计',ROOT/'docs/V156_CONDITIONAL_NEIGHBORHOOD_DIAGNOSTIC_DESIGN.md','review_evidence'),
        ('v156-neighborhood-audit','V156九空间角色完整实测',OUT/'audit.json','review_evidence'),('v156-neighborhood-verification','V156合法来源类质量及余弦见证复核',OUT/'verification.json','review_evidence'),
        ('v156-geometry-counterexample','V156父预测不变而最近类翻转反例',parent,'review_evidence')]
    assert not {z['id'] for z in cat['documents']}.intersection(z[0] for z in entries)
    cat['documents']=[dict(id=i,title=t,path=p.relative_to(ROOT).as_posix(),category=c,summary=t,sha256=sha(p),keywords=['V156','当前','邻近','完整','反例','下一步']) for i,t,p,c in entries]+cat['documents']
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024;cp.write_bytes(raw)
    rp=paths[0];s=rp.read_text(encoding='utf-8');title,rest=s.split('\n',1)
    rp.write_text(title+'\n\n当前方向（V156实测）：[条件邻近完整结果与下一项核查](docs/'+DOC.name+')。实际3次features功能求值，0新分类器前向/梯度/拟合/更新；全人口与困难/正确控制同时核验，几何不是完整分类器或新标签。最新实际训练仍V155，质量失败、未晋升，第二第三/full未关闭。\n\n'+rest.lstrip('\n'),encoding='utf-8')
    hp=paths[1];hp.write_text(hp.read_text(encoding='utf-8')+'\n\n## V156条件邻近完整实测\n\n'+DOC.relative_to(ROOT).as_posix()+'；OUT '+OUT.relative_to(ROOT).as_posix()+'。3601封存、3真实features求值、0classifier forward/grad/fit/update。三fold三space1015263角色行，保留全部原行/未知/混标/实际local；自己root全部排除，candidate仅合法TRAIN。输入/H1所选余弦269410重算最大差6.9e-15；H2仅复核支持身份未新features重放。hard578同类更近input48/H1 112/H2 16，但全体S在H2关系更好；严格正确反例及精确logit不变余弦翻转阻止因果/新教师越级。下一完整函数条件使用核查，不开始第七fit、不选层/阈值、不重启旧预算。latest actual V155未晋升；第二第三/full仍active。\n',encoding='utf-8')
    tp=paths[3];tp.write_text(tp.read_text(encoding='utf-8').replace('v155-results-review','v156-neighborhood-review').replace('self.assertIn("V155", json.dumps(fetch_result.structured_content, ensure_ascii=False))','self.assertIn("V156", json.dumps(fetch_result.structured_content, ensure_ascii=False))'),encoding='utf-8')
    save(OUT/'publication.json',dict(status='completed_geometry_published_without_classifier_authority_change',catalog_bytes=len(raw),latest_actual_classifier='V155',
        new_classifier_fits=0,new_updates=0,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),DOC,CASE,parent]+paths}))
    print(json.dumps(dict(published=True,latest_actual_classifier='V155',current_direction='V156',catalog_bytes=len(raw))))

def audit_space():return ['actual_CSR_input','all16_H1','all16_V146_A_H2']

if __name__=='__main__':main()
