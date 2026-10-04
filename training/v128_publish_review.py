"""Publish V128 research direction while retaining the V127 actual delivery."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    receipt=ROOT/'artifacts/v128_mechanism_review_20260929/verification.json'
    check=json.loads(receipt.read_text(encoding='utf-8'))
    if check['status']!='V128_no_fit_evidence_and_role_preparation_verified' or check['classifier_fits']!=0:
        raise ValueError('Review not verified')
    for path,h in check['artifact_sha256'].items():
        if sha(ROOT/path)!=h:raise ValueError('Review file changed '+path)
    intro=('当前方向（V128，仅审查与方案、未训练）：**复核V127发现纠错训练分数的暴露角色缺口：'
        'P判错的326条M在两个训练内基座中均判对，1980条S错例中1053条也均判对、895条则两者均错。'
        '正文冲突经验下界116，不等于组合模型上界。上一报告1/1/2个输入翻转对应原行2/10/4条。'
        '下一轮拟做嵌套来源交叉分数与配对训练内分数×常量/正文四臂：9个共享教师+12个修正拟合，共21次；'
        '角色清单已核验，新训练器及执行封存尚未实现。保留M/S、来源组和全量门槛；本轮0次训练、无模型晋升。** ')
    for name,link in [('README.md','docs/V128_SCORE_ROLE_REVIEW_AND_NEXT_PLAN.md'),
                      ('docs/TRAINING_PLAN.md','V128_SCORE_ROLE_REVIEW_AND_NEXT_PLAN.md')]:
        p=ROOT/name;s=p.read_text(encoding='utf-8')
        if '当前方向（V128' in s:raise ValueError('Already published '+name)
        title,rest=s.split('\n',1)
        p.write_text(title+'\n\n'+intro+'[本轮根因证据与下一轮对照方案]('+link+')。最新实际训练仍为V127，质量失败。\n'+rest,encoding='utf-8')
    cp=ROOT/'mcp_readonly/catalog.json';c=json.loads(cp.read_text(encoding='utf-8'))
    p=c['project']
    if p['authoritative_delivery_id']!='v127-delivery':raise ValueError('Unexpected current training')
    p['authoritative_direction_id']='v128-review'
    p['current_summary']=('最新实际训练V127：三折三臂9次新拟合、17800次网络更新、0次校准，质量失败，无模型晋升。'
        '当前V128仅审查和方案：326条M错例在两个训练内基座都判对，1980条S错例中1053条都判对、895条都判错。'
        '准备了来源交叉分数四臂21拟合方案及角色清单；本次未训练，运行契约尚未实现。')
    p['current_direction']=[
        '先检验纠错训练的分数角色：在每外折内建立合法内层教师，IS/CF与常量/正文四臂匹配；现成其他外折输出不能冒充合法CF。',
        '计划9个共享教师、6个常量和6个正文拟合，共21次；角色已准备，训练器和执行封存未完成，不能直接运行历史入口。',
        '唯一候选P_CF必须超过匹配P_IS和K_CF，并保持A0逐类/来源保护；角色修复不等于语义判据补齐。',
        '历史翻转单位补充：P相对训练均值残差1/1/2个不同输入等于2/10/4条原始记录，旧分类失败结论不变。']
    p['known_limits']=[
        '同记录训练内/外比较涉及不同教师与训练人口，只支持待检验假说，不能当因果证明。',
        '895条S在两个训练内基座均错；正文混标组外仍有208条M、1864条S错误，信息不足或模型缺口尚未完全分离。',
        '正文与合法记录端口的经验冲突28；加原始脱敏字符串可变0，但样本可区分不等于安全语义充分。',
        '原V127正文P错误M326/S1980，K326/1930；历史头部、小组和未知参数问题未解决。',
        '仅官方开发数据；没有新的训练、独立官方测试或真实企业迁移验收。']
    entries=[
        ('v128-review','V128分数角色根因审查与下一轮方案','docs/V128_SCORE_ROLE_REVIEW_AND_NEXT_PLAN.md','current_review','V127真实错误分层、翻转单位勘误、嵌套IS/CF四臂21拟合计划；本轮未训练。'),
        ('v128-plan','V128下一轮机器方案','training/review_policy/v128_next_training_plan.json','current_plan','角色已准备，运行契约未实现；21次主拟合，唯一候选P_CF，无自动确认。'),
        ('v128-audit','V128正文信息与原行单位复算','artifacts/v128_mechanism_review_20260929/audit.json','current_evidence','正文冲突116；P输入翻转1/1/2对应原行2/10/4；无拟合。'),
        ('v128-score-roles','V128同记录训练内外分数审查','artifacts/v128_mechanism_review_20260929/score_role_audit.json','current_evidence','326条M错误训练内均正确；1980条S错例按1053/32/895分层；因果边界明确。'),
        ('v128-verification','V128无训练证据核验','artifacts/v128_mechanism_review_20260929/verification.json','current_evidence','243个V127绑定文件未变；原行勘误及嵌套角色检查；不代表训练质量通过。'),
        ('v128-nested-roles','V128嵌套角色准备','artifacts/v128_mechanism_review_20260929/nested_role_preparation.json','current_evidence','9个内层教师角色已定义，两类均存在；0次新教师训练。'),
        ('v128-risks','V128历史反例执行动作','training/review_policy/v128_risk_actions.json','current_plan','训练分数角色污染、单位混淆、常量优势及来源小组退化等约束。')]
    if any(e['id']=='v128-review' for e in c['documents']):raise ValueError('Catalog already published')
    new=[]
    for ident,title,path,cat,summary in entries:
        new.append({'id':ident,'title':title,'path':path,'category':cat,'summary':summary,
                    'sha256':sha(ROOT/path),'keywords':['V128','当前','训练方案','来源交叉预测']})
    c['documents']=new+c['documents']
    for e in c['documents']:
        if e['path'] in ('README.md','docs/TRAINING_PLAN.md'):e['sha256']=sha(ROOT/e['path'])
    cp.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    test=ROOT/'mcp_readonly/tests/test_readonly_mcp.py'
    s=test.read_text(encoding='utf-8').replace('v127-results','v128-review')
    test.write_text(s,encoding='utf-8')
    print(json.dumps({'direction':'v128-review','actual_delivery':'v127-delivery','new_training':0,'new_catalog_entries':len(new)},ensure_ascii=False))


if __name__=='__main__':main()
