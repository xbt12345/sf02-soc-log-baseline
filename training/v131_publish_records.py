"""Publish verified V131 records to the project read-only index. Never trains."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v131_learning_trial_20260930'
REPORT='docs/V131_LEARNING_QUALIFICATION_TRAINING_RESULTS.md'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    d=read(OUT/'delivery.json');v=read(OUT/'verification.json')
    assert d['status']=='v131_completed_learning_passed_quality_failed'
    assert (d['classifier_fits'],d['optimizer_steps'])==(8,31400)
    assert v['all_checks_passed'] and sha(OUT/'verification.json')==d['verification_sha256']
    assert d['quality_acceptance'] is False and d['model_promoted'] is False
    assert (OUT/'diagnostics/receipt.json').is_file() and (OUT/'result_audit/receipt.json').is_file()
    issues={'version':'V131','scope':'Observed execution outcomes, not universal laws.',
        'evidence_sha256':{(OUT/'delivery.json').relative_to(ROOT).as_posix():sha(OUT/'delivery.json'),
            (OUT/'learning_selection.json').relative_to(ROOT).as_posix():sha(OUT/'learning_selection.json'),
            (OUT/'result_audit/audit.json').relative_to(ROOT).as_posix():sha(OUT/'result_audit/audit.json')},
        'cases':[
            {'id':'TRAIN_PASS_TRANSFER_FAIL','observed':'All fold1 arms pass TRAIN gates, R100 ASA M318->1956; no source fold improves total errors.',
             'required_action':'TRAIN qualification grants a transfer check only. Reject release and additional fitting when actual source/class quality fails.',
             'next_scope':'Nested source validation must separately protect M/S; no held-driven thresholds.'},
            {'id':'PERFECT_HARD_S_CAN_HIDE_M_DAMAGE','observed':'CO epoch60 hardS0, M52, matchedM44; epoch100 M0,S6.',
             'required_action':'Keep fixed endpoints and full-class checks; never select by difficult-S repair alone.',
             'executable_replay':'training/test_v131_observed_cases.py'},
            {'id':'LOW_SUPPORT_IS_NOT_SUFFICIENT_CAUSE','observed':'New M errors1566/1638 lack full-fact M support, but retained-correct M62618 also lack that support.',
             'required_action':'Include equally unsupported correct controls before claiming lack of support causes all errors; no automatic upweighting.'},
            {'id':'TRAIN_SUCCESS_HAS_ROLE_BOUNDARY','observed':'Fold1 nonmixed hardS527->14; fold0/2 still have nonmixed S28/41 and fold0 M6.',
             'required_action':'Keep per-role residuals; do not call all TRAIN learning problems resolved.'},
            {'id':'ALTERNATIVE_HYBRID_IS_NOT_UNIFORM_CANDIDATE','observed':'O/C/CO have new fold1 only; whole-population scores use old A0 elsewhere.',
             'required_action':'Compare four arms on common fold1 population; forbid ranking hybrid totals as uniform three-fold candidates.'}
        ],'technical_issues':[
            {'kind':'warning','description':'Torch sparse CSR beta warning; forward/gradient tests and actual fits passed; not a fit failure.'},
            {'kind':'diagnostic_query','description':'One ad-hoc pandas groupby query removed truth from the grouped frame; rerun with explicit class keys. No training or prediction mutation.'},
            {'kind':'status_schema','description':'Read-only MCP status now accepts registered limitations when a new delivery has no legacy validation_scope field. Quality remains determined by actual delivery.'}
        ]}
    write(ROOT/'training/review_policy/v131_observed_cases.json',issues)
    summary='最新实际训练V131：8次拟合、31400次参数更新，训练侧第1折资格通过，来源外质量失败，无模型晋升。困难S527→14，O/C/CO非混标困难S零错；R三折ASA M/S错1956/1641，原A0为318/2074，新增1638条M错，完整任务2499→3704错。停止扩训，先审查新增M与同样缺支持的正确对照。'
    prefix='最新实际训练与停止决定（V131）：**'+summary+'** [完整训练结果、逐类风险与问题记录]({link})。\n\n'
    for relative in ['README.md','docs/TRAINING_PLAN.md']:
        path=ROOT/relative;text=path.read_text(encoding='utf-8')
        if '最新实际训练与停止决定（V131）' not in text:
            first,rest=text.split('\n',1)
            link=REPORT if relative=='README.md' else REPORT.removeprefix('docs/')
            path.write_text(first+'\n\n'+prefix.format(link=link)+rest.lstrip('\n'),encoding='utf-8')
        text=path.read_text(encoding='utf-8').replace('当前聚焦方向（V130，未训练）','历史执行前方向（V130；已由V131执行）')
        text=text.replace('最新实际训练与停止决定（V128）','历史实际训练与停止决定（V128）')
        path.write_text(text,encoding='utf-8')
    catpath=ROOT/'mcp_readonly/catalog.json';cat=read(catpath)
    for entry in cat['documents']:
        file=ROOT/entry['path']
        if entry['path']=='docs/TRAINING_PLAN.md':entry['sha256']=sha(file)
        else:
            if sha(file)!=entry['sha256']:raise ValueError('Unexpected historical source mutation '+entry['path'])
    project=cat['project'];project.update({'as_of':'2026-09-30','current_summary':summary,
        'authoritative_delivery_id':'v131-delivery','authoritative_direction_id':'v131-results',
        'current_direction':[
            '本轮完成V130全部合格触发步骤：2小拟合+4主对照+2来源检查，固定终点；R质量失败，停止追加拟合及模型替换。',
            '第1折学习缺口已显著修复，其他训练角色仍有28/41条非混标S错误；训练学习与来源外M/S保护分开验收。',
            '下一步先零拟合审查新增1638条M，尤其2868新增1120条；加入同样支持不足却判对的M对照，避免支持缺口的单因果解释。',
            '后续内部跨来源监督与训练资格同时审查；保持官方全任务逐类评分，不使用外层结果同轮调权或选早检查点。'],
        'known_limits':[
            '来源外ASA M错误318→1956，S2074→1641；三折总错都退化，质量失败，无模型晋升。',
            'O/C/CO仅第1折新拟合，训练零非混标S错不等于迁移解决；其全量表是混合诊断。',
            '新增M错中1566条缺完整事实同类支持，但62618条正确M也缺该支持，不能自动把支持缺口当唯一根因。',
            '规范gram/事实输入不是完整保序原文；有用原文继续保留，不能用当前表示冲突宣布现实安全语义不可解。',
            'M/S细则未知，开发折已反复查看；没有新独立盲测、官方提交或企业迁移验收。']})
    new=[
        ('v131-results','V131训练学习资格与来源外失败结果',REPORT,'current_review','8拟合31400步；四臂训练资格通过，M迁移退化，逐项停止与问题证据。'),
        ('v131-delivery','V131八拟合实际交付','artifacts/v131_learning_trial_20260930/delivery.json','delivery','训练学习通过，来源外质量失败，无模型晋升。'),
        ('v131-verification','V131实际模型和官方原行核验','artifacts/v131_learning_trial_20260930/verification.json','verification','8终点模型重放及2056871官方原行，质量失败独立列明。'),
        ('v131-error-audit','V131退化支持与匹配轨迹','artifacts/v131_learning_trial_20260930/result_audit/audit.json','audit','1638新增M，2868贡献1120，支持缺口正确对照，R25/50/75/100迁移轨迹。'),
        ('v131-learning-selection','V131训练内固定选择','artifacts/v131_learning_trial_20260930/learning_selection.json','audit','先记录TRAIN选R，后读取新留出，四臂资格完整复算。'),
        ('v131-observed-cases','V131实际反例执行约束','training/review_policy/v131_observed_cases.json','review_constraints','训练通过不能替代迁移，完美困难S仍可退化M，混合诊断边界。')]
    ids={e['id'] for e in cat['documents']}
    for ident,title,path,category,desc in reversed(new):
        if ident in ids:raise ValueError('Do not silently overwrite registered record '+ident)
        cat['documents'].insert(0,{'id':ident,'title':title,'path':path,'category':category,'summary':desc,
            'sha256':sha(ROOT/path),'keywords':['V131','训练结果','恶意','可疑','M/S','第一性原理','当前方向']})
    write(catpath,cat)
    print(json.dumps({'catalog_documents':len(cat['documents']),'latest_actual':'v131-delivery','direction':'v131-results'},ensure_ascii=False))


if __name__=='__main__':main()
