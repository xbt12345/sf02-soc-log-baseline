"""Publish actual sealed failure and measured diagnostic; retain all history."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v159_real_preflight_failure_publication_20261002'

def main():
    assert not OUT.exists();OUT.mkdir()
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for p in mutable:
        target=OUT/'previous_publication_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    specs=[('v159-real-preflight-failure','V159真实预检失败及124头8梯度当前状态','docs/V159_REAL_PREFLIGHT_FAILURE_AND_REPEAT_DIAGNOSTIC.md'),
           ('v159-real-preflight-cost','V159原失败84头4梯度真实调用成本','artifacts/v159_preflight_failure_cost_receipt_20261002/receipt.json'),
           ('v159-real-repeat-vector-review','V159真实四保存梯度向量重算累计124头8梯度','artifacts/v159_gradient_diagnostic_saved_vector_review_v2_20261002/review.json'),
           ('v159-independent-repeat-vector-audit','V159独立真实重复梯度ULP及方向复核','artifacts/v159_independent_gradient_repeat_diagnostic_audit_20261002/audit.json')]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+[ROOT/s[2] for s in specs]}
    (OUT/'pre_publication_bindings.json').write_text(json.dumps({'source_sha256':bound},indent=2)+'\n',encoding='utf-8')
    proof=read(ROOT/specs[2][2]);assert proof['cumulative_actual']==dict(head_calls=124,opinion_feature_calls=124,full_class_gradients=8,fits=0,updates=0)
    cat=read(mutable[2]);assert cat['project']['authoritative_delivery_id']=='v158-delivery'
    for entry in cat['documents']:assert sha(ROOT/entry['path'])==entry['sha256'],entry['id']
    published=[]
    for ident,title,name in specs:
        src=ROOT/name;dest=OUT/'physical_source_snapshots'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        assert sha(src)==sha(dest) and dest.stat().st_size<=256*1024 and all(e['id']!=ident for e in cat['documents'])
        cat['documents'].insert(0,dict(id=ident,title=title,path=dest.relative_to(ROOT).as_posix(),category='current_review' if ident==specs[0][0] else 'review_evidence',sha256=sha(dest),keywords=['V159','真实失败','重复梯度','124','8','0fit']))
        published.append(dict(id=ident,path=dest.relative_to(ROOT).as_posix(),sha256=sha(dest)))
    p=cat['project'];p['authoritative_direction_id']='v159-real-preflight-failure'
    p['current_summary']='最新实际训练V158完成60 fits及完整2056871原行验收但质量失败。V159真实注册封存后fold0 A预检在梯度逐位重复断言失败84头/4完整类梯度；另行封存有限诊断40头/4类梯度，四完整向量保存并独立重算。累计124头及意见块、8完整类梯度、0fit/0update。M/S差仅输出最后48参数，A方向相同、B最大2ULP，具体算子原因未认证。等待固定尺度数值政策和真实尺寸非零toy资格；当前无模型进程，完整目标active。'
    p['current_direction']=[
        '真实旧v3 seal/preflight失败保持不可变；不是尚未调用，不自动重启旧入口。',
        '原失败84/4与新封存技术诊断40/4成本累计124/8；官方fit和更新仍0，最新实际完整训练交付仍V158。',
        '四保存向量独立重算：M最大4ULP、S1ULP；A方向逐位相同、B2ULP，lambda和下降符号稳定。仅认证fold0零输出重复结果，原因未证实。',
        '下一版本固定尺度重复检查须先实测真实输入尺寸非零合成状态及拒绝反例；argmax和能力保护严格，有限下降不能用容差代替。',
        '如完整重新预检，另行预登记有限技术增量到累计78196头/意见与2420完整类梯度；6 fits及2400训练梯度/600 proposals每fit不变，旧成本不得重置。',
        '训练学习、细行为支持、跨来源迁移、完整三分类及泛化尚未通过，goal active。']
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(raw)<=256*1024;mutable[2].write_bytes(raw)
    lead='当前真实状态：[V159预检失败与有限重复诊断](docs/V159_REAL_PREFLIGHT_FAILURE_AND_REPEAT_DIAGNOSTIC.md)。累计124次头前向/意见块、8次完整类梯度、0fit/0更新；旧预检已失败退出，无模型进程。数值政策及非零合成资格通过前不启动正式拟合。最新实际训练仍V158、质量失败；三个目标active。\n\n'
    mutable[0].write_text(lead+mutable[0].read_text(encoding='utf-8'),encoding='utf-8')
    mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V159真实失败与累计技术成本\n\n'+lead+'先前0新调用和未seal记录是历史状态。实际新seal已执行，旧preflight失败；成本124/8保存并重算，不能重置。当前方向v159-real-preflight-failure，正式交付v158-delivery。\n',encoding='utf-8')
    tests=mutable[3].read_text(encoding='utf-8').replace("'v159-full-oof-window-review'","'v159-real-preflight-failure'").replace('"v159-full-oof-window-review"','"v159-real-preflight-failure"')
    extra='''    def test_v159_actual_failed_preflight_costs_are_not_reset(self) -> None:
        catalog=readonly._load_catalog();entries=readonly._documents_by_id(catalog)
        self.assertEqual(catalog['project']['authoritative_direction_id'],'v159-real-preflight-failure')
        proof=json.loads(readonly._document_text(entries['v159-real-repeat-vector-review']))
        self.assertEqual(proof['cumulative_actual'],dict(head_calls=124,opinion_feature_calls=124,full_class_gradients=8,fits=0,updates=0))
        self.assertFalse(proof['new_repeat_policy_active'])
        self.assertFalse(proof['first_training_issue_passed'])
        self.assertFalse(proof['quality_acceptance'])
        self.assertEqual(proof['latest_actual_training'],'V158')

'''
    needle='    def test_non_loopback_binding_requires_token(self) -> None:\n';assert needle in tests;mutable[3].write_text(tests.replace(needle,extra+needle),encoding='utf-8')
    check_bindings(bound)
    (OUT/'publication.json').write_text(json.dumps(dict(status='actual_failure_and_finite_diagnostic_published',catalog_bytes=len(raw),documents=published,cumulative_actual=proof['cumulative_actual'],latest_actual_training='V158',official_live_process=False,quality_acceptance=False,goal_status='active',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+mutable}),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='actual_failed_preflight_and_diagnostic_published',catalog_bytes=len(raw),actual=proof['cumulative_actual'])))
if __name__=='__main__':main()
