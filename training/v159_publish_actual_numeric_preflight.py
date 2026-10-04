"""Publish completed actual preflight/cumulative costs; no model/grad calls."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v159_actual_numeric_preflight_publication_20261002'
def main():
    assert not OUT.exists();OUT.mkdir()
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for p in mutable:
        d=OUT/'previous_publication_snapshot'/p.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
    specs=[('v159-numeric-preflight-complete','V159当前三角色真实数值预检通过460头20梯度','docs/V159_NUMERIC_PREFLIGHT_ACTUAL_COMPLETE.md'),('v159-numeric-preflight-replay-review','V159真实预检保存向量和调用累计成本重算','artifacts/v159_numeric_preflight_execution_review_20261002/review.json')]
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+[ROOT/s[2] for s in specs]};(OUT/'pre_publication_bindings.json').write_text(json.dumps(dict(source_sha256=bound),indent=2)+'\n',encoding='utf-8')
    review=read(ROOT/specs[1][2]);assert review['cumulative_actual']==dict(head_calls=460,opinion_feature_calls=460,full_class_gradients=20,fits=0,updates=0)
    cat=read(mutable[2]);assert cat['project']['authoritative_delivery_id']=='v158-delivery'
    for e in cat['documents']:assert sha(ROOT/e['path'])==e['sha256'],e['id']
    for ident,title,name in specs:
        src=ROOT/name;dest=OUT/'physical_source_snapshots'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);assert sha(src)==sha(dest) and dest.stat().st_size<=256*1024
        assert all(e['id']!=ident for e in cat['documents']);cat['documents'].insert(0,dict(id=ident,title=title,path=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),category='current_review',keywords=['V159','真实预检','460','20','0fit']))
    p=cat['project'];p['authoritative_direction_id']='v159-numeric-preflight-complete'
    p['current_summary']='V158完成60次真实拟合及2056871完整原行验收，融合1200完整梯度/1200proposal/1200更新，质量失败、未晋升。V159新v4三角色预检真实exit0并保存全部重复向量，336头/意见块和12完整类梯度、参数未变、联合旧能力通过。加旧失败及诊断累计460头/意见、20完整类梯度、0fit/更新。保存结果重算exit0，旁路绑定脚本失败未写回执并如实保留。新累计封顶78196/2420；6fit与完整质量验收尚未执行，全目标active。'
    p['current_direction']=['当前唯一分类函数仍v3，真实训练/评价v4，数值政策固定8/16eps及方向无法分辨停止；argmax/旧能力严格。','三角色实际预检336头/12类梯度已经通过，完整梯度/q/logq/risk先保存后比较，联合部署旧能力通过。','累计旧84/4失败+40/4诊断+新336/12=460头意见/20类梯度；0fit/更新，当前无模型进程。','自己的事后保存审查真实exit0；旁路绑定失败明确保留，无事前绑定成功回执，独立审查文件本身先前已经核验。','接下来固定6fit，剩余77736头意见/2400训练梯度、每fit600proposal及200迭代不变，然后实际完整末五OOF和部署及2056871原行验收。','最新实际训练V158失败；数值预检不关闭学习、支持、迁移及完整目标。']
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(raw)<=256*1024;mutable[2].write_bytes(raw)
    lead='当前真实进度：[V159三角色数值预检通过](docs/V159_NUMERIC_PREFLIGHT_ACTUAL_COMPLETE.md)。累计460头/意见、20完整类梯度、0fit/更新；完整保存重算exit0、联合旧能力保护通过，固定6fit及完整验收待执行。\n\n'
    mutable[0].write_text(lead+mutable[0].read_text(encoding='utf-8'),encoding='utf-8');mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V159真实数值预检已完成\n\n'+lead+'旁路绑定失败未补造成功回执，详见当前说明与保存审查。方向v159-numeric-preflight-complete；最新实际训练仍V158。\n',encoding='utf-8')
    tests=mutable[3].read_text(encoding='utf-8').replace("'v159-numeric-v4-current-review'","'v159-numeric-preflight-complete'").replace('"v159-numeric-v4-current-review"','"v159-numeric-preflight-complete"')
    extra='''    def test_v159_real_three_role_preflight_does_not_close_training(self) -> None:
        entries=readonly._documents_by_id(readonly._load_catalog())
        proof=json.loads(readonly._document_text(entries['v159-numeric-preflight-replay-review']))
        self.assertEqual(proof['cumulative_actual'],dict(head_calls=460,opinion_feature_calls=460,full_class_gradients=20,fits=0,updates=0))
        self.assertTrue(proof['joint_TRAIN_retention']['passed'])
        self.assertFalse(proof['first_training_issue_passed'])
        self.assertFalse(proof['quality_acceptance'])
        self.assertTrue(proof['side_binding_failure']['not_relabelled_as_prospective_binding_success'])

'''
    needle='    def test_non_loopback_binding_requires_token(self) -> None:\n';assert needle in tests;mutable[3].write_text(tests.replace(needle,extra+needle),encoding='utf-8');check_bindings(bound)
    (OUT/'publication.json').write_text(json.dumps(dict(status='actual_numeric_preflight_complete_published_no_training_yet',catalog_bytes=len(raw),cumulative_actual=review['cumulative_actual'],new_model_calls=0,new_gradients=0,quality_acceptance=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+mutable}),indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status='actual_preflight_published',catalog_bytes=len(raw))))
if __name__=='__main__':main()
