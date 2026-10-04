"""Publish concrete V159 qualification, preserving V158 actual delivery."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v159_candidate_qualification_publication_20261001'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists();OUT.mkdir()
    mutable=[ROOT/p for p in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    for p in mutable:
        dest=OUT/'previous_publication_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    specs=[
        ('v159-function-budget-review','V159唯一函数与完整梯度预算资格','docs/V159_CLASS_BOUNDARY_FUNCTION_AND_BUDGET_QUALIFICATION.md','current_review'),
        ('v159-independent-decision','V159独立历史方法与学习方向审查','docs/V159_INDEPENDENT_NEXT_TRAINING_DECISION_AND_RESEARCH.md','review_evidence'),
        ('v159-numerical-review','V159概率零值与实际梯度边界','docs/V159_NUMERICAL_INITIALIZATION_AND_GRADIENT_REVIEW.md','review_evidence'),
        ('v159-update-scale-review','V159真实千级margin与注册前精化','docs/V159_UPDATE_SCALE_REVIEW_AND_CANDIDATE_REFINEMENT.md','review_evidence'),
        ('v159-input-capacity','V159完整当前输入冲突和OOF容量','artifacts/v159_independent_OOF_capacity_and_input_review_v2_20261001/audit.json','review_evidence'),
        ('v159-current-capacity','V159当前16专家与旧N1贡献区别','artifacts/v159_current_expert_capacity_review_20261001/audit.json','review_evidence'),
        ('v159-mgda-math','V159两类精确共同下降1000合成检验','artifacts/v159_mgda_synthetic_qualification_20261001/qualification.json','review_evidence'),
        ('v159-probability-domain','V159全合法保存概率数值域','artifacts/v159_saved_probability_numerical_audit_20261001/audit.json','review_evidence'),
        ('v159-saved-logit-scale','V159保存logits全角色尺度与稳定风险','artifacts/v159_saved_logit_scale_and_risk_audit_20261001/audit.json','review_evidence'),
        ('v159-numeric-gradient','V159零概率断梯度合成反例','artifacts/v159_zero_probability_gradient_qualification_20261001/qualification.json','review_evidence'),
        ('v159-actual-head-gradient','V159实际候选头CPU-CUDA完整梯度合成','artifacts/v159_boundary_torch_synthetic_qualification_v3_20261001/qualification.json','current_evidence'),
        ('v159-candidate-contract','V159单候选及77832前向2412梯度未来约束','training/review_policy/v159_candidate_qualification_contract.json','current_plan'),
        ('v159-qualification','V159源码输入初始化N预算已绑定但未激活','artifacts/v159_candidate_qualification_v2_20261001/qualification.json','current_evidence'),
        ('v159-candidate-source','V159固定概率基座与完整输入实际源码','training/v159_current_input_boundary_v2.py','review_evidence'),
        ('v159-direction-source','V159同尺度逐类方向与有限守卫源码','training/v159_class_direction.py','review_evidence')]
    paths=[Path(__file__).resolve()]+[ROOT/s[2] for s in specs]
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    save(OUT/'pre_publication_bindings.json',dict(status='bound_before_V159_review_publication',source_sha256=bindings))
    q=read(ROOT/'artifacts/v159_candidate_qualification_v2_20261001/qualification.json')
    assert not q['formal_runtime_ready'] and q['actual_official_fits']==q['actual_official_classifier_calls']==q['actual_official_gradients']==0
    cat=read(mutable[2]);assert cat['project']['authoritative_delivery_id']=='v158-delivery' and cat['project']['authoritative_direction_id']=='v158-independent-final-review'
    for e in cat['documents']:assert sha(ROOT/e['path'])==e['sha256'],e['id']
    published=[]
    for ident,title,path,category in specs:
        source=ROOT/path;dest=OUT/'physical_source_snapshots'/path
        if dest.suffix=='.py':dest=dest.with_suffix('.py.txt')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);assert sha(source)==sha(dest)
        assert dest.stat().st_size<=256*1024 and not any(e['id']==ident for e in cat['documents'])
        cat['documents'].insert(0,dict(id=ident,title=title,path=dest.relative_to(ROOT).as_posix(),category=category,summary=title,sha256=sha(dest),keywords=['V159','资格','分类','完整输入','梯度','预算','未激活']))
        published.append(dict(id=ident,original=path,snapshot=dest.relative_to(ROOT).as_posix(),sha256=sha(dest)))
    project=cat['project'];project['authoritative_direction_id']='v159-function-budget-review'
    project['current_summary']='V158完成60次真实拟合及2056871原行验收，融合1200完整梯度/1200proposal/1200更新。ASA414M/1872S，完整质量失败、未晋升；OOF角色0/1 S牺牲仍在。V159唯一完整当前输入+16意见自由三类头已源码和CPU/CUDA合成梯度资格、固定1e-12基座、3e-12起点容差、N处理和未来预算绑定；未来最多6fit/2412完整类梯度/77832前向分块，当前官方新fit/梯度/更新/函数0。正式入口/全运行库封存/真实全行零步尚未就绪，完整目标active。'
    project['current_direction']=[
        '唯一候选 v159_current_input_boundary_v2.py：完整66287当前列、16当前合法专家意见、1060832参数、零输出，同A/B seed15901；旧N1/prior不输入。',
        '千级旧logit必要修正审查后在注册前精化概率基座；floor1e-12、起点容差3e-12/argmax0变化。稳定log-softmax训练风险和保存概率clip CE分报。',
        '实际候选CPU/CUDA合成完整梯度通过，仅证明实现。未来两臂同函数/数值/初始化，A原频CE，B精确共同下降+逐类风险/OOF保持组合，不能拆解因果。',
        '未来6fit最多2400训练完整类梯度+12重复预飞行、3600proposal、1200更新；336+77016+480=77832前向分块。旧54项不重跑。',
        '下一步只准备并审查可计数正式入口和全验收，封存实际导入/运行库/数据后才可真实全行零步与重复逐类梯度；当前尚无正式激活入口。',
        '不把头训练OOF说成独立头验证，现外折不是盲测；全部当前能力、原行频率、混类和完整三分类/来源控制均保留。三个目标未完成。']
    project['known_limits'].append('V159 toy梯度资格和未来预算不代表官方零步/拟合资格或SOC学习有效；新N监督缺失，完整质量仍以V158失败为当前事实。')
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024;mutable[2].write_bytes(raw)
    text=mutable[0].read_text(encoding='utf-8');needle='当前证据与下一资格：';assert needle in text
    top='当前单候选源码与预算资格：[V159函数及预算](docs/V159_CLASS_BOUNDARY_FUNCTION_AND_BUDGET_QUALIFICATION.md)。实际CPU/CUDA合成完整头梯度通过；未来6fit/2412完整类梯度/77832前向已绑定，但当前官方新拟合0，正式入口、完整封存及真实全行零步尚未完成。最新实际模型仍V158，质量失败。\n\n'
    mutable[0].write_text(text.replace(needle,top+needle,1),encoding='utf-8')
    mutable[1].write_text(mutable[1].read_text(encoding='utf-8')+'\n\n## V159唯一候选资格已具体化，未正式激活\n\n详见 `docs/V159_CLASS_BOUNDARY_FUNCTION_AND_BUDGET_QUALIFICATION.md`。选定源码 `v159_current_input_boundary_v2.py` 和 `v159_class_direction.py`；当前完整CSR+16NN意见，floor1e-12基座、稳定三类log-softmax风险，同A/B初始化。实际toy CPU/CUDA完整头反向已exit0。机器约束 `training/review_policy/v159_candidate_qualification_contract.json`，未来6fit、2412完整类梯度、77832前向，当前官方新函数/特征/梯度/拟合/更新0。正式计数入口、运行库及全部物理输入封存、真实全行零步仍待实现和审查；不把合成资格提升为正式runtime ready。父独立输入容量、16与17专家、MGDA、零概率、实际logits尺度及三个审查文均发布物理快照。只读MCP方向v159-function-budget-review，实际交付仍v158-delivery。旧源码/失败保留，目标active。\n',encoding='utf-8')
    tests=mutable[3].read_text(encoding='utf-8').replace('v158-independent-final-review','v159-function-budget-review')
    addition='''    def test_v159_qualified_toy_head_does_not_activate_official_training(self) -> None:
        catalog = readonly._load_catalog()
        entries = readonly._documents_by_id(catalog)
        qualification = json.loads(readonly._document_text(entries['v159-qualification']))
        contract = json.loads(readonly._document_text(entries['v159-candidate-contract']))
        head = json.loads(readonly._document_text(entries['v159-actual-head-gradient']))
        self.assertEqual(contract['total_future_caps']['classifier_forward_chunks'], 77832)
        self.assertEqual(contract['total_future_caps']['full_class_gradients'], 2412)
        self.assertEqual(contract['allowed_activation_entries'], [])
        self.assertFalse(qualification['formal_runtime_ready'])
        self.assertFalse(qualification['official_zero_step_replay'])
        self.assertEqual(qualification['actual_official_fits'], 0)
        self.assertEqual(head['official_gradient_calls'], 0)
        self.assertTrue(head['CUDA_toy']['toy_sparse_CUDA_origin_probability_tolerance_pass'])
        self.assertLess(head['full_parameter_class_gradient_finite_difference_max_error'], 1e-9)
        self.assertEqual(contract['probability_base']['floor'], 1e-12)
        self.assertEqual(contract['probability_base']['origin_probability_tolerance'], 3e-12)

'''
    needle='    def test_non_loopback_binding_requires_token(self) -> None:\n';assert needle in tests
    mutable[3].write_text(tests.replace(needle,addition+needle),encoding='utf-8')
    check_bindings(bindings)
    save(OUT/'publication.json',dict(status='V159_concrete_single_candidate_qualification_published_not_activated',catalog_bytes=len(raw),
        authoritative_delivery='v158-delivery',authoritative_direction='v159-function-budget-review',documents=published,
        official_new_fits=0,official_new_classifier_calls=0,official_new_features_calls=0,official_new_gradients=0,official_new_updates=0,
        formal_runtime_ready=False,official_zero_step_replay=False,quality_acceptance=False,goal_status='active',
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve()]+mutable}))
    print(dict(status='V159_candidate_qualification_published',catalog_bytes=len(raw),official_new_fits=0))

if __name__=='__main__':main()
