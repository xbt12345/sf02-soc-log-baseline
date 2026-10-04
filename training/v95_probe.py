"""Training-only sign, role, sparse gradient and clone isolation checks."""
import copy
import json
import numpy as np
import torch
from run_v75 import save, sha
from threadpoolctl import threadpool_limits
from v92_train import Branch, csr
from v95_prepare import DEST
from v95_four_arm import make_context


def main():
    assert not (DEST / 'gradient_probe.json').exists()
    ctx = make_context()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(ctx['reg']['seed'])
    m = Branch(ctx['x'].shape[1]).to(device)
    x = csr(ctx['x'][ctx['used']], device)
    t = torch.as_tensor(np.asarray(ctx['zz'][ctx['train_ids']]), device=device, dtype=torch.float64)
    c = torch.as_tensor(ctx['cc'], device=device, dtype=torch.float64)
    a = torch.as_tensor(ctx['aux_weights']['A'], device=device, dtype=torch.float64)
    b = torch.as_tensor(ctx['aux_weights']['B'], device=device, dtype=torch.float64)
    def logits(net): return t + net(x).double()
    def ce(z, weights): return (-torch.log_softmax(z, 1) * weights).sum()
    def main_loss(net): return ce(logits(net), c) / ctx['full_n']
    loss = main_loss(m)
    loss.backward()
    grads = {n: float(p.grad.norm().item()) for n, p in m.named_parameters()}
    assert all(np.isfinite(v) for v in grads.values()) and grads['out.bias'] > 0
    with torch.no_grad():
        initial = [p.detach().clone() for p in m.parameters()]
        first = m.out.bias
        j = int(first.grad.abs().argmax().item())
        step = 1e-5
        before = float(loss.item())
        first[j] -= step * first.grad[j]
        after = float(main_loss(m).item())
        first[j] += step * first.grad[j]
    assert after < before and all(torch.equal(p, old) for p, old in zip(m.parameters(), initial))
    clone = copy.deepcopy(m)
    clone.zero_grad(set_to_none=True)
    inner = ce(logits(clone), a); inner.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in clone.parameters())
    with torch.no_grad():
        for p in clone.parameters(): p.add_(p.grad, alpha=-.01)
    clone.zero_grad(set_to_none=True)
    feedback = ce(logits(clone), b); feedback.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in clone.parameters())
    assert all(torch.equal(p, old) for p, old in zip(m.parameters(), initial))
    result = {'status': 'passed', 'classifier_fits': 0,
              'A_B_main_supervised_rows': ctx['full_n'],
              'main_loss_before': before, 'main_loss_after_test_step': after,
              'parameter_grad_norms': grads, 'inner_A_loss': float(inner.item()),
              'feedback_B_loss_on_A_updated_clone': float(feedback.item()),
              'teacher_or_original_parameters_changed': False,
              'V_gradient_rows': 0, 'auxiliary_episodes': ctx['episodes'],
              'source_sha256': sha(__file__)}
    save(DEST / 'gradient_probe.json', result)
    print(json.dumps({'status': 'passed', 'main_grad_norm': grads['out.bias'],
                      'main_loss_decreased': after < before,
                      'clone_feedback_gradient_finite': True,
                      'auxiliary_episodes': len(ctx['episodes'])}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4): main()
