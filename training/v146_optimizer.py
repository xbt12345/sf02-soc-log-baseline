"""Normalized true-gradient Armijo, with actual classification guard and rollback."""
import torch


def acceptable(base_objective,trial_objective,step,gradient_norm,classification_guard,armijo):
    return bool(classification_guard and trial_objective<base_objective and trial_objective<=base_objective-armijo*step*gradient_norm)


@torch.no_grad()
def assign(parameters,base,direction,step):
    for p,a,d in zip(parameters,base,direction):p.copy_(a-step*d)


@torch.no_grad()
def restore(parameters,base):
    for p,a in zip(parameters,base):p.copy_(a)
