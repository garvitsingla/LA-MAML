import torch
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence
from maml_rl.metalearners.base import GradientBasedMetaLearner
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters)
from maml_rl.utils.optimization import conjugate_gradient
from maml_rl.utils.reinforcement_learning import reinforce_loss
from collections import OrderedDict


class ANILTRPO(GradientBasedMetaLearner):
        
    def __init__(self,
                 policy,
                 fast_lr=0.5,
                 first_order=False,
                 device='cpu',
                 head_layer_prefix='layer3'):
        """
        Args:
            policy: The policy network
            fast_lr: Learning rate for inner loop adaptation
            first_order: Whether to use first-order approximation
            device: Device to run on
            head_layer_prefix: Prefix identifying the head layer(s) to update
        """
        super(ANILTRPO, self).__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.first_order = first_order
        self.head_layer_prefix = head_layer_prefix
        
        # Identify head parameters
        self._identify_head_params()
    
    def _identify_head_params(self):
        """Identify which parameters belong to the head (to be updated in inner loop)"""
        self.head_param_names = []
        self.body_param_names = []
        
        for name, _ in self.policy.named_parameters():
            if self.head_layer_prefix in name:
                self.head_param_names.append(name)
            else:
                self.body_param_names.append(name)
        
        print(f"[ANIL] Head parameters (updated in inner loop): {self.head_param_names}")
        print(f"[ANIL] Body parameters (frozen in inner loop): {self.body_param_names}")
    
    def adapt(self, train_futures, first_order=None):
        """
        ANIL Inner Loop Adaptation
        
        Key difference from MAML: Only updates HEAD parameters, keeps BODY frozen.
        """
        if first_order is None:
            first_order = self.first_order
        
        params = None
        
        for futures in train_futures:
            inner_loss = reinforce_loss(self.policy,
                                        futures,
                                        params=params)
            
            # ANIL: Only update head parameters
            params = self._update_head_only(inner_loss,
                                           params=params,
                                           step_size=self.fast_lr,
                                           first_order=first_order)
        return params
    
    def _update_head_only(self, loss, params=None, step_size=0.5, first_order=False):
        """
        Update only the head parameters, keeping body frozen.
        
        This is the core ANIL modification.
        """
        if params is None:
            params = OrderedDict(self.policy.named_parameters())
        
        # Get gradients for HEAD parameters only
        head_params = OrderedDict(
            (name, param) for name, param in params.items() 
            if self.head_layer_prefix in name
        )
        
        # Compute gradients only for head
        grads = torch.autograd.grad(loss, head_params.values(),
                                   create_graph=not first_order)
        
        # Build updated params dict
        updated_params = OrderedDict()
        grad_iter = iter(grads)
        
        for name, param in params.items():
            if self.head_layer_prefix in name:
                # Update head parameters
                grad = next(grad_iter)
                updated_params[name] = param - step_size * grad
            else:
                # Keep body parameters frozen (just copy them)
                updated_params[name] = param
        
        return updated_params

    def hessian_vector_product(self, kl, damping=1e-2):
        grads = torch.autograd.grad(kl,
                                    self.policy.parameters(),
                                    create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                         self.policy.parameters(),
                                         retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, train_futures, valid_futures, old_pi=None):
        first_order = (old_pi is not None) or self.first_order

        # ANIL adaptation (head only)
        params = self.adapt(train_futures, first_order=first_order)

        with torch.set_grad_enabled(old_pi is None):
            valid_episodes = valid_futures
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            log_ratio = (pi.log_prob(valid_episodes.actions) - old_pi.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)

            losses = -weighted_mean(ratio * valid_episodes.advantages,
                                    lengths=valid_episodes.lengths)
            kls = weighted_mean(kl_divergence(pi, old_pi),
                                lengths=valid_episodes.lengths)

        return losses.mean(), kls.mean(), old_pi

    def step(self,
             train_futures,
             valid_futures,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        """
        Outer loop step - updates ALL parameters (both body and head)
        
        This is identical to MAML's outer loop.
        """
        num_tasks = len(train_futures[0])
        logs = {}

        # Compute the surrogate loss
        old_losses, old_kls, old_pis = self._async_gather([
            self.surrogate_loss(train, valid, old_pi=None)
            for (train, valid) in zip(zip(*train_futures), valid_futures)])

        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)

        old_loss = sum(old_losses) / num_tasks
        grads = torch.autograd.grad(old_loss,
                                    self.policy.parameters(),
                                    retain_graph=True)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir,
                              hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())

            losses, kls, _ = self._async_gather([
                self.surrogate_loss(train, valid, old_pi=old_pi)
                for (train, valid, old_pi)
                in zip(zip(*train_futures), valid_futures, old_pis)])

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())

        return logs
