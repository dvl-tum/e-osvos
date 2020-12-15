from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from .meta_model import MetaModel


class MetaOptimizer(nn.Module):

    def __init__(self, model, init_lr, learn_model_init,
                 second_order_gradients, lr_hierarchy_level,
                 use_log_init_lr, max_lr):
        super(MetaOptimizer, self).__init__()

        self._optim = None
        self._train_loss = None
        self._device = 'cpu'
        self._learn_model_init = learn_model_init
        self._second_order_gradients = second_order_gradients
        self._use_log_init_lr = use_log_init_lr
        self._max_lr = max_lr
        self._lr_hierarchy_level = lr_hierarchy_level

        self.meta_model = MetaModel(model)

        if self._lr_hierarchy_level == 'SINGLE':
            log_init_lr = torch.ones(1, 1).mul(init_lr)

            if self._use_log_init_lr:
                log_init_lr = log_init_lr.log()

            self.log_init_lr = torch.nn.Parameter(log_init_lr)

        elif self._lr_hierarchy_level == 'TENSOR':
            log_init_lr = torch.ones(
                self.meta_model.num_param_groups, 1).mul(init_lr)
            log_init_lr += torch.rand_like(log_init_lr).sub(0.5) * init_lr

            if self._use_log_init_lr:
                log_init_lr = log_init_lr.log()

            self.log_init_lr = torch.nn.Parameter(log_init_lr)

        elif self._lr_hierarchy_level == 'PARAM' or self._lr_hierarchy_level == 'NEURON':
            self.log_init_lr = []
            for name, param in model.named_parameters():
                if param.requires_grad:

                    if self._lr_hierarchy_level == 'PARAM':
                        log_init_lr_param = torch.ones_like(param).mul(init_lr)
                    else:
                        lr_neuron_shape = (param.shape[0], ) + (1,) * (len(param.shape) - 1)
                        log_init_lr_param = torch.ones(
                            lr_neuron_shape).mul(init_lr)

                    log_init_lr_param += torch.rand_like(
                        log_init_lr_param).sub(0.5) * init_lr

                    if self._use_log_init_lr:
                        log_init_lr_param = log_init_lr_param.log()

                    log_init_lr_param = torch.nn.Parameter(log_init_lr_param)
                    self.register_parameter(f"log_init_lr_{name.replace('.', '-')}",
                                            log_init_lr_param)
                    self.log_init_lr.append(log_init_lr_param)
        else:
            raise NotImplementedError

        self._model_init = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._model_init[name] = param

        if self._learn_model_init:
            for name, param in self._model_init.items():
                self.register_parameter(f"model_init_{name.replace('.', '-')}", param)

        self.state = {}
        self._init_state()

    @property
    def init_lr(self):
        if isinstance(self.log_init_lr, list):
            if self._use_log_init_lr:
                return torch.Tensor([l.exp().mean() for l in self.log_init_lr])
            else:
                return torch.Tensor([l.mean() for l in self.log_init_lr])
        else:
            if self._use_log_init_lr:
                return self.log_init_lr.exp()
            else:
                return self.log_init_lr

    @property
    def state_lr(self):
        if isinstance(self.state["log_lr"], list):
            if self._use_log_init_lr:
                return torch.tensor([l.exp().mean() for l in self.state["log_lr"]])
            else:
                return torch.tensor([l.mean() for l in self.state["log_lr"]])
        else:
            if self._use_log_init_lr:
                return self.state["log_lr"].exp()
            else:
                return self.state["log_lr"]

    def init_zero_grad(self):
        output = 0.0
        for param in self.parameters():
            output += param.mean()
        output.backward()
        self.zero_grad()

    def clamp_init_lr(self):
        if self._use_log_init_lr:
            min_clamp = -33
        else:
            min_clamp = 0

        max_clamp = None
        if self._max_lr is not  None:
            if self._use_log_init_lr:
               max_clamp = torch.log(torch.tensor(self._max_lr))
            else:
                max_clamp = self._max_lr

        if isinstance(self.log_init_lr, list):
            self.log_init_lr = [l.data.clamp_(min_clamp, max_clamp)
                                for l in self.log_init_lr]
        else:
            self.log_init_lr.data.clamp_(min_clamp, max_clamp)

    def to(self, device):
        super(MetaOptimizer, self).to(device)
        self._device = device

        if isinstance(self.state['log_lr'], list):
            self.state['log_lr'] = [log_lr.to(device) for log_lr in self.state['log_lr']]
        else:
            self.state['log_lr'] = self.state['log_lr'].to(device)

    def reset(self, keep_state=False):
        if keep_state:
            if isinstance(self.log_init_lr, list):
                self.state['log_lr'] = [log_lr.detach() for log_lr in self.state['log_lr']]
            else:
                self.state['log_lr'] = self.state['log_lr'].detach()

            self.meta_model.detach_param_groups()
        else:
            self.meta_model.init_param_groups(self._model_init)

            self._init_state()

    def _init_state(self):
        if self._lr_hierarchy_level == 'SINGLE':
            self.state["log_lr"] = self.log_init_lr.repeat(self.meta_model.num_param_groups, 1)
        else:
            self.state["log_lr"] = self.log_init_lr

        self.state["num_steps"] = 0

    def set_train_loss(self, train_loss):
        # TODO: refactor
        if not self.training or not self._second_order_gradients:
            train_loss = train_loss.detach()

        # if self.state["num_steps"]:
        #     self._train_loss_diff = train_loss - self._prev_train_loss
        # else:
        #     self._train_loss_diff = torch.zeros_like(train_loss)
        # self._prev_train_loss = train_loss
        self._train_loss = train_loss

    def step(self, train_loss):
        state_lr = self.state["log_lr"]
        if self._use_log_init_lr:
            if isinstance(state_lr, list):
                state_lr = [lr.exp() for lr in state_lr]
            else:
                state_lr = state_lr.exp()

        create_graph = self.training and self._second_order_gradients
        if create_graph:
            param_group_grads = torch.autograd.grad(
                train_loss,
                [p for p in self.meta_model.model.parameters() if p.requires_grad],
                create_graph=True)

            param_group_grads = {n: param_group_grads[i]
                                 for i, n in enumerate([n for n, p in self.meta_model.model.named_parameters() if p.requires_grad])}

            param_groups_without_second_order_derivate_names = [n for n, _ in self.meta_model.model.named_parameters_without_second_order_derivate()]

            param_group_grads = [grad.detach() if n in param_groups_without_second_order_derivate_names
                                 else grad
                                 for n, grad in param_group_grads.items()]

        else:
            param_group_grads = torch.autograd.grad(train_loss,
                                                    [p for p in self.meta_model.model.parameters()
                                                     if p.requires_grad])

        param_group_step = [grad.to(lr.device) * lr
                            for lr, grad in zip(state_lr, param_group_grads)]

        if hasattr(self, 'only_box_head') and self.only_box_head:
            self.meta_model.apply_param_groups_step_box_head(param_group_step)
        else:
            self.meta_model.apply_param_groups_step(param_group_step)

        self.state["num_steps"] += 1

