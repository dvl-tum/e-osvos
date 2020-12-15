import torch
import torch.nn as nn


class MetaModel:
    """
        A helper class that keeps track of meta updates
        It's done by replacing parameters with variables and applying updates to
        them.
    """

    def __init__(self, model):
        self.model = model

    def init_zero_grad(self):
        output = 0.0
        for param in self.model.parameters():
             if param.requires_grad:
                output += param.mean()
        output.backward()
        self.model.zero_grad()

    def deparameterize(self):
        for _, module, n_p, p in self.param_groups():
            module._parameters[n_p] = p.data
            module._parameters[n_p].grad = p.grad

    def get_flat_params(self):
        params = []

        for p in self.model.parameters():
            if p.requires_grad:
                params.append(p.view(-1))

        return torch.cat(params).unsqueeze(-1).detach()

    def set_flat_params(self, flat_params, keep_grads=True):
        offset = 0

        for _, module, n_p, p in self.param_groups():
            p_flat_size = p.numel()
            flat_param = flat_params[offset:offset + p_flat_size].view(*p.size())
            if keep_grads:
                module._parameters[n_p] = flat_param
            else:
                module._parameters[n_p].data.copy_(flat_param.data)
            offset += p_flat_size

    def param_groups(self):
        # _parameters includes only direct parameters of a module and not all
        # parameters of its potential submodules.
        for n_m, module in self.model.named_modules():
            if len(module._parameters):
                for n_p, p in module._parameters.items():
                    if p is not None and p.requires_grad:
                        yield n_m, module, n_p, p

    @property
    def num_param_groups(self):
        return len(list(self.param_groups()))

    def detach_param_groups(self):
        for _, module, n_p, p in self.param_groups():
            module._parameters[n_p] = p.detach()
            module._parameters[n_p].requires_grad = True

    def init_param_groups(self, group_inits):
        for n_m, module, n_p, _ in self.param_groups():
            group_key = f"{n_m}.{n_p}"
            if group_key in group_inits:
                module._parameters[n_p] = group_inits[group_key]

    def apply_param_groups_step_box_head(self, param_groups_step):
        for (_, module, n_p, p), p_g_s in zip(self.param_groups(), param_groups_step):
            if True:
                module._parameters[n_p] = p - p_g_s

    def apply_param_groups_step(self, param_groups_step):
        for (_, module, n_p, p), p_g_s in zip(self.param_groups(), param_groups_step):
            module._parameters[n_p] = p - p_g_s

    def copy_params_from(self, model: nn.Module):
        for model_to, model_from in zip(self.model.parameters(), model.parameters()):
            model_to.data.copy_(model_from.data)

    def copy_params_to(self, model: nn.Module):
        for model_from, model_to in zip(self.model.parameters(), model.parameters()):
            model_to.data.copy_(model_from.data)

    def copy_grads_from(self, model: nn.Module):
        for model_to, model_from in zip(self.model.parameters(), model.parameters()):
            model_to.grad.data.copy_(model_from.grad.data)

    def copy_batch_norm_running_stats_from(self, model: nn.Module):
        bn_modules_from = [m for m in model.modules()
                           if isinstance(m, torch.nn.BatchNorm2d)]
        bn_modules_to = [m for m in self.model.modules()
                         if isinstance(m, torch.nn.BatchNorm2d)]

        for bn_module_to, bn_module_from in zip(bn_modules_to, bn_modules_from):
            bn_module_to.running_mean = bn_module_from.running_mean.clone().to(bn_module_to.running_mean.device)
            bn_module_to.running_var = bn_module_from.running_var.clone().to(bn_module_to.running_var.device)
            bn_module_to.num_batches_tracked = bn_module_from.num_batches_tracked.clone().to(bn_module_to.num_batches_tracked.device)

    def copy_batch_norm_batch_stats_from(self, model: nn.Module):
        bn_modules_from = [m for m in model.modules()
                           if isinstance(m, torch.nn.BatchNorm2d)]
        bn_modules_to = [m for m in self.model.modules()
                         if isinstance(m, torch.nn.BatchNorm2d)]

        for bn_module_to, bn_module_from in zip(bn_modules_to, bn_modules_from):
            bn_module_to.running_mean.data.copy_(bn_module_from.batch_mean.data)
            bn_module_to.running_var.data.copy_(bn_module_from.batch_var.data)
            bn_module_to.num_batches_tracked.fill_(1)
