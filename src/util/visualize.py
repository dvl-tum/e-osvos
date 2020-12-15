import copy
import sacred

from .vis_utils import LineVis, TextVis
from .helper_func import data_loaders, init_parent_model


def dict_to_html(dict_obj):
    html_str = '<ul style="padding-left: 25px; list-style: none;">\n'
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            html_str += f'<li>{key}:</li>'
            html_str += dict_to_html(value)
        else:
            html_str += f'<li>{key}: {value}</li>'
    html_str += '</ul>\n'
    return html_str


def init_vis(run_name: str, env_suffix: str, _config: dict, _run: sacred.run.Run,
             torch_cfg: dict, datasets: dict, resume_meta_run_epoch_mode: str,
             no_vis: bool):
    run_name = f"{_run.experiment_info['name']}_{run_name}"
    vis_dict = {}
    if no_vis:
        return vis_dict

    resume  = False if resume_meta_run_epoch_mode is None else True

    opts = dict(title=f"CONFIG and NON META BASELINE (RUN: {_run._id})",
                width=1000, height=2000)
    vis_dict['config_vis'] = TextVis(opts, env=run_name, **torch_cfg['vis'])
    if not resume:
        vis_dict['config_vis'].plot(dict_to_html(_config))

    legend = [
        'MEAN seq TRAIN loss',
        'MEAN seq META loss',
        'STD seq META loss',
        'MAX seq META loss',
        'MIN seq META loss',
        'RUN TIME per ITER [min]']
    opts = dict(
        title=f"TRAIN METRICS (RUN: {_run._id})",
        xlabel='META ITERS',
        ylabel='METRICS',
        width=1000,
        height=450,
        legend=legend)
    vis_dict['meta_metrics_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    data_cfg = copy.deepcopy(_config['data_cfg'])

    for dataset_name, dataset in datasets.items():
        if dataset['eval'] and dataset['split'] is not None:
            loader, *_ = data_loaders(dataset, **
                                        data_cfg)  # pylint: disable=E1120
            legend = ['TIME PER FRAME', 'MEAN_TRAIN_LOSS']
            if _config['parent_model']['architecture'] == 'MaskRCNN':
                legend.extend(['MEAN_TRAIN_LOSS_cls_score',
                                'MEAN_TRAIN_LOSS_bbox_pred',
                                'MEAN_TRAIN_LOSS_mask_fcn_logits'])
            legend.extend(['J & F MEAN', 'J MEAN', 'J RECALL MEAN', 'J DECAY MEAN', 'F MEAN', 'F RECALL MEAN', 'F DECAY MEAN', 'INIT J MEAN'])
            # for seq_name in loader.dataset.seqs_names:
            #     loader.dataset.set_seq(seq_name)
            #     if loader.dataset.num_objects == 1:
            #         legend.append(f"INIT J_{seq_name}_1")

            for seq_name in loader.dataset.seqs_names:
                loader.dataset.set_seq(seq_name)
                if loader.dataset.num_object_groups == 1:
                    legend.extend([f"INIT J_{seq_name}_{i + 1}" for i in range(loader.dataset.num_objects)])

            for seq_name in loader.dataset.seqs_names:
                loader.dataset.set_seq(seq_name)
                legend.extend([f"J_{seq_name}_{i + 1}" for i in range(loader.dataset.num_objects)])

            opts = dict(
                title=f"EVAL: {dataset_name.upper()} - (RUN: {_run._id})",
                xlabel='META ITERS',
                ylabel=f'METRICS',
                width=1000,
                height=450,
                legend=legend)
            vis_dict[f'{dataset_name}_eval_seq_vis'] = LineVis(
                opts, env=run_name, resume=resume, **torch_cfg['vis'])

    legend = ['MEAN']
    if _config['parent_model']['architecture'] == 'MaskRCNN':
        legend.extend(['MEAN cls_score', 'MEAN bbox_pred', 'MEAN mask_fcn_logits'])
    # legend.extend([f"{seq_name}" for seq_name in train_loader.dataset.seqs_names])
    opts = dict(
        title=f"FINAL META LOSS (RUN: {_run._id})",
        xlabel='META EPOCHS',
        ylabel=f'META LOSS',
        width=1000,
        height=450,
        legend=legend)
    vis_dict[f'meta_loss_seq_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    legend = ['MEAN']
    if _config['parent_model']['architecture'] == 'MaskRCNN':
        legend.extend(
            ['MEAN cls_score', 'MEAN bbox_pred', 'MEAN mask_fcn_logits'])
    # legend.extend(
    #     [f"{seq_name}" for seq_name in train_loader.dataset.seqs_names])
    opts = dict(
        title=f"INIT TRAIN LOSS (RUN: {_run._id})",
        xlabel='META EPOCHS',
        ylabel=f'TRAIN LOSS',
        width=1000,
        height=450,
        legend=legend)
    vis_dict[f'train_loss_seq_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    model, _ = init_parent_model(**_config['parent_model'])
    legend = ['MEAN', 'STD'] + [f"{n}"
              for n, p in model.named_parameters()
              if p.requires_grad]
    opts = dict(
        title=f"FIRST EPOCH INIT LR (RUN: {_run._id})",
        xlabel='META ITERS',
        ylabel='LR',
        width=1000,
        height=450,
        legend=legend)
    vis_dict['init_lr_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    opts = dict(
        title=f"LRS HIST (RUN: {_run._id})",
        xlabel='FINE-TUNE EPOCHS',
        ylabel='LR',
        width=1000,
        height=450,
        legend=legend)
    vis_dict['lrs_hist_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    return vis_dict
