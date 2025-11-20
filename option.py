# -*- encoding: utf-8 -*-
'''
@Filename    :option.py
@Time        :2020/07/10 23:23:10
@Author      :Kai Li
@Version     :1.0
'''

import os
import yaml


def _apply_placeholders(value, placeholders):
    if isinstance(value, str):
        try:
            return value.format(**placeholders)
        except KeyError:
            return value
    if isinstance(value, dict):
        return {k: _apply_placeholders(v, placeholders) for k, v in value.items()}
    if isinstance(value, list):
        return [_apply_placeholders(v, placeholders) for v in value]
    return value


def load_config(opt_path):
    with open(opt_path, mode='r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    placeholders = cfg.get('paths', {}) or {}
    placeholders = {k: str(v) for k, v in placeholders.items()}
    if placeholders:
        cfg = _apply_placeholders(cfg, placeholders)
    return cfg


def parse(opt_path, is_train=True):
    '''
       opt_path: the path of yml file
       is_train: True
    '''
    opt = load_config(opt_path)
    # Export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])

    # is_train into option
    opt['is_train'] = is_train

    return opt


if __name__ == "__main__":
    parse('./train/train.yml')
