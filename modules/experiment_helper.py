import os
import yaml
import copy
import collections


def get_log_path(sub_exp_settings):
    experiment_name = sub_exp_settings['experiment_name']
    sub_exp_name = sub_exp_settings['sub_exp_name']
    log_path = f'../logs/{experiment_name}/{sub_exp_name}'
    return log_path


def get_model_save_path(sub_exp_settings):
    experiment_name = sub_exp_settings['experiment_name']
    sub_exp_name = sub_exp_settings['sub_exp_name']

    if not os.path.isdir('../saved_models'):
        os.mkdir('../saved_models')

    saving_folder = '../saved_models/' + experiment_name
    if not os.path.isdir(saving_folder):
        os.mkdir(saving_folder)

    model_save_path = saving_folder + '/' + sub_exp_name
    return model_save_path


def print_sub_exp_settings(sub_exp_settings, indent_level=0):
    for key, value in sub_exp_settings.items():
        indent = '\t' * indent_level
        if type(value) is dict:
            print(f'{indent}{key}:')
            print_sub_exp_settings(sub_exp_settings[key], indent_level+1)
        else:
            print(f'{indent}{key}: {value}')


def parse_experiment_settings(experiment_path, only_this_sub_exp=''):
    with open(experiment_path, 'r') as file:
        experiment_settings = yaml.full_load(file)

    template_exp_settings = experiment_settings['template']
    template_exp_settings['experiment_name'] = experiment_settings['experiment_name']

    def deep_update(source, overrides):
        for key, value in overrides.items():
            if isinstance(value, collections.Mapping) and value:
                returned = deep_update(source.get(key, {}), value)
                source[key] = returned
            else:
                source[key] = overrides[key]
        return source

    exp_list = [template_exp_settings]

    for sub_exp_overrides in experiment_settings.get('sub_experiments', []):
        sub_exp_settings = copy.deepcopy(template_exp_settings)
        sub_exp_settings = deep_update(sub_exp_settings, sub_exp_overrides)
        exp_list.append(sub_exp_settings)

    if only_this_sub_exp:
        for sub_exp in exp_list:
            if sub_exp['sub_exp_name'] == only_this_sub_exp:
                return sub_exp
        print('sub_exp not found!')
        return {}

    return exp_list
