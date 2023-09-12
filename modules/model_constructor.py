import importlib
from modules.compound_model import CompoundModel
from modules.experiment_helper import parse_experiment_settings, get_model_save_path


def _create_model_instance(model_category, model_name):
    model_class = importlib.import_module(f'model_library.{model_category}s.{model_name}').Model
    return model_class()


def create_model_by_experiment_settings(experiment_settings, load_from=''):
    if 'compound_model' in experiment_settings:
        compound_model_setting = experiment_settings['compound_model']
        sub_models = {}
        for model_category in ['generator', 'discriminator', 'regressor', 'profiler']:
            if model_category in compound_model_setting:
                sub_models[model_category] = _create_model_instance(
                    model_category,
                    compound_model_setting[model_category]
                )

        if not load_from and 'load_pretrain_weight' in compound_model_setting:
            pretrain_weight_setting = compound_model_setting['load_pretrain_weight']
            from_experiment = pretrain_weight_setting.get('from_experiment', experiment_settings['experiment_name'])
            from_sub_exp = pretrain_weight_setting['from_sub_exp']
            load_from = get_model_save_path({
                'experiment_name': from_experiment,
                'sub_exp_name': from_sub_exp
            })

        load_pretrain_except = compound_model_setting.get('load_pretrain_weight', {}).get('except', [])
        if load_from:
            for model_category, sub_model in sub_models.items():
                if model_category not in load_pretrain_except:
                    sub_model.load_weights(f'{load_from}/{model_category}')

        compound_model = CompoundModel(**sub_models)
        return 'compound_model', compound_model

    for single_model_category in ['regressor', 'profiler']:
        if single_model_category in experiment_settings:
            single_model = _create_model_instance(single_model_category, experiment_settings[single_model_category])
            if load_from:
                single_model.load_weights(f'{load_from}')
            return single_model_category, single_model


# This function is faciliating creating model instance in jupiter notebook
def create_model_by_experiment_path_and_stage(experiment_path, sub_exp_name):
    sub_exp_settings = parse_experiment_settings(experiment_path, only_this_sub_exp=sub_exp_name)
    model_save_path = get_model_save_path(sub_exp_settings)
    model_type, model = create_model_by_experiment_settings(sub_exp_settings, load_from=model_save_path)
    return model
