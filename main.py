import argparse
import os
import tensorflow as tf

from modules.single_model_trainer import train_single_model
from modules.compound_model_trainer import train_compound_model
from modules.training_helper import get_tensorflow_datasets
from modules.experiment_helper import parse_experiment_settings, get_model_save_path,\
    get_log_path, print_sub_exp_settings
from modules.model_constructor import create_model_by_experiment_settings


def execute_sub_exp(sub_exp_settings, omit_completed_sub_exp):
    log_path = get_log_path(sub_exp_settings)

    print(f'Now running: {log_path}')
    if omit_completed_sub_exp and os.path.isdir(log_path):
        print('Sub-experiment already done before, skipped ಠ_ಠ')
        return
    print_sub_exp_settings(sub_exp_settings)
    
    summary_writer = tf.summary.create_file_writer(log_path)
    datasets = get_tensorflow_datasets(**sub_exp_settings['data'])
    model_save_path = get_model_save_path(sub_exp_settings)

    model_type, model = create_model_by_experiment_settings(sub_exp_settings)

    if model_type == 'compound_model':
        training_settings = sub_exp_settings['train_compound_model']
        trainer_function = train_compound_model
    elif model_type in ['regressor', 'profiler']:
        training_settings = sub_exp_settings[f'train_{model_type}']
        trainer_function = train_single_model

    trainer_function(
        model,
        datasets,
        summary_writer,
        model_save_path,
        **training_settings
    )


def main(experiment_path, GPU_limit, omit_completed_sub_exp):
    # shut up tensorflow!
    tf.get_logger().setLevel('ERROR')

    # restrict the memory usage
    if GPU_limit != -1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_limit)]
            )

    # parse yaml to get experiment settings
    experiment_list = parse_experiment_settings(experiment_path)

    for sub_exp_settings in experiment_list:
        execute_sub_exp(sub_exp_settings, omit_completed_sub_exp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path', help='name of the experiment setting, should match one of them file name in experiments folder')
    parser.add_argument('--GPU_limit', type=int, default=-1)
    parser.add_argument('-o', '--omit_completed_sub_exp', action='store_true')
    parser.add_argument('-d', '--CUDA_VISIBLE_DEVICES', type=str, default='')
    args = parser.parse_args()
    if args.CUDA_VISIBLE_DEVICES:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    main(args.experiment_path, args.GPU_limit, args.omit_completed_sub_exp)
