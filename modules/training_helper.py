import numpy as np
import tensorflow as tf
from collections import defaultdict
from modules.feature_generator import load_dataset
from modules.image_processor import evenly_rotate, random_rotate, is_polar_coordinate, crop_center


def image_augmentation(images, crop_width=64):
    rotated_images = random_rotate(images)
    if is_polar_coordinate(images):
        return rotated_images

    # cart_coordinate image need center cropping
    cropped_images = crop_center(rotated_images, crop_width)
    return cropped_images


def calculate_Vmax_R34_RMW_from_profiles(profiles):
    calculated_Vmax = tf.math.reduce_max(profiles, axis=1, keepdims=True)

    calculated_R34 = tf.math.reduce_max(
        tf.cast(profiles >= 34.0, tf.float32) * tf.range(0, 751, 5, dtype=tf.float32),
        axis=1, keepdims=True
    )

    calculated_RMW = tf.cast(tf.math.argmax(profiles, axis=1), tf.float32) * 5
    calculated_RMW = tf.expand_dims(calculated_RMW, axis=1)

    return calculated_Vmax, calculated_R34, calculated_RMW


def is_profile_label(label):
    if len(label.shape) == 2 and label.shape[1] == 151:
        return True
    return False


def transfer_label_to_dict(label):
    if is_profile_label(label):
        calculated_Vmax, calculated_R34, calculated_RMW = calculate_Vmax_R34_RMW_from_profiles(label)
        pred_dict = {
            'profile': label,
            'Vmax': calculated_Vmax,
            'R34': calculated_R34,
            'RMW': calculated_RMW
        }
    else:
        pred_dict = {
            'Vmax': label
        }
    return pred_dict


def calculate_loss_dict(pred_label, loss_function, label, Vmax_loss_sample_weight_exponent):
    pred_dict = transfer_label_to_dict(pred_label)
    label_dict = transfer_label_to_dict(label)
    loss_dict = {}
    if 'Vmax' in pred_dict:
        big_sample_weight = label_dict['Vmax'] ** Vmax_loss_sample_weight_exponent
        normalized_sample_weight = big_sample_weight / tf.reduce_mean(big_sample_weight)
        loss_dict['Vmax'] = loss_function(label_dict['Vmax'], pred_dict['Vmax'], sample_weight=normalized_sample_weight)

    if 'profile' in pred_dict:
        have_valid_profile = tf.cast(label_dict['profile'][:, 0] != -1, tf.float32)
        loss_dict['profile'] = loss_function(label_dict['profile'], pred_dict['profile'], sample_weight=have_valid_profile)

    if 'R34' in pred_dict:
        loss_dict['R34'] = loss_function(label_dict['R34'], pred_dict['R34'])

    if 'RMW' in pred_dict:
        loss_dict['RMW'] = loss_function(label_dict['RMW'], pred_dict['RMW'])

    return loss_dict


def get_sample_data(dataset, count):
    for batch_index, (images, feature, profile, Vmax, R34) in dataset.enumerate():
        valid_profile = profile[:, 0] != -999
        preprocessed_images = image_augmentation(images)
        sample_images = tf.boolean_mask(preprocessed_images, valid_profile)[:count, ...]
        sample_feature = tf.boolean_mask(feature, valid_profile)[:count, ...]
        sample_profile = tf.boolean_mask(profile, valid_profile)[:count, ...]
        sample_Vmax = tf.boolean_mask(Vmax, valid_profile)[:count, ...]
        sample_R34 = tf.boolean_mask(R34, valid_profile)[:count, ...]
        return sample_images, sample_feature, sample_profile, sample_Vmax, sample_R34


def rotation_blending(model, blending_num, images, feature):
    evenly_rotated_images = evenly_rotate(images, blending_num)
    pred_list = []
    for image in evenly_rotated_images:
        pred_label = model(image, feature, training=False)
        pred_list.append(pred_label)

    blended_pred = tf.reduce_mean(pred_list, 0)
    return blended_pred


def apply_loss_ratio_to_losses(loss_dict, loss_ratio):
    for loss_type in loss_dict:
        loss_dict[loss_type] *= loss_ratio.get(loss_type, 0.0)
    return loss_dict


def evaluate_blending_loss(model, dataset, loss_function, blending_num=10):
    if loss_function == 'MSE':
        loss = tf.keras.losses.MeanSquaredError()
    elif loss_function == 'MAE':
        loss = tf.keras.losses.MeanAbsoluteError()

    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))

    for batch_index, (images, feature, profile, Vmax, R34) in dataset.enumerate():
        blended_pred = rotation_blending(model, blending_num, images, feature)
        pred_dict = transfer_label_to_dict(blended_pred)
        if is_profile_label(blended_pred):
            label_dict = transfer_label_to_dict(profile)
        else:
            label_dict = transfer_label_to_dict(Vmax)

        for pred_type in pred_dict:
            batch_loss = loss(label_dict[pred_type], pred_dict[pred_type])
            avg_losses[pred_type].update_state(batch_loss)

    blending_loss_dict = {
        loss_type: avg_loss.result()
        for loss_type, avg_loss in avg_losses.items()
    }

    return blending_loss_dict


def do_blending_evaluation_and_write_summary(epoch_index, summary_writer, model, datasets, loss_function, profiler_loss_ratio):
    # calculate blending loss
    total_loss = {}
    for phase in ['train', 'valid']:
        blending_loss_dict = evaluate_blending_loss(model, datasets[phase], loss_function)
        with summary_writer.as_default():
            for loss_type, loss_value in blending_loss_dict.items():
                tf.summary.scalar(f'[{phase}] blending_{loss_type}_loss', loss_value, step=epoch_index)
        if profiler_loss_ratio:
            blending_loss_dict = apply_loss_ratio_to_losses(blending_loss_dict, profiler_loss_ratio)
        total_loss[phase] = sum(blending_loss_dict.values())

    return total_loss['train'], total_loss['valid']


def upsampling_good_quality_VIS_data(is_good_quality_VIS):
    good_quality_count = tf.reduce_sum(is_good_quality_VIS)
    total_count = tf.reduce_sum(tf.ones_like(is_good_quality_VIS))
    good_quality_rate = good_quality_count / total_count
    sample_weight_after_upsampling = is_good_quality_VIS / good_quality_rate
    return sample_weight_after_upsampling


def replace_original_channel_with_generation(datasets, generator, replace_VIS=True, replace_PMW=True):

    def generate_new_images(generator, images, feature, replace_VIS, replace_PMW):
        generated_VIS, generated_PMW = generator(images, feature)
        IR1_WV = tf.gather(images, axis=-1, indices=[0, 1])
        VIS = generated_VIS if replace_VIS else tf.gather(images, axis=-1, indices=[2])
        PMW = generated_PMW if replace_PMW else tf.gather(images, axis=-1, indices=[3])
        new_images = tf.concat([IR1_WV, VIS, PMW], axis=-1)
        return new_images.numpy()

    def unpack_replace_repack(dataset, generator, replace_VIS, replace_PMW, mini_batch=3):
        prefetched_dataset = dataset
        batched_dataset = prefetched_dataset._input_dataset
        shuffled_dataset = batched_dataset._input_dataset
        zipped_dataset = shuffled_dataset._input_dataset
        images_dataset = zipped_dataset._datasets[0]
        feature_dataset = zipped_dataset._datasets[1]
        profile_dataset = zipped_dataset._datasets[2]
        Vmax_dataset = zipped_dataset._datasets[3]
        R34_dataset = zipped_dataset._datasets[4]

        prefetch_buffer = prefetched_dataset._buffer_size
        batch_size = batched_dataset._batch_size
        shuffle_buffer = shuffled_dataset._buffer_size

        new_images_batch_numpy_list = [
            generate_new_images(
                generator, images, feature,
                replace_VIS, replace_PMW
            )
            for images, feature in tf.data.Dataset.zip((images_dataset, feature_dataset)).batch(mini_batch)
        ]
        new_images_tensor = np.concatenate(new_images_batch_numpy_list)

        new_images_dataset = tf.data.Dataset.from_tensor_slices(new_images_tensor)
        new_zipped_dataset = tf.data.Dataset.zip((
            new_images_dataset, feature_dataset, profile_dataset, Vmax_dataset, R34_dataset
        ))
        new_shuffled_dataset = new_zipped_dataset.shuffle(shuffle_buffer)
        new_batched_dataset = new_shuffled_dataset.batch(batch_size)
        new_prefetched_dataset = new_batched_dataset.prefetch(prefetch_buffer)
        return new_prefetched_dataset

    new_datasets = {}
    for phase in datasets:
        new_datasets[phase] = unpack_replace_repack(datasets[phase], generator, replace_VIS, replace_PMW)

    del datasets
    return new_datasets


def phase_rules_interpretor(phase_rules):
    if not phase_rules:
        default_years_dict = {
            'train': list(range(2004, 2015)),
            'valid': list(range(2015, 2017)),
            'test': list(range(2017, 2019))
        }
        return default_years_dict

    years_dict = {}
    for phase, rules in phase_rules.items():
        years = set()
        if 'range' in rules:
            start, end = rules['range']
            years = years.union(range(start, end))
        if 'exclude' in rules:
            years -= set(rules['exclude'])
        if 'add' in rules:
            years = years.union(rules['add'])

        years_dict[phase] = list(years)
    return years_dict


def get_tensorflow_datasets(
    data_folder, batch_size, shuffle_buffer, prefetch_buffer,
    good_VIS_only=False, valid_profile_only=False, coordinate='cart', phase_rules={}
):
    years_dict = phase_rules_interpretor(phase_rules)
    datasets = dict()
    for phase, year_list in years_dict.items():
        phase_data = load_dataset(data_folder, year_list, good_VIS_only, valid_profile_only, coordinate)
        images_tensor = phase_data['image']
        feature_tensor = phase_data['feature'].to_numpy(dtype='float32')
        profile_tensor = phase_data['profile']
        Vmax_tensor = phase_data['label'][['Vmax']].to_numpy(dtype='float32')
        R34_tensor = phase_data['label'][['R34']].to_numpy(dtype='float32')

        print(phase, images_tensor.shape)
        images_dataset = tf.data.Dataset.from_tensor_slices(images_tensor)
        feature_dataset = tf.data.Dataset.from_tensor_slices(feature_tensor)
        profile_dataset = tf.data.Dataset.from_tensor_slices(profile_tensor)
        Vmax_dataset = tf.data.Dataset.from_tensor_slices(Vmax_tensor)
        R34_dataset = tf.data.Dataset.from_tensor_slices(R34_tensor)

        zipped_dataset = tf.data.Dataset.zip((images_dataset, feature_dataset, profile_dataset, Vmax_dataset, R34_dataset))
        shuffled_dataset = zipped_dataset.shuffle(shuffle_buffer)
        batched_dataset = shuffled_dataset.batch(batch_size)
        prefetched_dataset = batched_dataset.prefetch(prefetch_buffer)

        datasets[phase] = prefetched_dataset

    return datasets
