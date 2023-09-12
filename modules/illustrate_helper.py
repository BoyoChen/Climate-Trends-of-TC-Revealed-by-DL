import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio
from modules.image_processor import is_polar_coordinate, polar2cart, cart2polar
from modules.training_helper import transfer_label_to_dict


def draw_original_image(sample_data, summary_writer):
    for phase, (sample_images, sample_feature, sample_profile, sample_Vmax, sample_R34) in sample_data.items():
        if is_polar_coordinate(sample_images):
            polar_VIS = tf.gather(sample_images, axis=-1, indices=[2])
            polar_PMW = tf.gather(sample_images, axis=-1, indices=[3])
            cart_VIS = tf.map_fn(lambda x: polar2cart(x), polar_VIS)
            cart_PMW = tf.map_fn(lambda x: polar2cart(x), polar_PMW)
        else:
            cart_VIS = tf.gather(sample_images, axis=-1, indices=[2])
            cart_PMW = tf.gather(sample_images, axis=-1, indices=[3])
            polar_VIS = tf.map_fn(lambda x: cart2polar(x), cart_VIS)
            polar_PMW = tf.map_fn(lambda x: cart2polar(x), cart_PMW)

        with summary_writer.as_default():
            tf.summary.image(phase + '_original_polar_VIS', polar_VIS, step=0, max_outputs=polar_VIS.shape[0])
            tf.summary.image(phase + '_original_polar_PMW', polar_PMW, step=0, max_outputs=polar_PMW.shape[0])
            tf.summary.image(phase + '_original_cart_VIS', cart_VIS, step=0, max_outputs=cart_VIS.shape[0])
            tf.summary.image(phase + '_original_cart_PMW', cart_PMW, step=0, max_outputs=cart_PMW.shape[0])


def output_sample_generation(generator, sample_data, summary_writer, epoch_index, postfix):
    for phase in ['train', 'valid']:
        sample_images, sample_feature, sample_profile, sample_Vmax, sample_R34 = sample_data[phase]
        generated_VIS, generated_PMW = generator(sample_images, sample_feature)
        if is_polar_coordinate(generated_VIS):
            with summary_writer.as_default():
                tf.summary.image(
                    phase + '_VIS_polar_' + postfix,
                    generated_VIS,
                    step=epoch_index+1,
                    max_outputs=generated_VIS.shape[0]
                )
                tf.summary.image(
                    phase + '_PMW_polar_' + postfix,
                    generated_PMW,
                    step=epoch_index+1,
                    max_outputs=generated_PMW.shape[0]
                )
            generated_cart_VIS = tf.map_fn(lambda x: polar2cart(x), generated_VIS)
            generated_cart_PMW = tf.map_fn(lambda x: polar2cart(x), generated_PMW)
        else:
            generated_cart_VIS = generated_VIS
            generated_cart_PMW = generated_PMW

        with summary_writer.as_default():
            tf.summary.image(
                phase + '_VIS_cart_' + postfix,
                generated_cart_VIS,
                step=epoch_index+1,
                max_outputs=generated_cart_VIS.shape[0]
            )
            tf.summary.image(
                phase + '_PMW_cart_' + postfix,
                generated_cart_PMW,
                step=epoch_index+1,
                max_outputs=generated_cart_PMW.shape[0]
            )


def output_sample_profile_chart(profiler, sample_data, summary_writer, epoch_index):
    for phase, (sample_images, sample_feature, sample_profile, sample_Vmax, sample_R34) in sample_data.items():
        pred_label = profiler(sample_images, sample_feature, training=False)
        pred_dict = transfer_label_to_dict(pred_label)
        if 'profile' not in pred_dict:
            # will do nothing if the model is regressor but not profiler.
            return
        charts = []
        for i in range(10):
            charts.append(
                _draw_profile_chart(sample_profile[i], pred_dict['profile'][i], pred_dict['Vmax'][i], pred_dict['R34'][i])
            )
        chart_matrix = np.stack(charts).astype(np.int)
        with summary_writer.as_default():
            tf.summary.image(
                f'{phase}_profile_chart',
                chart_matrix.astype(np.float)/255,
                step=epoch_index,
                max_outputs=chart_matrix.shape[0]
            )


def _draw_profile_chart(profile, pred_profile, calculated_Vmax, calculated_R34):
    tmp_id = np.random.randint(10000000)
    km = np.arange(0, 751, 5)
    plt.figure(figsize=(15, 10), linewidth=2)
    plt.plot(km, profile, color='r', label="profile")
    plt.plot(km, pred_profile, color='g', label="pred_profile")
    plt.axhline(y=calculated_Vmax, color='b', linestyle='-')
    plt.axvline(x=calculated_R34, color='y', linestyle='-')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 120)
    plt.xlabel("Radius", fontsize=30)
    plt.ylabel("Velocity", fontsize=30)
    plt.legend(loc="best", fontsize=20)
    plt.savefig(f'tmp-{tmp_id}.png')
    plt.close('all')
    RGB_matrix = imageio.imread(f'tmp-{tmp_id}.png')
    os.remove(f'tmp-{tmp_id}.png')
    return RGB_matrix
