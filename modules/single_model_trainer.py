import tensorflow as tf
from collections import defaultdict
from modules.training_helper import do_blending_evaluation_and_write_summary, is_profile_label,\
    get_sample_data, apply_loss_ratio_to_losses, calculate_loss_dict, image_augmentation
from modules.illustrate_helper import output_sample_profile_chart


def train_single_model(
    model,
    datasets,
    summary_writer,
    saving_path,
    evaluate_freq,
    max_epoch,
    early_stop_tolerance=None,
    overfit_tolerance=None,
    loss_function='MSE',
    profiler_loss_ratio={},
    Vmax_loss_sample_weight_exponent=0  # only applied to Vmax loss
):
    optimizer = tf.keras.optimizers.Adam()
    if loss_function == 'MSE':
        loss = tf.keras.losses.MeanSquaredError()
    elif loss_function == 'MAE':
        loss = tf.keras.losses.MeanAbsoluteError()
    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))

    @tf.function
    def train_step(model, image, feature, profile, Vmax, R34, training=True):
        with tf.GradientTape() as tape:
            pred_label = model(image, feature, training)
            if is_profile_label(pred_label):
                loss_dict = calculate_loss_dict(pred_label, loss, profile, Vmax_loss_sample_weight_exponent)
            else:
                loss_dict = calculate_loss_dict(pred_label, loss, Vmax, Vmax_loss_sample_weight_exponent)

            for loss_type, loss_value in loss_dict.items():
                avg_losses[f'{loss_type}_{loss_function}_loss'].update_state(loss_value)

            if profiler_loss_ratio:
                loss_dict = apply_loss_ratio_to_losses(loss_dict, profiler_loss_ratio)

            total_loss = sum(loss_dict.values())

        avg_losses[f'overall_{loss_function}_loss'].update_state(total_loss)
        if training is True:
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return

    sample_data = {
        phase: get_sample_data(datasets[phase], 10)
        for phase in ['train', 'valid']
    }

    # use stack to keep track on validation loss and help early stopping
    valid_loss_stack = []
    for epoch_index in range(1, max_epoch+1):
        print(f'Executing epoch #{epoch_index}')
        for batch_index, (images, feature, profile, Vmax, R34) in datasets['train'].enumerate():
            preprocessed_images = image_augmentation(images)
            train_step(model, preprocessed_images, feature, profile, Vmax, R34)

        with summary_writer.as_default():
            for loss_name, avg_loss in avg_losses.items():
                tf.summary.scalar(f'[train] {loss_name}', avg_loss.result(), step=epoch_index)
                avg_loss.reset_states()

        if epoch_index % evaluate_freq == 0:
            print(f'Completed {epoch_index} epochs, do some evaluation')
            # draw profile chart, will do nothing if the model is regressor but not profiler.
            output_sample_profile_chart(model, sample_data, summary_writer, epoch_index)
            # calculate blending loss
            train_blending_loss, valid_blending_loss = do_blending_evaluation_and_write_summary(
                epoch_index, summary_writer, model, datasets,
                loss_function, profiler_loss_ratio
            )

            for valid_batch_index, (valid_images, valid_feature, valid_profile, valid_Vmax, valid_R34) in datasets['valid'].enumerate():
                train_step(
                    model, valid_images,
                    valid_feature, valid_profile, valid_Vmax, valid_R34,
                    training=False
                )
            for loss_name, avg_loss in avg_losses.items():
                with summary_writer.as_default():
                    tf.summary.scalar(f'[valid] {loss_name}', avg_loss.result(), step=epoch_index)
                avg_loss.reset_states()

            # save the best model and check for early stopping
            while valid_loss_stack and valid_loss_stack[-1] >= valid_blending_loss:
                valid_loss_stack.pop()
            if not valid_loss_stack:
                model.save_weights(saving_path, save_format='tf')
                print('Get the best validation performance so far! Saving the model.')
            elif early_stop_tolerance and len(valid_loss_stack) > early_stop_tolerance:
                print('Exceed the early stop tolerance, training procedure will end!')
                break
            elif overfit_tolerance and (valid_blending_loss - train_blending_loss) >= overfit_tolerance:
                print('Exceed the orverfit tolerance, training procedure will end!')
                # since valid loss is using blending, if train loss can beat valid loss,
                # that probably means model is already overfitting.
                break
            valid_loss_stack.append(valid_blending_loss)
