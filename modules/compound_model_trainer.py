import tensorflow as tf
from collections import defaultdict
from modules.training_helper import do_blending_evaluation_and_write_summary, \
    get_sample_data, upsampling_good_quality_VIS_data, \
    apply_loss_ratio_to_losses, calculate_loss_dict, is_profile_label,\
    replace_original_channel_with_generation, image_augmentation
from modules.illustrate_helper import output_sample_profile_chart, \
    output_sample_generation, draw_original_image


def train_compound_model(
    compound_model,
    datasets,
    summary_writer,
    saving_path,
    evaluate_freq,
    max_epoch,
    optimizing_target,
    use_VIS_channel,
    use_PMW_channel,
    G_D_loss_ratio={},
    loss_function='MSE',
    profiler_loss_ratio={},
    random_target_m2n=False,
    Vmax_loss_sample_weight_exponent=0
):
    G_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    D_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    R_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    P_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    BC = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    MSE = tf.keras.losses.MeanSquaredError()
    if loss_function == 'MSE':
        loss = MSE
    elif loss_function == 'MAE':
        loss = tf.keras.losses.MeanAbsoluteError()
    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))

    @tf.function
    def train_R_P_step(model, images, feature, profile, Vmax, R34, training=True):
        with tf.GradientTape() as tape:
            pred_label = model(images, feature, training)
            if is_profile_label(pred_label):
                loss_dict = calculate_loss_dict(pred_label, loss, profile, Vmax_loss_sample_weight_exponent)
            else:
                loss_dict = calculate_loss_dict(pred_label, loss, Vmax, Vmax_loss_sample_weight_exponent)

            for loss_type, loss_value in loss_dict.items():
                avg_losses[f'Predictor: {loss_type}_{loss_function}_loss'].update_state(loss_value)

            if profiler_loss_ratio:
                loss_dict = apply_loss_ratio_to_losses(loss_dict, profiler_loss_ratio)

            predict_loss = sum(loss_dict.values())
            avg_losses[f'Predictor: overall_{loss_function}_loss'].update_state(predict_loss)

        if training is True:
            if compound_model.regressor is not None:
                R_gradients = tape.gradient(predict_loss, compound_model.regressor.trainable_variables)
                R_optimizer.apply_gradients(zip(R_gradients, compound_model.regressor.trainable_variables))
            elif compound_model.profiler is not None:
                P_gradients = tape.gradient(predict_loss, compound_model.profiler.trainable_variables)
                P_optimizer.apply_gradients(zip(P_gradients, compound_model.profiler.trainable_variables))
        return

    @tf.function
    def train_G_D_step(
        compound_model, images, feature, profile, Vmax, R34,
        random_target_m2n, use_VIS_channel, use_PMW_channel, training=True
    ):
        # prepare some material
        minutes_to_noon = feature[:, 7:8]
        '''
        Since we use L2 distance, we need to use same target m2n as the label in training,
        else the model would try to fit the brightness difference from IR1 and WV.
        '''
        if random_target_m2n:
            target_minutes_to_noon = tf.random.uniform(shape=minutes_to_noon.shape, maxval=300)
        else:
            target_minutes_to_noon = minutes_to_noon

        # get sample weight for upsampling data having good quality VIS
        is_good_quality_VIS = feature[:, 8:9]
        sample_weight = upsampling_good_quality_VIS_data(is_good_quality_VIS)

        # joinly train generator and discriminator
        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            fake_images = compound_model.generate_fake_images(images, feature, target_minutes_to_noon, training=training)
            # discriminator
            pred_minutes_to_noon, real_VIS_judgement, real_PMW_judgement = compound_model.discriminator(images, minutes_to_noon, training=training)
            fake_minutes_to_noon, fake_VIS_judgement, fake_PMW_judgement = compound_model.discriminator(fake_images, target_minutes_to_noon, training=training)
            # --------------------------------------------------------------------
            if not training or G_D_loss_ratio.get('predict_loss', 0.0):
                pred_label = compound_model(images, feature, training=True)
                if is_profile_label(pred_label):
                    loss_dict = calculate_loss_dict(pred_label, loss, profile, Vmax_loss_sample_weight_exponent)
                else:
                    loss_dict = calculate_loss_dict(pred_label, loss, Vmax, Vmax_loss_sample_weight_exponent)
                if profiler_loss_ratio:
                    loss_dict = apply_loss_ratio_to_losses(loss_dict, profiler_loss_ratio)
                predict_loss = sum(loss_dict.values())
            else:
                predict_loss = tf.convert_to_tensor(0.0)
            # --------------------------------------------------------------------
            if not training or G_D_loss_ratio.get('VIS_GAN', 0.0):
                real_VIS_judgement_loss = BC(tf.ones_like(real_VIS_judgement), real_VIS_judgement, sample_weight=sample_weight)
                fake_VIS_judgement_loss = BC(tf.zeros_like(fake_VIS_judgement), fake_VIS_judgement)
                VIS_judgement_loss = real_VIS_judgement_loss + fake_VIS_judgement_loss
                VIS_disguise_loss = BC(tf.ones_like(fake_VIS_judgement), fake_VIS_judgement)
            else:
                VIS_judgement_loss = tf.convert_to_tensor(0.0)
                VIS_disguise_loss = tf.convert_to_tensor(0.0)
            # --------------------------------------------------------------------
            if not training or G_D_loss_ratio.get('PMW_GAN', 0.0):
                real_PMW_judgement_loss = BC(tf.ones_like(real_PMW_judgement), real_PMW_judgement)
                fake_PMW_judgement_loss = BC(tf.zeros_like(fake_PMW_judgement), fake_PMW_judgement)
                PMW_judgement_loss = (real_PMW_judgement_loss + fake_PMW_judgement_loss)
                PMW_disguise_loss = BC(tf.ones_like(fake_PMW_judgement), fake_PMW_judgement)
            else:
                PMW_judgement_loss = tf.convert_to_tensor(0.0)
                PMW_disguise_loss = tf.convert_to_tensor(0.0)
            # --------------------------------------------------------------------
            if not training or G_D_loss_ratio.get('VIS_hours', 0.0):
                VIS_pred_hour_loss = loss(minutes_to_noon, pred_minutes_to_noon, sample_weight=is_good_quality_VIS)
                VIS_tuning_hour_loss = loss(target_minutes_to_noon, fake_minutes_to_noon)
            else:
                VIS_pred_hour_loss = tf.convert_to_tensor(0.0)
                VIS_tuning_hour_loss = tf.convert_to_tensor(0.0)

            if not training or G_D_loss_ratio.get('VIS_L2', 0.0):
                true_VIS = tf.gather(images, axis=-1, indices=[2])
                fake_VIS = tf.gather(fake_images, axis=-1, indices=[2])
                VIS_L2_loss = MSE(true_VIS, fake_VIS, sample_weight=is_good_quality_VIS)
            else:
                VIS_L2_loss = tf.convert_to_tensor(0.0)

            if not training or G_D_loss_ratio.get('PMW_L2', 0.0):
                true_PMW = tf.gather(images, axis=-1, indices=[3])
                fake_PMW = tf.gather(fake_images, axis=-1, indices=[3])
                PMW_L2_loss = MSE(true_PMW, fake_PMW)
            else:
                PMW_L2_loss = tf.convert_to_tensor(0.0)

            total_discriminator_loss = \
                VIS_pred_hour_loss * G_D_loss_ratio.get('VIS_hours', 0.0) \
                + VIS_judgement_loss * G_D_loss_ratio.get('VIS_GAN', 0.0) \
                + PMW_judgement_loss * G_D_loss_ratio.get('PMW_GAN', 0.0)

            total_generator_loss = \
                VIS_tuning_hour_loss * G_D_loss_ratio.get('VIS_hours', 0.0) \
                + VIS_disguise_loss * G_D_loss_ratio.get('VIS_GAN', 0.0) \
                + PMW_disguise_loss * G_D_loss_ratio.get('PMW_GAN', 0.0) \
                + VIS_L2_loss * G_D_loss_ratio.get('VIS_L2', 0.0) \
                + PMW_L2_loss * G_D_loss_ratio.get('PMW_L2', 0.0) \
                + predict_loss * G_D_loss_ratio.get('predict_loss', 0.0)

        avg_losses['Predictor: predict_loss'].update_state(predict_loss)
        avg_losses['Generator: VIS_disguise_loss'].update_state(VIS_disguise_loss)
        avg_losses['Generator: PMW_disguise_loss'].update_state(PMW_disguise_loss)
        avg_losses['Generator: VIS_tuning_hour_loss'].update_state(VIS_tuning_hour_loss)
        avg_losses['Generator: VIS_L2_loss'].update_state(VIS_L2_loss)
        avg_losses['Generator: PMW_L2_loss'].update_state(PMW_L2_loss)
        avg_losses['Generator: total_loss'].update_state(total_generator_loss)
        avg_losses['Discriminator: VIS_judgement_loss'].update_state(VIS_judgement_loss)
        avg_losses['Discriminator: PMW_judgement_loss'].update_state(PMW_judgement_loss)
        avg_losses['Discriminator: VIS_pred_hour_loss'].update_state(VIS_pred_hour_loss)
        avg_losses['Discriminator: total_loss'].update_state(total_discriminator_loss)

        if training:
            D_gradients = D_tape.gradient(total_discriminator_loss, compound_model.discriminator.trainable_variables)
            G_gradients = G_tape.gradient(total_generator_loss, compound_model.generator.trainable_variables)
            D_optimizer.apply_gradients(zip(D_gradients, compound_model.discriminator.trainable_variables))
            G_optimizer.apply_gradients(zip(G_gradients, compound_model.generator.trainable_variables))

        return

    sample_data = {
        phase: get_sample_data(datasets[phase], 10)
        for phase in ['train', 'valid']
    }
    draw_original_image(sample_data, summary_writer)

    if use_VIS_channel == 'fixed_generation' or use_PMW_channel == 'fixed_generation':
        datasets = replace_original_channel_with_generation(
            datasets, compound_model.generator,
            replace_VIS=(use_VIS_channel == 'fixed_generation'),
            replace_PMW=(use_PMW_channel == 'fixed_generation')
        )

    compound_model.set_generate_VIS(use_VIS_channel == 'dynamic_generation')
    compound_model.set_generate_PMW(use_PMW_channel == 'dynamic_generation')
    compound_model.set_freeze_VIS_generator(not G_D_loss_ratio.get('VIS_GAN', 0.0))
    compound_model.set_freeze_PMW_generator(not G_D_loss_ratio.get('PMW_GAN', 0.0))

    best_optimizing_loss = None
    for epoch_index in range(1, max_epoch+1):
        print(f'Executing epoch #{epoch_index}')
        for batch_index, (images, feature, profile, Vmax, R34) in datasets['train'].enumerate():
            preprocessed_images = image_augmentation(images)

            if optimizing_target == 'generator':
                train_G_D_step(
                    compound_model, preprocessed_images, feature, profile, Vmax, R34,
                    random_target_m2n, use_VIS_channel, use_PMW_channel
                )
            if optimizing_target in ['regressor', 'profiler']:
                train_R_P_step(compound_model, preprocessed_images, feature, profile, Vmax, R34)

        for loss_name, avg_loss in avg_losses.items():
            with summary_writer.as_default():
                tf.summary.scalar(f'[train] {loss_name}', avg_loss.result(), step=epoch_index)
            avg_loss.reset_states()

        if epoch_index % evaluate_freq == 0:
            print(f'Completed {epoch_index} epochs, do some evaluation')
            # draw profile chart, will do nothing if the model is regressor but not profiler.
            output_sample_profile_chart(compound_model, sample_data, summary_writer, epoch_index)
            # calculate blending loss
            train_blending_loss, valid_blending_loss = do_blending_evaluation_and_write_summary(
                epoch_index, summary_writer, compound_model, datasets,
                loss_function, profiler_loss_ratio
            )

            if optimizing_target == 'generator':
                output_sample_generation(compound_model.generator, sample_data, summary_writer, epoch_index, 'general_progress')
                # calculate generator loss on validation data
                for valid_batch_index, (valid_images, valid_feature, valid_profile, valid_Vmax, valid_R34) in datasets['valid'].enumerate():
                    preprocessed_images = image_augmentation(valid_images)
                    train_G_D_step(
                        compound_model, preprocessed_images,
                        valid_feature, valid_profile, valid_Vmax, valid_R34,
                        random_target_m2n, use_VIS_channel, use_PMW_channel,
                        training=False
                    )
                valid_generator_total_loss = avg_losses['Generator: total_loss'].result()

            if optimizing_target in ['regressor', 'profiler']:
                for valid_batch_index, (valid_images, valid_feature, valid_profile, valid_Vmax, valid_R34) in datasets['valid'].enumerate():
                    train_R_P_step(
                        compound_model, valid_images,
                        valid_feature, valid_profile, valid_Vmax, valid_R34,
                        training=False
                    )

            for loss_name, avg_loss in avg_losses.items():
                with summary_writer.as_default():
                    tf.summary.scalar(f'[valid] {loss_name}', avg_loss.result(), step=epoch_index)
                avg_loss.reset_states()

            if optimizing_target in ['regressor', 'profiler']:
                valid_optimizing_loss = valid_blending_loss
            elif optimizing_target == 'generator':
                valid_optimizing_loss = valid_generator_total_loss

            if best_optimizing_loss is None or best_optimizing_loss >= valid_optimizing_loss:
                best_optimizing_loss = valid_optimizing_loss
                print(f'Get best loss so far at epoch {epoch_index}! Saving the model.')
                compound_model.save_weights(saving_path, save_format='tf')
                if optimizing_target == 'generator':
                    output_sample_generation(compound_model.generator, sample_data, summary_writer, epoch_index, 'saved_generator')
