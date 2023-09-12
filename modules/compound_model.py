import tensorflow as tf


class CompoundModel():
    def __init__(self, generator, discriminator, regressor=None, profiler=None):
        self.generator = generator
        self.discriminator = discriminator
        self.regressor = regressor
        self.profiler = profiler
        self.generate_VIS = True
        self.generate_PMW = True
        self.freeze_VIS_generator = False
        self.freeze_PMW_generator = False

    def set_generate_VIS(self, new_val):
        self.generate_VIS = new_val

    def set_generate_PMW(self, new_val):
        self.generate_PMW = new_val

    def set_freeze_VIS_generator(self, new_val):
        self.freeze_VIS_generator = new_val

    def set_freeze_PMW_generator(self, new_val):
        self.freeze_PMW_generator = new_val

    def save_weights(self, saving_path, save_format):
        self.generator.save_weights(saving_path + '/' + 'generator', save_format=save_format)
        self.discriminator.save_weights(saving_path + '/' + 'discriminator', save_format=save_format)
        if self.regressor:
            self.regressor.save_weights(saving_path + '/' + 'regressor', save_format=save_format)
        if self.profiler:
            self.profiler.save_weights(saving_path + '/' + 'profiler', save_format=save_format)

    def generate_fake_images(self, images, feature, target_minutes_to_noon, training=False):
        IR1_WV = tf.gather(images, axis=-1, indices=[0, 1])
        if self.generate_VIS:
            VIS = self.generator(images, feature, target_minutes_to_noon, training=training, channel='VIS')
        else:
            VIS = tf.gather(images, axis=-1, indices=[2])

        if self.generate_PMW:
            PMW = self.generator(images, feature, target_minutes_to_noon, training=training, channel='PMW')
        else:
            PMW = tf.gather(images, axis=-1, indices=[3])

        if self.freeze_VIS_generator:
            VIS = tf.stop_gradient(VIS)

        if self.freeze_PMW_generator:
            PMW = tf.stop_gradient(PMW)

        return tf.concat([IR1_WV, VIS, PMW], axis=-1)

    def generate_noon_images(self, images, feature, training=False):
        minutes_to_noon = feature[:, 7:8]
        zero_minutes_to_noon = tf.zeros_like(minutes_to_noon)
        return self.generate_fake_images(images, feature, zero_minutes_to_noon, training=training)

    def __call__(self, images, feature, training=False):
        if self.generate_VIS or self.generate_PMW:
            input_images = self.generate_noon_images(images, feature, training)
        else:
            input_images = images

        if self.regressor:
            return self.regressor(input_images, feature, training=training)
        if self.profiler:
            return self.profiler(input_images, feature, training=training)
