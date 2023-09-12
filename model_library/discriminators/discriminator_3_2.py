import tensorflow as tf
from tensorflow.keras import layers

# ------------
# Based on discriminator_3_1
# modify the structure to be stronger
# ------------


class Model(tf.keras.Model):
    # a modified PatchGAN
    def __init__(self):
        super().__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        # ========= VIS discriminator =========
        self.VIS_down_layers = [
            layers.Conv2D(64, [4, 3], 2, padding='same', use_bias=False, kernel_initializer=initializer),
            layers.LeakyReLU(),
            layers.Conv2D(128, [4, 3], 2, padding='same', use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(256, [4, 3], 2, padding='same', use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.LeakyReLU()
        ]

        self.m2n_layers = [
            layers.Conv2D(256, [4, 3], 2, padding='same', use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(512, [4, 3], 2, padding='same', use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation=None)
        ]

        self.VIS_discriminator_layers = [
            layers.Conv2D(512, [4, 3], 1, padding='same', use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.ZeroPadding2D(),
            layers.Conv2D(1, [4, 3], 1, kernel_initializer=initializer)
        ]

        # ========= PMW discriminator =========
        self.PMW_layers = [
            layers.Conv2D(64, [4, 3], 2, padding='same', use_bias=False, kernel_initializer=initializer),
            layers.LeakyReLU(),
            layers.Conv2D(128, [4, 3], 2, padding='same', use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(256, [4, 3], 2, padding='same', use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(512, [4, 3], 1, padding='same', use_bias=False, kernel_initializer=initializer),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.ZeroPadding2D(),
            layers.Conv2D(1, [4, 3], 1, kernel_initializer=initializer)
        ]

    def apply_layers(self, x, layers, training):
        for layer in layers:
            x = layer(x, training=training)
        return x

    def discriminate_VIS(self, IR1_WV_VIS, minutes_to_noon, training):
        x = self.apply_layers(IR1_WV_VIS, self.VIS_down_layers, training)

        pred_minutes_to_noon = self.apply_layers(x, self.m2n_layers, training)

        minutes_to_noon_matrix = tf.map_fn(
            lambda m2n: tf.broadcast_to(m2n, x.shape[1:-1] + [1]),
            minutes_to_noon
        )
        x = tf.concat([x, minutes_to_noon_matrix], axis=-1)
        pred_real = self.apply_layers(x, self.VIS_discriminator_layers, training)

        return pred_minutes_to_noon, pred_real

    def discriminate_PMW(self, IR1_WV_PMW, training):
        x = self.apply_layers(IR1_WV_PMW, self.PMW_layers, training)
        return x

    def call(self, image, minutes_to_noon, training=False):
        IR1_WV_VIS = tf.gather(image, axis=-1, indices=[0, 1, 2])
        IR1_WV_PMW = tf.gather(image, axis=-1, indices=[0, 1, 3])
        pred_VIS_minutes_to_noon, pred_VIS_real = self.discriminate_VIS(IR1_WV_VIS, minutes_to_noon, training)
        pred_PMW_real = self.discriminate_PMW(IR1_WV_PMW, training)

        return pred_VIS_minutes_to_noon, pred_VIS_real, pred_PMW_real
