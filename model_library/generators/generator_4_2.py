import tensorflow as tf
from tensorflow.keras import layers

# ------------
# based on generator_4_1
# modify the structure to be stronger
# ------------


class DownSample(layers.Layer):
    def __init__(self, filters, size, strides, initializer, apply_batch_norm):
        super().__init__()
        self.conv = layers.Conv2D(filters, size, strides, padding='same', use_bias=False, kernel_initializer=initializer)
        self.batch_norm = layers.BatchNormalization() if apply_batch_norm else None
        self.leaky_relu = layers.LeakyReLU()

    def call(self, x, training):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x, training=training)
        x = self.leaky_relu(x)
        return x


class UpSample(layers.Layer):
    def __init__(self, filters, size, strides, initializer, apply_dropout):
        super().__init__()
        self.reverse_conv = layers.Conv2DTranspose(filters, size, strides, padding='same', use_bias=False, kernel_initializer=initializer)
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.5) if apply_dropout else None
        self.relu = layers.ReLU()

    def call(self, x, training, crop_height=0, crop_width=0):
        x = self.reverse_conv(x)
        if crop_height:
            x = x[:, :crop_height, :, :]
        if crop_width:
            x = x[:, :, :crop_width, :]
        x = self.batch_norm(x, training=training)
        if self.dropout:
            x = self.dropout(x, training=training)
        x = self.relu(x)
        return x


class subModel(layers.Layer):
    # a modified U-Net
    def __init__(self, output_channel_num=1):
        super().__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.down_layers = [
            DownSample(64, [4, 3], 2, initializer, apply_batch_norm=False),
            DownSample(128, [4, 3], 2, initializer, apply_batch_norm=True),
            DownSample(256, [4, 3], 2, initializer, apply_batch_norm=True),
            DownSample(256, [4, 3], 2, initializer, apply_batch_norm=True),
            DownSample(256, [4, 3], 2, initializer, apply_batch_norm=True),
            DownSample(512, [4, 3], 2, initializer, apply_batch_norm=True)
        ]
        self.up_layers = [
            UpSample(512, [4, 3], 2, initializer, apply_dropout=True),
            UpSample(256, [4, 3], 2, initializer, apply_dropout=True),
            UpSample(256, [4, 3], 2, initializer, apply_dropout=False),
            UpSample(256, [4, 3], 2, initializer, apply_dropout=False),
            UpSample(128, [4, 3], 2, initializer, apply_dropout=False),
            UpSample(64, [4, 3], 2, initializer, apply_dropout=False)
        ]
        self.last_layer = layers.Conv2D(
            output_channel_num, 4,
            strides=1,
            kernel_initializer=initializer,
            padding='same',
            activation='relu'
        )

    def auxiliary_feature(self, feature, target_minutes_to_noon, height, width):
        # feature: ['lon', 'lat', 'region_code', 'yday_cos', 'yday_sin', 'hour_cos', 'hour_sin', 'minutes_to_noon', 'is_good_quality_VIS']
        lon_lat = feature[:, :2]
        region_code_one_hot = tf.one_hot(tf.cast(feature[:, 2], tf.int32), 6)
        yday = feature[:, 3:5]
        flat_feature = tf.concat(
            [
                lon_lat,
                region_code_one_hot,
                yday,
                target_minutes_to_noon
            ], 1
        )

        # flat_feature shape: (batch_size, 11) -> four_dim_feature shape: (batch_size, 1, 1, 11)
        four_dim_feature = tf.expand_dims(
            tf.expand_dims(
                flat_feature, 1
            ), 1
        )

        # four_dim_feature shape: (batch_size, 1, 1, 11) -> broadcasted_feature shape: (batch_size, height, width, 11)
        broadcasted_feature = tf.tile(four_dim_feature, [1, height, width, 1])
        return broadcasted_feature

    def call(self, image, feature, target_minutes_to_noon, training):
        down_sample_stages = []
        # not using VIS AND PMW here!
        image = tf.gather(image, axis=-1, indices=[0, 1])
        batch_size, original_image_height, original_image_width, channel_num = image.shape
        for down_layer in self.down_layers:
            image = down_layer(image, training=training)
            down_sample_stages.append(image)

        condensed_image = down_sample_stages.pop()

        batch_size, height, width, channel_num = condensed_image.shape
        auxiliary_feature = self.auxiliary_feature(feature, target_minutes_to_noon, height, width)
        condensed_image = tf.concat([condensed_image, auxiliary_feature], axis=-1)

        for up_layer in self.up_layers:
            if down_sample_stages:
                down_sample_stage = down_sample_stages.pop()
                batch_size, height, width, channel_num = down_sample_stage.shape
                condensed_image = up_layer(condensed_image, training=training, crop_height=height, crop_width=width)
                condensed_image = tf.concat([condensed_image, down_sample_stage], axis=-1)
            else:
                condensed_image = up_layer(condensed_image, training=training, crop_height=original_image_height, crop_width=original_image_width)

        generated_image = self.last_layer(condensed_image)

        return generated_image


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.VIS_generator = subModel()
        self.PMW_generator = subModel()

    def __call__(self, image, feature, target_minutes_to_noon=None, training=False, channel='both'):
        if target_minutes_to_noon is None:
            target_minutes_to_noon = tf.zeros_like(feature[:, 0:1])
        if channel == 'both':
            generated_VIS = self.VIS_generator(image, feature, target_minutes_to_noon, training)
            generated_PMW = self.PMW_generator(image, feature, target_minutes_to_noon, training)
            return generated_VIS, generated_PMW
        if channel == 'PMW':
            generated_PMW = self.PMW_generator(image, feature, target_minutes_to_noon, training)
            return generated_PMW
        if channel == 'VIS':
            generated_VIS = self.VIS_generator(image, feature, target_minutes_to_noon, training)
            return generated_VIS
