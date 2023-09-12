import tensorflow as tf
from tensorflow.keras import layers

# ------------
# Based on profiler_2_8
# use IR1 only
# ------------


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # batch_norm
        self.input_norm = layers.BatchNormalization()
        self.conv1_norm = layers.BatchNormalization()
        self.conv2_norm = layers.BatchNormalization()
        self.conv3_norm = layers.BatchNormalization()
        self.conv4_norm = layers.BatchNormalization()
        self.conv5_norm = layers.BatchNormalization()
        self.conv6_norm = layers.BatchNormalization()
        self.fc1_norm = layers.BatchNormalization()
        self.fc2_norm = layers.BatchNormalization()

        # conv layers
        self.conv1 = layers.Conv2D(16, kernel_size=(4, 3), strides=[2, 2], padding='same')
        self.conv2 = layers.Conv2D(32, kernel_size=(4, 3), strides=[2, 2], padding='same')
        self.conv3 = layers.Conv2D(64, kernel_size=(4, 3), strides=[2, 2], padding='same')
        self.conv4 = layers.Conv2D(128, kernel_size=(4, 3), strides=[2, 2], padding='same')
        self.conv5 = layers.Conv2D(256, kernel_size=(4, 3), strides=[2, 2], padding='same')
        self.conv6 = layers.Conv2D(512, kernel_size=(4, 3), strides=[2, 2], padding='same')
        self.flatten = layers.Flatten()

        # isolated relu function
        self.relu = layers.ReLU()

        # profile layers
        self.fc1 = layers.Dense(256)
        self.fc2 = layers.Dense(64)
        self.profile = layers.Dense(151, activation=None)

    def image_preprocessing(self, image, training):
        IR1 = tf.gather(image, axis=-1, indices=[0])
        normalized_image = self.input_norm(IR1, training=training)
        return normalized_image

    def auxiliary_feature(self, feature):
        # feature: ['lon', 'lat', 'region_code', 'yday_cos', 'yday_sin', 'hour_cos', 'hour_sin', 'minutes_to_noon', 'is_good_quality_VIS']
        region_code_one_hot = tf.one_hot(tf.cast(feature[:, 2], tf.int32), 6)
        yday_and_hour = feature[:, 3:7]
        return tf.concat([region_code_one_hot, yday_and_hour], 1)

    def visual_layers(self, x, training):
        x = self.conv1(x)
        x = self.conv1_norm(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv2_norm(x, training=training)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.conv3_norm(x, training=training)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.conv4_norm(x, training=training)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.conv5_norm(x, training=training)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.conv6_norm(x, training=training)
        x = self.relu(x)
        x = self.flatten(x)
        return x

    def profile_layers(self, x, training):
        x = self.fc1(x)
        x = self.fc1_norm(x, training=training)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc2_norm(x, training=training)
        x = self.relu(x)
        x = self.profile(x)
        return x

    def call(self, image, feature, training=False):
        processed_image = self.image_preprocessing(image, training)
        visual_feature = self.visual_layers(processed_image, training)
        auxiliary_feature = self.auxiliary_feature(feature)
        combine_feature = tf.concat([visual_feature, auxiliary_feature], 1)
        pred_profile = self.profile_layers(combine_feature, training)
        return pred_profile
