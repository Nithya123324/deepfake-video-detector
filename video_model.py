import tensorflow as tf

def build_video_deepfake_model(frames_per_video: int = 60, image_size: tuple = (224, 224, 3)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(frames_per_video, *image_size), name="video_input")
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=image_size, pooling=None)
    frame_features = tf.keras.layers.TimeDistributed(backbone, name="efficientnet_timedistributed")(inputs)
    frame_features = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(), name="frame_global_pool")(frame_features)
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64, name="video_attention")(frame_features, frame_features)
    x = tf.keras.layers.Add(name="residual_attention")([frame_features, attention_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="sequence_norm")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="temporal_pool")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="deepfake_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="video_deepfake_model")
    return model

def compile_model(model: tf.keras.Model, learning_rate: float = 1e-4):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="roc_auc")])
    return model
