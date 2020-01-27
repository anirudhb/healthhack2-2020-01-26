import tensorflow as tf
import numpy as np
import datetime
import os

from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.mixed_precision import experimental as mixed_precision


BATCH_SIZE = 128
LEARNING_RATE = 0.001
TRAIN_STEPS = 100
VAL_STEPS = 10
NUM_EPOCHS = 50
total_classes = ["anger", "contempt", "fear", "happiness", "neutral", "sadness", "surprise", "disgust"]

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

def load_image(image_file): 
    img_raw = tf.io.read_file("./lfw/{}".format(image_file[0].numpy().decode("utf8"))) 
    img = tf.io.decode_image(img_raw, dtype=tf.float32) 
    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    return img

def get_onehot(emotion): 
    index = total_classes.index(emotion[0].numpy().decode("utf8").lower()) 
    one_hot = tf.one_hot(index, len(total_classes))
    return one_hot

def tf_load_data(image_file, emotion): 
    image_shape = (250, 250, 3)
    label_shape = (len(total_classes),) 
    [image,] = tf.py_function(load_image, [image_file], [tf.float32]) 
    image.set_shape(image_shape) 
    [label,] = tf.py_function(get_onehot, [emotion], [tf.float32]) 
    label.set_shape(label_shape) 
    return image, label

def file_exists(image_file):
    exists = os.path.isfile("./lfw/{}".format(image_file[0].numpy().decode("utf8")))
    return exists

def tf_file_exists(image_file, label):
    [exists,] = tf.py_function(file_exists, [image_file], [tf.bool]) 
    return exists

def is_emotion(emotion, match_emotion):
    return emotion[0].numpy().decode("utf8").lower() == match_emotion

def tf_is_emotion(image_file, emotion, match_emotion):
    [match,] = tf.py_function(lambda x: is_emotion(x, match_emotion), [emotion], [tf.bool]) 
    return match

def get_backbone_model():
    inception_resnet = inception_resnet_v2.InceptionResNetV2(
        include_top=False, weights="imagenet", input_shape=(250, 250, 3)
    )
    output = tf.keras.layers.GlobalAveragePooling2D()(inception_resnet.output)
    base_model = tf.keras.models.Model(inception_resnet.input, output)
    return base_model

def get_classification_model(base_model):
    input_image = tf.keras.layers.Input((250, 250, 3), name="face")
    feature_map = base_model(input_image)
    x = tf.keras.layers.Dense(512)(feature_map)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(len(total_classes), activation=tf.nn.softmax)(x)
    model = tf.keras.models.Model(
        inputs=[input_image], outputs=x
    )
    return model
if __name__ == "__main__":
    dataset = tf.data.experimental.make_csv_dataset("./facial_expressions/data/legend.csv", 1)
    dataset = dataset.map(lambda inp: (inp["image"], inp["emotion"]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(tf_file_exists)
    # split_ds_list = []
    # for emotion in total_classes:
    #     ds = dataset.filter(lambda x, y: tf_is_emotion(x, y, emotion))
    #     split_ds_list.append(ds)
    # dataset = tf.data.experimental.sample_from_datasets(split_ds_list)
    dataset = dataset.map(tf_load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = dataset.take(10000)
    train_dataset = train_dataset.shuffle(5000).repeat()
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.skip(10000).repeat()
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        base_model = get_backbone_model()
        model = get_classification_model(base_model)

        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "./model.h5", verbose=1
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=TRAIN_STEPS,
        validation_steps=VAL_STEPS,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint, tensorboard_callback],
    )