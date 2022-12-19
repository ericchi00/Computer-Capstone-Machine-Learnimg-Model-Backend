import tensorflow as tf
from keras import Sequential, layers


def create_model():
    batch_size = 128
    img_height = 300
    img_width = 300

    data_dir = "./Cat Dataset"

    # remove all files that are not accepted by tensorflow
    # image_extensions = [".jpg", ".jpeg"]
    #
    # img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    # for filepath in pathlib.Path(data_dir).rglob("*"):
    #     if filepath.suffix.lower() in image_extensions:
    #         img_type = imghdr.what(filepath)
    #         if img_type is None:
    #             os.remove(filepath)
    #         elif img_type not in img_type_accepted_by_tf:
    #             os.remove(filepath)

    # training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(directory=data_dir, validation_split=0.2,
                                                           subset="training",
                                                           seed=12345, image_size=(img_height, img_width),
                                                           batch_size=batch_size)
    # validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(directory=data_dir, validation_split=0.2,
                                                         subset="validation", seed=54321,
                                                         image_size=(img_height, img_width),
                                                         batch_size=batch_size)

    class_names = train_ds.class_names

    # configures the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # data augmentation to increase accuracy of model
    data_augmentation = Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    num_classes = len(class_names)
    # keras sequential machine learning model
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(num_classes, name="outputs", activation='softmax'),
    ])

    # compiles and trains the model for accuracy
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[callback])

    model.save('./Model')


if __name__ == "__main__":
    create_model()
