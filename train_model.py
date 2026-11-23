import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#超参数
BATCH_SIZE = 64
EPOCHS = 30
VALIDATION_SPLIT = 0.1
INPUT_SHAPE = (28, 28, 1)

#加载数据
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

#预处理
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32")
test_images = test_images.reshape(-1, 28, 28, 1).astype("float32")
train_images /= 255.0
test_images /= 255.0

#数据增强
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode="nearest"
)
datagen.fit(train_images)

#构建CNN模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=INPUT_SHAPE),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(10)
])

#编译模型
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

#训练回调
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    #保存最优模型到 fashion_mnist_model.h5
    tf.keras.callbacks.ModelCheckpoint(
        filepath="fashion_mnist_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

#训练
print("\n开始训练...")
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE, shuffle=True),
    epochs=EPOCHS,
    validation_data=(test_images, test_labels),
    callbacks=callbacks,
    verbose=1
)

#评估最终保存的最优模型
best_model = models.load_model("fashion_mnist_model.h5")
test_loss, test_acc = best_model.evaluate(test_images, test_labels, verbose=1)

print(f"\n===== 测试结果 =====")
print(f"测试准确率: {test_acc:.4f}")
print(f"模型已保存为: fashion_mnist_model.h5")