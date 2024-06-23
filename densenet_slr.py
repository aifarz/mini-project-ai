import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Enable memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Function to calculate mAP
def mean_average_precision(y_true, y_pred, num_classes):
    average_precisions = []
    for class_id in range(num_classes):
        true_positive = np.sum((y_true == class_id) & (y_pred == class_id))
        false_positive = np.sum((y_true != class_id) & (y_pred == class_id))
        false_negative = np.sum((y_true == class_id) & (y_pred != class_id))
        precision = true_positive / (true_positive + false_positive + 1e-6)
        recall = true_positive / (true_positive + false_negative + 1e-6)
        average_precision = (precision * recall) / (precision + recall + 1e-6)
        average_precisions.append(average_precision)
    return np.mean(average_precisions)

# Define the create_datasets function
def create_datasets(base_dir, target_size=(224, 224), batch_size=16):  # Reduce batch size to 16
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = validation_generator = test_generator = None

    if os.path.exists(os.path.join(base_dir, 'train')):
        train_generator = train_datagen.flow_from_directory(os.path.join(base_dir, 'train'), target_size=target_size,
                                                            batch_size=batch_size, class_mode='categorical')
        print(f"Found {train_generator.samples} images in train directory.")
    else:
        print('Train directory not found at ' + os.path.join(base_dir, 'train'))

    if os.path.exists(os.path.join(base_dir, 'val')):
        validation_generator = train_datagen.flow_from_directory(os.path.join(base_dir, 'val'), target_size=target_size,
                                                                 batch_size=batch_size, class_mode='categorical')
        print(f"Found {validation_generator.samples} images in validation directory.")
    else:
        print('Validation directory not found at ' + os.path.join(base_dir, 'val'))

    if os.path.exists(os.path.join(base_dir, 'test')):
        test_generator = test_datagen.flow_from_directory(os.path.join(base_dir, 'test'), target_size=target_size,
                                                          batch_size=batch_size, class_mode='categorical',
                                                          shuffle=False)
        print(f"Found {test_generator.samples} images in test directory.")
    else:
        print('Test directory not found at ' + os.path.join(base_dir, 'test'))

    return train_generator, validation_generator, test_generator

def create_model(base_model, num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def plot_confusion_matrix(model_name, epoch, y_true, y_pred_classes, class_names):
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(12, 10))  # Increase figure size for better visibility
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='horizontal')  # Set rotation to horizontal for xticks

    # Increase font sizes for better visibility
    plt.title(f'{model_name} (Epoch {epoch}) - Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted label', fontsize=16)
    plt.ylabel('True label', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Annotate each cell in the confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i, j]}', horizontalalignment='center', verticalalignment='center', color='black',
                     fontsize=16)

    plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping
    plt.savefig(f'{model_name}_epoch_{epoch}_confusion_matrix.png')
    plt.close()

# Define the learning rate schedule function
def lr_schedule(epoch, lr):
    if epoch < 20:
        return 0.01
    elif epoch < 40:
        return 0.001
    else:
        return 0.0001

# Load datasets
train_gen, val_gen, test_gen = create_datasets('Processed_Data')

num_classes = len(train_gen.class_indices)
model_name = 'DenseNet121'
base_model = DenseNet121(weights='imagenet', include_top=False)

tf.keras.backend.clear_session()  # Clear memory before training a new model
print(f"Training {model_name} with scheduled learning rate")

model = create_model(base_model, num_classes)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

start_time = time.time()
history = model.fit(train_gen, epochs=60, validation_data=val_gen, callbacks=[lr_scheduler])  # Train for 60 epochs
training_time = time.time() - start_time

# Save model and training history
trained_model = model
history_data = history

# Evaluate model
loss, accuracy = model.evaluate(test_gen)
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes
mAP = mean_average_precision(y_true, y_pred_classes, num_classes)

print(
    f"{model_name} with scheduled learning rate - Loss: {loss}, Accuracy: {accuracy}, mAP: {mAP}, Training Time: {training_time}s")

# Plot training history
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title(f'{model_name} - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{model_name}_loss.png')

plt.figure()
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title(f'{model_name} - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'{model_name}_accuracy.png')

# Confusion matrix
plot_confusion_matrix(model_name, 60, y_true, y_pred_classes, list(test_gen.class_indices.keys()))
