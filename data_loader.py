import os
import shutil
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import PIL

# Setup logging configuration
logging.basicConfig(filename='data_processing.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Define class names
class_names = ['American Black Bear', 'Asian Black Bear', 'Grizzly Bear', 'Polar Bear', 'Sloth Bear']


# Function to verify images and remove corrupt files
def verify_images(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                img = load_img(filepath)
                logging.info(f'Loaded image {filepath} successfully.')
            except (IOError, SyntaxError, PIL.UnidentifiedImageError) as e:
                logging.error(f'Bad file {filepath}: {e}')
                os.remove(filepath)


# Function to create the directory structure
def create_directory_structure(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        for class_name in class_names:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)


# Function to split data into training, validation, and testing sets and copy files
def split_and_copy_data(source_folder, destination_folder, train_size=0.6, val_size=0.2, test_size=0.2):
    for class_name in class_names:
        class_files = []
        class_folder = os.path.join(source_folder, class_name)
        for file in os.listdir(class_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                class_files.append(os.path.join(class_folder, file))

        logging.info(f'Found {len(class_files)} image files in {class_folder}.')

        if len(class_files) == 0:
            logging.error(f'No image files found in class {class_name}. Check the source directory path and contents.')
            continue

        train_files, test_files = train_test_split(class_files, train_size=train_size + val_size, test_size=test_size,
                                                   random_state=42)
        train_files, val_files = train_test_split(train_files, train_size=train_size / (train_size + val_size),
                                                  test_size=val_size / (train_size + val_size), random_state=42)

        def copy_files(files, split_name):
            for file in files:
                destination_path = os.path.join(destination_folder, split_name, class_name)
                shutil.copy(file, destination_path)
            logging.info(f'{len(files)} files copied to {split_name}/{class_name}')

        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')

    logging.info('Data successfully split and copied to the destination directory.')


# Function to create data generators for the datasets
def create_datasets(base_dir, target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = validation_generator = test_generator = None

    if os.path.exists(os.path.join(base_dir, 'train')):
        train_generator = train_datagen.flow_from_directory(os.path.join(base_dir, 'train'), target_size=target_size,
                                                            batch_size=batch_size, class_mode='categorical')
        logging.info(f"Found {train_generator.samples} images in train directory.")
    else:
        logging.error('Train directory not found at ' + os.path.join(base_dir, 'train'))

    if os.path.exists(os.path.join(base_dir, 'val')):
        validation_generator = train_datagen.flow_from_directory(os.path.join(base_dir, 'val'), target_size=target_size,
                                                                 batch_size=batch_size, class_mode='categorical')
        logging.info(f"Found {validation_generator.samples} images in validation directory.")
    else:
        logging.error('Validation directory not found at ' + os.path.join(base_dir, 'val'))

    if os.path.exists(os.path.join(base_dir, 'test')):
        test_generator = test_datagen.flow_from_directory(os.path.join(base_dir, 'test'), target_size=target_size,
                                                          batch_size=batch_size, class_mode='categorical')
        logging.info(f"Found {test_generator.samples} images in test directory.")
    else:
        logging.error('Test directory not found at ' + os.path.join(base_dir, 'test'))

    return train_generator, validation_generator, test_generator


# Function to log the contents of directories
def list_directory_contents(directory):
    for root, dirs, files in os.walk(directory):
        logging.info(f'Listing {root}')
        for name in files:
            logging.info(f'File: {name}')
        for name in dirs:
            logging.info(f'Directory: {name}')


# Main execution block
if __name__ == "__main__":
    base_dir = 'Data'  # Path to the dataset
    destination_dir = 'Processed_Data'  # Path to save processed dataset
    logging.info('Starting image verification.')
    verify_images(base_dir)
    logging.info('Image verification completed. Creating directory structure.')
    create_directory_structure(destination_dir)
    logging.info('Directory structure created. Starting data splitting and copying.')
    split_and_copy_data(base_dir, destination_dir)
    logging.info('Data splitting and copying completed. Listing contents of processed data directory.')
    list_directory_contents(destination_dir)  # List contents of the processed directory
    logging.info('Setting up data generators.')
    train_gen, val_gen, test_gen = create_datasets(destination_dir)
    if train_gen and val_gen and test_gen:
        logging.info('Data generators set up successfully.')
    else:
        logging.info('One or more data generators could not be set up due to missing directories.')
