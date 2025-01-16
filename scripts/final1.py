print("Dataset Exploration, Preprocessing and Visualization\n")
import os # to handle file operations
import cv2 # to load images
import pandas as pd 
from PIL import Image #visualization
import matplotlib.pyplot as plt #visualization

#define base path where the dataset exists to load images, lidar and GPS IMU data
base_path = r"D:\Autonomous_project\KITTI_dataset\raw_dataset\2011_09_26\2011_09_26_drive_0001_sync"

# define subfolders to load images from left right camera images for grayscale (img_00,img_01) and RGB (img_02, img_03)
folders = ["image_00", "image_01", "image_02", "image_03"]

# initialize a list to store data information such as images, timestamps and type of image
data_info = []

# initialize a dictionary to store the images
images = {}

# iterate through each folder and load images and timestamps
for folder in folders:
    folder_path = os.path.join(base_path, folder, "data")
    timestamps_file = os.path.join(base_path, folder, "timestamps.txt")
 
    # Check if the folder exists
    if os.path.exists(folder_path):
        images[folder] = []  # initialize a list for each folder

        # Load timestamps
        with open(timestamps_file, 'r') as f:
            timestamps = f.read().strip().split('\n')

        # Load all the images gray scale and RBG
        for idx, img_name in enumerate(sorted(os.listdir(folder_path))):
            img_path = os.path.join(folder_path, img_name)

            # Load image using OpenCV 
            img = cv2.imread(img_path)

            # Convert BGR to RGB for visualization
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image to 224x224 for consistency
            img_resized = cv2.resize(img_rgb, (224, 224))

            # Append image and metadata
            images[folder].append(img_resized)
            data_info.append({
                "folder": folder,
                "timestamp": timestamps[idx] if idx < len(timestamps) else None,
                "camera_type": "grayscale" if "image_00" in folder or "image_01" in folder else "color",
                "image_name": img_name,
                "image_path": img_path
            })

# Convert data_info to a Pandas DataFrame
data_df = pd.DataFrame(data_info)

# Display first few 5 of the DataFrame
print(data_df.head())

# Display a few images from each folder
plt.figure(figsize=(12, 8))
for i, folder in enumerate(folders):
    if folder in images and len(images[folder]) > 0:
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[folder][0])  # Show the first image from the folder to check if the images are laoded and processed properly
        plt.title(f"{folder}")
        plt.axis("off")
plt.tight_layout()
plt.show()


print("Define Object Based Labels as given in the KITTI Website for this Raw Dataset")
# Define object-based labels manually from the information provided in the Kitti website fot raw data
object_labels = {
    "image_00": {"cars": 12, "vans": 0, "trucks": 0, "pedestrians": 0, "cyclists": 2, "trams": 1, "misc": 0},
    "image_01": {"cars": 12, "vans": 0, "trucks": 0, "pedestrians": 0, "cyclists": 2, "trams": 1, "misc": 0},
    "image_02": {"cars": 12, "vans": 0, "trucks": 0, "pedestrians": 0, "cyclists": 2, "trams": 1, "misc": 0},
    "image_03": {"cars": 12, "vans": 0, "trucks": 0, "pedestrians": 0, "cyclists": 2, "trams": 1, "misc": 0},
}

# Add these labels to the pandas DataFrame
data_df['object_labels'] = data_df['folder'].map(object_labels)

# Display the updated DataFrame with labels added
print(data_df[['image_name', 'folder', 'object_labels']].head())


print("Visualiza Labelled Data")
plt.figure(figsize=(15, 8))
for i, folder in enumerate(folders):
    if folder in images and len(images[folder]) > 0:
        plt.subplot(4, 1, i + 1)
        plt.imshow(images[folder][0])  # Show the first image from the folder
        plt.title(f"{folder}: {object_labels[folder]}")
        plt.axis("off")
plt.tight_layout()
plt.show()


print("Convert images and labels to numpy array and split the dataset into training, testing and validation")
import os #file operations
import numpy as np #handling numerical data for this dataset converting numerical data to arrays
import tensorflow as tf #building and training cnn models
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input #import the layers for neural network
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping #import callbacks for stopping the model if the accuracy does not improve or the loss does not reduce 
from sklearn.model_selection import train_test_split #standard split the raw data into training and testing data
import matplotlib.pyplot as plt #visualization

# Load and preprocess images
#define base path and image folder paths 
base_path = r"D:\Autonomous_project\KITTI_dataset\raw_dataset\2011_09_26\2011_09_26_drive_0001_sync"
folders = ["image_00", "image_01", "image_02", "image_03"]
image_size = (224, 224) #resizing the images (standard size for CNN models)

X = []  # define a empty list for images
y = []  # define a empty list for labels

#define label map as given in the KITTI dataset overview for raw data
label_map = {
    "Car": 0,
    "Cyclist": 1,
    "Tram": 2,
}  #map the object classes to neumerical data

for folder in folders:
    folder_path = os.path.join(base_path, folder, "data")
    for img_name in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_name)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize and convert image to array
        X.append(img_array)

        # Assign a label based on the folder 
        if "image_00" in folder:
            y.append(label_map["Car"])
        elif "image_01" in folder:
            y.append(label_map["Cyclist"])
        elif "image_02" in folder:
            y.append(label_map["Tram"])
        else:
            y.append(label_map["Car"])  # Default

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Build a simple CNN model
model = Sequential([
    Input(shape=(224, 224, 3)),  # Explicitly define input shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Data augmentation to increase the diversity of data(include blur and noise to simulate for diverse weather conditions)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)



# Train the model FOR 50 Epochs
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=50)


# Evaluate the model
loss, accuracy_cnn = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy_cnn * 100:.2f}%")



from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_map.keys())

# Plot Confusion Matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# Visualize training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
# Load VGG16 model with pre-trained weights
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in vgg_base.layers:
    layer.trainable = False

# Add custom classification layers
model = Sequential([
    vgg_base,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)



# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=50
                    )


# Evaluate the model
loss, accuracy_vgg16 = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy_vgg16 * 100:.2f}%")

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_map.keys())

# Plot Confusion Matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# Visualize training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# Load ResNet50 model with pre-trained weights
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in resnet_base.layers:
    layer.trainable = False

# Add custom classification layers
model = Sequential([
    resnet_base,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)


# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=50
                   )


# Evaluate the model
loss, accuracy_resnet = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy_resnet * 100:.2f}%")

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_map.keys())


# Plot Confusion Matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Visualize training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

# Load DenseNet121 model with pre-trained weights
densenet_base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in densenet_base.layers:
    layer.trainable = False

# Add custom classification layers
model = Sequential([
    densenet_base,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)


# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=50)


# Evaluate the model
loss, accuracy_densenet121 = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy_densenet121 * 100:.2f}%")

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_map.keys())

# Plot Confusion Matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# Visualize training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Compare Models
model_names = ["Simple CNN", "VGG16", "ResNet", "DenseNet"]
accuracies = [accuracy_cnn, accuracy_vgg16, accuracy_resnet, accuracy_densenet121]

plt.bar(model_names, accuracies, color=['blue', 'orange', 'green', 'red'])
plt.title("Model Comparison")
plt.ylabel("Accuracy (%)")
plt.xlabel("Models")
plt.ylim(0, .99)
plt.show()




import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from matplotlib.image import imread
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from imgaug import augmenters as iaa

# Define paths to image, LIDAR, and GPS/IMU folders
image_folder = r"D:\Autonomous_project\KITTI_dataset\raw_dataset\2011_09_26\2011_09_26_drive_0001_sync\image_03\data"
lidar_folder = r"D:\Autonomous_project\KITTI_dataset\raw_dataset\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data"
oxts_folder = r"D:\Autonomous_project\KITTI_dataset\raw_dataset\2011_09_26\2011_09_26_drive_0001_sync\oxts\data"
calib_cam_to_cam_path = r"D:\Autonomous_project\KITTI_dataset\raw_dataset\2011_09_26\calib\calib_cam_to_cam.txt"
calib_velo_to_cam_path = r"D:\Autonomous_project\KITTI_dataset\raw_dataset\2011_09_26\calib\calib_velo_to_cam.txt"

# Load calibration matrices
def load_calibration_matrices():
    # Load velo_to_cam calibration
    with open(calib_velo_to_cam_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("R:"):
                R = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
            if line.startswith("T:"):
                T = np.array([float(x) for x in line.split()[1:]]).reshape(3, 1)

    # Load cam_to_cam calibration
    with open(calib_cam_to_cam_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("P_rect_02:"):
                P_rect = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)

    return R, T, P_rect

R, T, P_rect = load_calibration_matrices()

# Transform LIDAR points to the camera frame
def transform_lidar_to_camera(lidar_points):
    # Convert to homogeneous coordinates
    lidar_points_h = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    # Apply rotation and translation
    lidar_in_camera_frame = np.dot(R, lidar_points_h[:, :3].T).T + T.T
    return lidar_in_camera_frame

# Project points to the image plane
def project_to_image_plane(camera_points):
    camera_points_h = np.hstack((camera_points, np.ones((camera_points.shape[0], 1))))
    image_points = np.dot(P_rect, camera_points_h.T).T
    # Normalize by the third coordinate
    image_points[:, 0] /= image_points[:, 2]
    image_points[:, 1] /= image_points[:, 2]
    return image_points[:, :2]

def parse_lidar_file(lidar_file_path):
    """Parse LIDAR .bin file."""
    # Load binary data as a float32 numpy array
    return np.fromfile(lidar_file_path, dtype=np.float32).reshape(-1, 4)


valid_image_files = [
    f for f in os.listdir(image_folder)
    if isinstance(f, str) and f.endswith(('.png', '.jpg', '.jpeg'))
]

def visualize_lidar_overlay(image_file):
    """Visualize LIDAR points overlaid on an image."""
    # Extract the base name without extension
    base_name = os.path.splitext(image_file)[0]
    lidar_file = f"{base_name}.bin"  # Append .bin for LIDAR file

    img_path = os.path.join(image_folder, image_file)
    lidar_path = os.path.join(lidar_folder, lidar_file)

    # Debug: Check paths
    print(f"Image path: {img_path}")
    print(f"LIDAR path: {lidar_path}")

    # Check if LIDAR file exists
    if not os.path.exists(lidar_path):
        print(f"Missing LIDAR data for {image_file}")
        return

    # Read the image and LIDAR data
    img = imread(img_path)
    lidar_points = parse_lidar_file(lidar_path)

    # Transform and project LIDAR points
    lidar_in_camera_frame = transform_lidar_to_camera(lidar_points)
    image_points = project_to_image_plane(lidar_in_camera_frame)

    # Filter points that fall outside the image bounds
    img_h, img_w, _ = img.shape
    valid_indices = (
        (image_points[:, 0] >= 0) & (image_points[:, 0] < img_w) &
        (image_points[:, 1] >= 0) & (image_points[:, 1] < img_h)
    )
    image_points = image_points[valid_indices]

    # Plot the image and overlay LIDAR points
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.scatter(image_points[:, 0], image_points[:, 1], s=1, c='red', label='LIDAR Points')
    plt.title(f"LIDAR Overlay on {image_file}")
    plt.legend()
    plt.axis('off')
    plt.show()

print("Image Files:", os.listdir(image_folder))
print("LIDAR Files:", os.listdir(lidar_folder))



visualize_lidar_overlay(valid_image_files[0])
visualize_lidar_overlay(valid_image_files[23])
visualize_lidar_overlay(valid_image_files[100])


# Data augmentation
augmentation_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-10, 10)),
    iaa.Multiply((0.8, 1.2)),
    iaa.GaussianBlur(sigma=(0, 1.0))
])

def augment_images(image_files):
    augmented_images = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented_img = augmentation_pipeline(image=img)
        augmented_images.append(augmented_img)
    return augmented_images

augmented_images = augment_images(valid_image_files)


# Split data into training, validation, and test sets
train_files, test_files = train_test_split(valid_image_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)

print(f"Training set: {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")
print(f"Test set: {len(test_files)} images")

# KITTI Environment
class KITTIEnvironment:
    def __init__(self, image_folder, lidar_folder, valid_image_files):
        self.image_folder = image_folder
        self.lidar_folder = lidar_folder
        self.image_files = valid_image_files
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self._get_state(self.current_index)

    def step(self, action):
        reward = random.uniform(0, 1)  # Placeholder reward logic
        done = self.current_index >= len(self.image_files) - 1
        self.current_index += 1
        next_state = self._get_state(self.current_index) if not done else None
        return next_state, reward, done, {}

    def _get_state(self, index):
        img_path = os.path.join(self.image_folder, self.image_files[index])
        lidar_path = os.path.join(self.lidar_folder, os.path.splitext(self.image_files[index])[0] + ".bin")

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (84, 84)) / 255.0

        # Load LIDAR points
        lidar_points = parse_lidar_file(lidar_path)

        # Transform and project LIDAR points
        lidar_in_camera_frame = transform_lidar_to_camera(lidar_points)
        image_points = project_to_image_plane(lidar_in_camera_frame)

        return {
            "image": img,
            "lidar": image_points
        }

# Initialize environment
env = KITTIEnvironment(image_folder, lidar_folder, train_files)

# DQN Model
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize DQN Model
action_size = 3  # left, right, forward
policy_net = DQN(action_size)
target_net = DQN(action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Define training parameters
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
learning_rate = 1e-4
target_update = 10
num_episodes = 100

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
replay_buffer = deque(maxlen=10000)

# Training Loop
def train():
    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).permute(0, 3, 1, 2)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).permute(0, 3, 1, 2)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = loss_fn(q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main Training Loop
rewards_per_episode = []
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    for t in range(200):  # Max steps per episode
        if random.random() < epsilon:
            action = random.randint(0, action_size - 1)  # Explore
        else:
            state_tensor = torch.tensor(state["image"], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
            q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()  # Exploit

        next_state, reward, done, _ = env.step(action)

        # Skip adding to replay buffer if next_state is None
        if next_state is not None:
            replay_buffer.append((state["image"], action, reward, next_state["image"], done))

        state = next_state
        total_reward += reward

        # Train the model
        train()

        if done:
            break

    rewards_per_episode.append(total_reward)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")

# Visualize Training Rewards
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Rewards")
plt.show()

# Save the trained model
torch.save(policy_net.state_dict(), "dqn_kitti_model.pth")


# Evaluate the model
def evaluate_model(env, policy_net, num_episodes=20):
    policy_net.eval()  # Set the model to evaluation mode
    total_rewards = []
    success_count = 0  # Count successful episodes

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state["image"], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()  # Select the best action

            next_state, reward, done, _ = env.step(action)
            if next_state is not None:
                state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)
        if episode_reward > np.mean(total_rewards):  # Example threshold for success
            success_count += 1

    avg_reward = np.mean(total_rewards)
    success_rate = success_count / num_episodes * 100

    print(f"Evaluation Results over {num_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")

    return avg_reward, success_rate


# Visualize agent decisions
def visualize_agent_decisions(env, policy_net, num_images=5):
    policy_net.eval()
    sampled_images = random.sample(env.image_files, num_images)

    for img_file in sampled_images:
        state = env._get_state(env.image_files.index(img_file))  # Load state

        # Predict the action
        state_tensor = torch.tensor(state["image"], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        action = torch.argmax(q_values).item()

        # Visualize the action
        plt.figure(figsize=(10, 6))
        plt.imshow(state["image"])
        plt.title(f"Action Taken: {['Left', 'Straight', 'Right'][action]}")
        plt.axis('off')
        plt.show()

# Visualize decisions
visualize_agent_decisions(env, policy_net)


import nbformat

# Replace with your notebook file path
notebook_file = r'D:\Autonomous_project\KITTI_dataset\scripts\22091524-ASHWIN-VARDHARAJAN.ipynb'

# Load the notebook
with open(notebook_file, 'r', encoding='utf-8') as file:
    notebook = nbformat.read(file, as_version=4)

# Extract all code cells
code_cells = [cell['source'] for cell in notebook.cells if cell.cell_type == 'code']

# Save to a .py file or print
with open('extracted_code.py', 'w') as f:
    for code in code_cells:
        f.write(code + '\n\n')

print("Code cells extracted and saved to extracted_code.py")


