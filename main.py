!pip install patchify
from patchify import patchify
import numpy as np
from tensorflow import keras
import zipfile
import nibabel as nib
import os
from scipy import ndimage
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras import layers


# Download url of normal CT scans.
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
filename = os.path.join(os.getcwd(), "CT-0.zip")
keras.utils.get_file(filename, url)

# Download url of abnormal CT scans.
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
filename = os.path.join(os.getcwd(), "CT-23.zip")
keras.utils.get_file(filename, url)

# Make a directory to store the data.
if not os.path.exists("MosMedData"):
  
  os.makedirs("MosMedData")

# Unzip data in the newly created directory.
with zipfile.ZipFile("CT-0.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")

with zipfile.ZipFile("CT-23.zip", "r") as z_fp:
    z_fp.extractall("./MosMedData/")

############################################################ for quick trial s
norm = os.listdir('/content/MosMedData/CT-0')
patho = os.listdir('/content/MosMedData/CT-23')

for i in norm[:90]:
  os.remove(os.path.join('/content/MosMedData/CT-0',i))

for i in patho[:90]:
  os.remove(os.path.join('/content/MosMedData/CT-23',i))

norm = os.listdir('/content/MosMedData/CT-0')
patho = os.listdir('/content/MosMedData/CT-23')


########################################################


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
 
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 32
    desired_width = 64
    desired_height = 64



    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def create_patches(img):
    return patchify(img,(32,32,8))
  


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    volume = create_patches(volume)
    return volume

normal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-0", x)
    for x in os.listdir("MosMedData/CT-0")
]
# Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# involvement of lung parenchyma.
abnormal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData/CT-23", x)
    for x in os.listdir("MosMedData/CT-23")
]


# Each scan is resized across height, width, and depth and rescaled.
abnormal_scans =([process_scan(path) for path in abnormal_scan_paths])
normal_scans = ([process_scan(path) for path in normal_scan_paths])

# # For the CT scans having presence of viral pneumonia
# # assign 1, for the normal ones assign 0.
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])


# # Split data in the ratio 70-30 for training and validation.
norm_len= len(norm)
patho_len =len(patho)


x_train = np.concatenate((abnormal_scans[0][:int(0.2*norm_len)], normal_scans[0][:int(0.2*norm_len)]), axis=0)
y_train = np.concatenate((abnormal_labels[:int(0.2*norm_len)], normal_labels[:int(0.2*norm_len)]), axis=0) 

x_val = np.concatenate((abnormal_scans[0][int(0.2*norm_len):int(0.3*norm_len)], normal_scans[0][int(0.2*norm_len):int(0.3*norm_len)]), axis=0)
y_val = np.concatenate((abnormal_labels[int(0.2*norm_len):int(0.3*norm_len)], normal_labels[int(0.2*norm_len):int(0.3*norm_len)]), axis=0)


x_train= np.reshape(x_train, (-1, x_train.shape[-3], x_train.shape[-2], x_train.shape[-1]))
y_train = np.concatenate( ([1 for _ in range(int(len(x_train)/2))]  , [0 for _ in range(int(len(x_train)/2))] ),axis=0)

x_val= np.reshape(x_val, (-1, x_val.shape[-3], x_val.shape[-2], x_val.shape[-1]))
y_val = np.concatenate( ( [1 for _ in range(int(len(x_val)/2))] , [0 for _ in range(int(len(x_val)/2))] ),axis=0)

unlabelled = np.concatenate((abnormal_scans[0][int(0.3*norm_len)::], normal_scans[0][int(0.3*norm_len)::]), axis=0) 
unlabelled= np.reshape(unlabelled, (-1, unlabelled.shape[-3], unlabelled.shape[-2], unlabelled.shape[-1]))

no_labels = np.array([2 for _ in range(len(unlabelled))])

print(x_train.shape,y_train.shape)

train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
unlabelled_loader =tf.data.Dataset.from_tensor_slices((unlabelled,no_labels))

batch_size = 8
# Augment the on the fly during training.
labeled_train_dataset = (
    train_loader.shuffle(len(x_train))
    .batch(batch_size)
    # .prefetch(2)
)
# Only rescale.
test_dataset = (
    validation_loader.shuffle(len(x_val))
    .batch(batch_size)
    #.prefetch(2)
)

unlabeled_train_dataset = (
    unlabelled_loader.shuffle(len(x_val))
    .batch(batch_size)
    # .prefetch(2) 
)


def prepare_dataset():
    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    )#.prefetch(buffer_size=tf.data.AUTOTUNE)
   
    return train_dataset, labeled_train_dataset, test_dataset

train_dataset, labeled_train_dataset, test_dataset = prepare_dataset()


unlabeled_dataset_size = unlabelled.shape[0]
labeled_dataset_size = x_train.shape[0]+ x_val.shape[0]
image_size = 32
image_channels = 8
# Algorithm hyperparameters
num_epochs = 5
batch_size = 8  
width = 32
height=32
depth=8
temperature = 0.1
# Stronger augmentations for contrastive, weaker ones for supervised training
contrastive_augmentation = {"min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
classification_augmentation = {"min_area": 0.75, "brightness": 0.3, "jitter": 0.1}

# Distorts the color distibutions of images
class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def get_config(self):
        config = super().get_config()
        config.update({"brightness": self.brightness, "jitter": self.jitter})
        return config

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )
            # Different for all colors
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 8, 8), minval=-self.jitter, maxval=self.jitter
            )

            color_transforms = (
                tf.eye(8, batch_shape=[batch_size, 1]) * brightness_scales
                + jitter_matrices
            )
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images


# Image augmentation module
def get_augmenter(min_area, brightness, jitter):
    zoom_factor = 1.0 - math.sqrt(min_area)
    return keras.Sequential(
        [
            keras.Input(shape=(image_size, image_size, image_channels)),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            RandomColorAffine(brightness, jitter),
         
        ]
    )

def get_encoder():
    return keras.Sequential(
        [
            keras.Input(shape=(width, height, depth,1)),
            layers.Conv3D(filters=64, kernel_size=3, activation="relu"),
            layers.MaxPool3D(pool_size=2),
            layers.BatchNormalization(),
            layers.Conv3D(filters=64, kernel_size=3, activation="relu"),
            layers.MaxPool3D(pool_size=1),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(width, activation="relu"),
         
        ],
        name="encoder",
    )

baseline_model = keras.Sequential(
    [
        keras.Input(shape=(width, height, depth,1)),
        get_augmenter(**classification_augmentation),
        get_encoder(),
        layers.Dense(2),
    ],
    name="baseline_model",
)
baseline_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# baseline_history = baseline_model.fit(
#     labeled_train_dataset, epochs=3, validation_data=test_dataset
# )

# print(
#     "Maximal validation accuracy: {:.2f}%".format(
#         max(baseline_history.history["val_acc"]) * 100
#     )
# )
 
######################### 

class ContrastiveModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.temperature = temperature
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.classification_augmenter = get_augmenter(**classification_augmentation)
        self.encoder = get_encoder()
        # Non-linear MLP as projection head
        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(width,)),
                layers.Dense(width, activation="relu"),
                layers.Dense(width),
            ],
            name="projection_head",
        )
        # Single dense layer for linear probing
        self.linear_probe = keras.Sequential(
            [layers.Input(shape=(width,)), layers.Dense(2)], name="linear_probe"
        )

        self.encoder.summary()
        self.projection_head.summary()
        self.linear_probe.summary()

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        
        (unlabeled_images, _), (labeled_images, labels) = data
        
        # Both labeled and unlabeled images are used, without labels
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        
        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        # Labels are only used in evalutation for an on-the-fly logistic regression
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=True
        )
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            features = self.encoder(preprocessed_images, training=False)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_images, labels = data

        # For testing the components are used with a training=False flag
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}


# Contrastive pretraining
pretraining_model = ContrastiveModel()

pretraining_model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
)

pretraining_history = pretraining_model.fit(
    train_dataset, epochs=num_epochs, validation_data=test_dataset
)
print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(pretraining_history.history["val_p_acc"]) * 100
    )
)


finetuning_model = keras.Sequential(
    [
        layers.Input(shape=(image_size, image_size, image_channels)),
        get_augmenter(**classification_augmentation),
        pretraining_model.encoder,
        layers.Dense(2),
    ],
    name="finetuning_model",
)
finetuning_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

finetuning_history = finetuning_model.fit(
    labeled_train_dataset, epochs=num_epochs, validation_data=test_dataset
)
print(
    "Maximal validation accuracy: {:.2f}%".format(
        max(finetuning_history.history["val_acc"]) * 100
    )
)

finetuning_model.save('semi_supervised.h5')
# finetuning_model.save_weights('semi_supervised_weights.h5')
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(finetuning_history.history['acc'])
plt.plot(finetuning_history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(122)
plt.plot(finetuning_history.history['loss'])
plt.plot(finetuning_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('/content/drive/MyDrive/Loss_ultra_sound.png')
plt.show()

################################ predictions
# for predicttions
# finetuning_model.predict(np.expand_dims(x_val[0], axis=0))[0]


scan =process_scan('/content/MosMedData/CT-0/study_0004.nii.gz')

scan = np.reshape(scan , (-1, x_train.shape[-3], x_train.shape[-2], x_train.shape[-1]))
print(scan.shape)
scan = tf.data.Dataset.from_tensor_slices((scan))

scan = (
    scan.shuffle(len(x_train))
    .batch(batch_size))
  
predictions = (finetuning_model.predict(scan))

preds = np.argmax(predictions,axis =1)
zero_class= preds[preds==0]
one_class =preds[preds==1]

len_one_class = one_class.shape[0]
len_zero_class = zero_class.shape[0]

if len_one_class> len_zero_class:
  print("ONE CLASS",len_one_class,len_zero_class)
else:
  print("ZERO CLASS",len_zero_class,len_one_class)
