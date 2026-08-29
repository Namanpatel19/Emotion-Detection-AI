"""
model.py
────────
Defines the CNN architecture used for facial emotion recognition.

Architecture overview
─────────────────────
Input  : (48, 48, 1)  — grayscale face crops, normalised to [0, 1]
Output : (6,)         — softmax probabilities for 6 emotion classes

Block structure (each block = Conv → BatchNorm → ReLU → MaxPool → Dropout):
  Block 1:  64 filters  (3×3)
  Block 2: 128 filters  (3×3)
  Block 3: 256 filters  (3×3)
  Block 4: 512 filters  (3×3)
Classifier head: Flatten → Dense(256) → Dense(128) → Dense(6, softmax)

Design choices:
  • BatchNormalization after every Conv2D stabilises gradients and speeds training.
  • Dropout (0.25 after pooling, 0.5 in dense layers) prevents overfitting.
  • L2 regularisation on Dense layers adds another layer of regularisation.
"""

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout, Activation
)
from tensorflow.keras.regularizers import l2

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 48     # image height and width (pixels)
NUM_CLASSES = 6      # Angry, Fear, Happy, Sad, Surprise, Neutral


# ── CNN Builder ───────────────────────────────────────────────────────────────

def build_emotion_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1),
                      num_classes=NUM_CLASSES,
                      l2_reg=1e-4) -> Model:
    """
    Construct and compile the emotion-recognition CNN.

    Parameters
    ----------
    input_shape : tuple  — (height, width, channels), default (48, 48, 1)
    num_classes : int    — number of output classes, default 6
    l2_reg      : float  — L2 regularisation strength for dense layers

    Returns
    -------
    model : compiled tf.keras.Model ready for training
    """

    inputs = Input(shape=input_shape, name='face_input')

    # ── Convolutional Block 1 — 64 filters ─────────────────────────────────
    x = Conv2D(64, (3, 3), padding='same', name='conv1_1')(inputs)
    x = BatchNormalization(name='bn1_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='conv1_2')(x)
    x = BatchNormalization(name='bn1_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = Dropout(0.25, name='drop1')(x)
    # Output: (24, 24, 64)

    # ── Convolutional Block 2 — 128 filters ────────────────────────────────
    x = Conv2D(128, (3, 3), padding='same', name='conv2_1')(x)
    x = BatchNormalization(name='bn2_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='conv2_2')(x)
    x = BatchNormalization(name='bn2_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    x = Dropout(0.25, name='drop2')(x)
    # Output: (12, 12, 128)

    # ── Convolutional Block 3 — 256 filters ────────────────────────────────
    x = Conv2D(256, (3, 3), padding='same', name='conv3_1')(x)
    x = BatchNormalization(name='bn3_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_2')(x)
    x = BatchNormalization(name='bn3_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    x = Dropout(0.25, name='drop3')(x)
    # Output: (6, 6, 256)

    # ── Convolutional Block 4 — 512 filters ────────────────────────────────
    x = Conv2D(512, (3, 3), padding='same', name='conv4_1')(x)
    x = BatchNormalization(name='bn4_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv4_2')(x)
    x = BatchNormalization(name='bn4_2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
    x = Dropout(0.25, name='drop4')(x)
    # Output: (3, 3, 512)

    # ── Classifier Head ─────────────────────────────────────────────────────
    x = Flatten(name='flatten')(x)

    x = Dense(256, activation='relu',
              kernel_regularizer=l2(l2_reg), name='fc1')(x)
    x = BatchNormalization(name='bn_fc1')(x)
    x = Dropout(0.5, name='drop_fc1')(x)

    x = Dense(128, activation='relu',
              kernel_regularizer=l2(l2_reg), name='fc2')(x)
    x = BatchNormalization(name='bn_fc2')(x)
    x = Dropout(0.5, name='drop_fc2')(x)

    # Output layer: 6 neurons + softmax → probability distribution
    outputs = Dense(num_classes, activation='softmax', name='emotion_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='EmotionCNN')

    # ── Compile ─────────────────────────────────────────────────────────────
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',   # standard loss for multi-class
        metrics=['accuracy']
    )

    return model


# ── Quick sanity-check ───────────────────────────────────────────────────────
if __name__ == '__main__':
    model = build_emotion_cnn()
    model.summary()
    print("\n[TEST] model.py — architecture built and compiled successfully!")
