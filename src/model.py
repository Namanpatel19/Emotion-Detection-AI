import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def build_model(num_classes, img_size=224):
    """
    Builds the EfficientNetB0 transfer learning model.
    EfficientNetB0 handles its own internal rescaling (expects inputs in [0, 255]).
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))
    
    # Load pretrained EfficientNetB0
    backbone = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_tensor=inputs
    )
    
    # Freeze the backbone initially for Stage 1 training
    backbone.trainable = False
    
    # Add custom classification head
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="emosense_efficientnet")
    return model, backbone

def unfreeze_backbone(backbone, unfreeze_from_layer=None):
    """
    Unfreezes the backbone for fine-tuning.
    If unfreeze_from_layer is None, unfreezes the entire backbone.
    """
    backbone.trainable = True
    
    if unfreeze_from_layer is not None:
        for layer in backbone.layers[:unfreeze_from_layer]:
            # Keep BatchNormalization layers frozen usually, but we'll freeze all before the index
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = False
