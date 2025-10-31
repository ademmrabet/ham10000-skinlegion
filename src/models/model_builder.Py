import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50

def build_transfer_learning_model(input_shape, num_classes, dropout_rate=0.5):
    """
    Today Boys will become MEN since we will be building our own Transfer Learning Model on ResNet50 for 7-class classification
    """

    #1. Load the pre-trained base model (ResNet50)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    #I definitly need to thank Nvidia and google for this knowledge
    #2. Freeze the base model layers
    # This prevents the pre-trained weights from being modified during initial training
    base_model.trainable = False
    # NEW STEP: Unfreeze the LAST few blocks for Fine-Tuning
    for layer in base_model.layers:
        if layer.name.startswith('conv5_block') or layer.name.startswith('res5'):
            layer.trainable = True

    # 3. Add custom classification layers (the Head or what we call in tunis RASS)
    x = base_model.output
    # then we add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add a dense layer for feature refinement
    x = Dense(512, activation='relu')(x)

    # Add Dropout layer for regularization (crucial to prevent overfitting on rare classes)
    x = Dropout(dropout_rate)(x)

    #final output layer with softmax activation for multi-class classification
    predictions = Dense(num_classes, activation='softmax')(x)

    # 4. Construct the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # 5. Compile the model
    # We will use Adam optimizer (not me im just a human) and Categorical Crossentropy
    optimizer= tf.keras.optimizers.Adam(learning_rate=1e-5)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        #This are dedicated to you Mr. Haithem
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='sensitivity'),
            tf.keras.metrics.SpecificityAtSensitivity(0.5, name='specificity')
        ]
    )

    return model

# Example usage
if __name__=='__main__':
    model=build_transfer_learning_model(input_shape=(224,224,3), num_classes=7)
    model.summary()