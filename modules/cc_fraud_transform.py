import tensorflow as tf
import tensorflow_transform as tft

# List of numerical features to scale
NUMERICAL_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
LABEL_KEY = "Class"

def transformed_name(key):
    """Renaming transformed features by appending _xf"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to RawTensors.
    Returns:
        outputs: map from feature keys to TransformedTensors.
    """
    outputs = {}
    
    # Standardize numerical features using Z-score scaling
    for key in NUMERICAL_FEATURES:
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])
        
    # Pass through target label, casting to float32 for model compat
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)
    
    return outputs
