import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

NUMERICAL_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
LABEL_KEY = "Class"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs=None,
             batch_size=128) -> tf.data.Dataset:
    """Get post-transform features & create batches of data"""
    
    # Get post-transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    
    return dataset

# Model builder using a dictionary of hyperparameters
def model_builder(hp):
    """Build machine learning model"""
    # Use hyperparameter values
    num_layers = hp.get('num_layers', 2)
    dense_units = hp.get('dense_units', 64)
    dropout_rate = hp.get('dropout_rate', 0.2)
    learning_rate = hp.get('learning_rate', 1e-3)

    # Inputs for all transformed numerical features
    inputs = {}
    for key in NUMERICAL_FEATURES:
        inputs[transformed_name(key)] = tf.keras.Input(shape=(1,), name=transformed_name(key), dtype=tf.float32)

    # Concatenate features
    x = tf.keras.layers.concatenate(list(inputs.values()))

    # Build dense layers
    for _ in range(num_layers):
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

    # Output layer (sigmoid activation for binary classification)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses raw TF.Examples and runs model inference."""
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        
        return model(transformed_features)
        
    return serve_tf_examples_fn

# Trainer function
def run_fn(fn_args: FnArgs) -> None:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Determine steps
    train_steps = fn_args.train_steps if fn_args.train_steps and fn_args.train_steps > 0 else None
    eval_steps = fn_args.eval_steps if fn_args.eval_steps and fn_args.eval_steps > 0 else None

    # Load datasets (repeat if steps are specified, otherwise load once per epoch)
    train_set = input_fn(
        fn_args.train_files,
        tf_transform_output,
        num_epochs=None if train_steps else 1,
        batch_size=128
    )
    val_set = input_fn(
        fn_args.eval_files,
        tf_transform_output,
        num_epochs=None if eval_steps else 1,
        batch_size=128
    )

    # Extract tuner hyperparameters if available
    if fn_args.hyperparameters and 'values' in fn_args.hyperparameters:
        hp = fn_args.hyperparameters['values']
    else: 
        hp = {
            'num_layers': 2,
            'dense_units': 64,
            'dropout_rate': 0.2,
            'learning_rate': 1e-3
        }

    model = model_builder(hp)
    model.summary()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
    
    # Save checkpoint weights to a temp directory
    checkpoint_dir = os.path.join(fn_args.serving_model_dir, 'checkpoint')
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'best_weights'),
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    )

    # Train model
    model.fit(
        x=train_set,
        validation_data=val_set,
        epochs=10,
        steps_per_epoch=train_steps,
        validation_steps=eval_steps,
        callbacks=[tensorboard_callback, es, mc]
    )

    # Load best weights before exporting
    try:
        model.load_weights(os.path.join(checkpoint_dir, 'best_weights'))
        print("Successfully loaded best weights from checkpoint.")
    except Exception as e:
        print(f"Could not load best weights from checkpoint: {e}. Saving final epoch model.")

    # Save final model with serving signature
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
