import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from keras_tuner.engine import base_tuner
from keras_tuner import RandomSearch
import keras_tuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs
from typing import Any, Dict, NamedTuple, Text

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

# Model builder for hyperparameter tuning
def model_builder(hp):
    """Build machine learning model"""
    # Hyperparameters to tune
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
    dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

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

TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', base_tuner.BaseTuner),
    ('fit_kwargs', Dict[Text, Any]),
])

# Tuner function
def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Determine steps
    train_steps = fn_args.train_steps if fn_args.train_steps and fn_args.train_steps > 0 else None
    eval_steps = fn_args.eval_steps if fn_args.eval_steps and fn_args.eval_steps > 0 else None

    # Load datasets (repeat if steps are specified, otherwise load once per epoch)
    train_set = input_fn(
        fn_args.train_files[0],
        tf_transform_output,
        num_epochs=None if train_steps else 1,
        batch_size=128
    )
    val_set = input_fn(
        fn_args.eval_files[0],
        tf_transform_output,
        num_epochs=None if eval_steps else 1,
        batch_size=128
    )

    model_tuner = RandomSearch(
        hypermodel=model_builder,
        objective=kt.Objective('val_accuracy', direction='max'),
        max_trials=3,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='cc_fraud_tuner',
    )

    return TunerFnResult(
        tuner=model_tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': train_steps,
            'validation_steps': eval_steps,
            'callbacks': [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    mode='max',
                    patience=2,
                    verbose=1
                )
            ]
        }
    )
