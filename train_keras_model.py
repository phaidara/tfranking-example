import sys

import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_transform as tft

TRAIN_PATH = "./output/*.tfrecords"
LABEL_FEATURE = "label"

# Loading the Tensorflow Transform output
tf_transform_output = tft.TFTransformOutput("./output")

# Features to use for training
context_dict = {
    "context_num": tf.feature_column.numeric_column(key="context_num", shape=[])
}

examples_dict = {
    "example_num": tf.feature_column.numeric_column(key="example_num", shape=[]),
    "example_categ": tf.feature_column.embedding_column(
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_file(
            key="example_categ",
            dtype=tf.int64,
            vocabulary_file=tf_transform_output.vocabulary_file_by_name(vocab_filename="example_categ")),
        dimension=2)
}


def make_dataset(file_pattern,
                 batch_size,
                 num_epochs):
    """
    Create a Tensorflow dataset from  ELWC tfrecord files
    """
    context_features = context_dict.values()
    label_column = tf.feature_column.numeric_column(
        LABEL_FEATURE, dtype=tf.int64, default_value=-1)
    example_label_features = list(examples_dict.values()) + [label_column]
    context_feature_spec = tf.feature_column.make_parse_example_spec(
        context_features)
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        example_label_features)

    dataset = tfr.data.build_ranking_dataset(
        file_pattern=file_pattern,
        data_format=tfr.data.ELWC,
        batch_size=batch_size,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=True,
        num_epochs=num_epochs
    )

    def _separate_features_and_label(features):
        label = tf.squeeze(features.pop(LABEL_FEATURE), axis=2)
        label = tf.cast(label, tf.float32)
        return features, label

    dataset = dataset.map(
        _separate_features_and_label)
    return dataset


def make_keras_tft_serving_fn(model):
    """
    Serving signature function that parses a serialized ELWC and applies the TFtransform transformation :
    https://www.tensorflow.org/tfx/guide/keras
    """
    model.tft_layer = tf_transform_output.transform_features_layer()
    context_feature_spec = tf.feature_column.make_parse_example_spec(context_dict.values())
    example_feature_spec = tf.feature_column.make_parse_example_spec(examples_dict.values())

    parsing_fn = tfr.data.make_parsing_fn(tfr.data.ELWC, context_feature_spec=context_feature_spec,
                                          example_feature_spec=example_feature_spec
                                          )

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        parsed_features = parsing_fn(serialized_tf_examples)
        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


def train_and_eval_keras_model(train_path, model_dir, hidden_layer_dims, batch_size, num_epochs, save=False):
    """Train a TFranking model with the Keras API and saves it with the make_keras_tft_serving_fn signature"""
    network = tfr.keras.canned.DNNRankingNetwork(
        context_feature_columns=context_dict,
        example_feature_columns=examples_dict,
        hidden_layer_dims=hidden_layer_dims,
        activation=tf.nn.relu,
        dropout=0,
        use_batch_norm=False,
        name="dnn_ranking_model")

    loss = tfr.losses.RankingLossKey.PAIRWISE_HINGE_LOSS
    loss = tfr.keras.losses.get(loss)

    default_metrics = [
        tfr.keras.metrics.OPAMetric(name="metric/OPA")
    ]
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.05)
    ranker = tfr.keras.model.create_keras_model(
        network=network,
        loss=loss,
        metrics=default_metrics,
        optimizer=optimizer,
        size_feature_name=None)

    train_dataset = make_dataset(train_path, batch_size=batch_size, num_epochs=num_epochs)

    ranker.fit(
        x=train_dataset,
        y=None,
        epochs=sys.maxsize,
        verbose=2,
        shuffle=True,
        steps_per_epoch=10
    )

    if save:
        signatures = {
            'serving_default':
                make_keras_tft_serving_fn(
                    ranker
                ).get_concrete_function(
                    tf.TensorSpec(
                        shape=[None],
                        dtype=tf.string,
                        name='examples'
                    )
                ),
        }
        ranker.save(model_dir, save_format='tf', signatures=signatures)


if __name__ == '__main__':
    train_and_eval_keras_model(TRAIN_PATH,
                               model_dir="./output/model",
                               hidden_layer_dims=[1],
                               batch_size=4,
                               num_epochs=5,
                               save=True)
