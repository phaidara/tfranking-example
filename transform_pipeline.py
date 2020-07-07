import numpy as np
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from apache_beam.options.pipeline_options import StandardOptions, PipelineOptions
import apache_beam as beam

from elwc_coder import ELWCProtoCoder

LIST_SIZE = 4

# Encoding feature specs and metadata
context_specs = {"context_num": tf.io.FixedLenFeature([], tf.float32)}
examples_specs = {"example_categ": tf.io.VarLenFeature(tf.int64),
                  "example_num": tf.io.FixedLenFeature([LIST_SIZE, ], tf.float32),
                  "label": tf.io.FixedLenFeature([LIST_SIZE, ], tf.int64)
                  }

feature_specs = {**context_specs, **examples_specs}

raw_metadata = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec(feature_specs))


def generate_data(sample_size):
    """
    Generates a sample (size sample_size) of data to encode to ELWC
    """
    return [{"context_num": np.random.randint(1, 100),
             "example_categ": np.random.randint(0, 100, size=LIST_SIZE),
             "example_num": np.random.randn(LIST_SIZE),
             "label": np.random.randint(0, 2, size=LIST_SIZE)
             } for _ in range(sample_size)]


def preprocessing_fn(inputs):
    """Tftransform processing function"""
    tft.vocabulary(inputs["example_categ"], vocab_filename="example_categ")
    return {
        "context_num": tft.scale_to_0_1(inputs["context_num"]),
        "example_categ": inputs["example_categ"],
        "example_num": tft.scale_to_0_1(inputs["example_num"]),
        "label": inputs["label"]
    }


def encode():
    """
    Creates a Beam pipeline that generates data, transforms it and encodes it in ELWC
    """
    output_path = "./output"
    options = PipelineOptions()
    options.view_as(StandardOptions).runner = "DirectRunner"

    with beam.Pipeline(options=options) as pipeline:
        with tft_beam.Context(temp_dir="./tmp"):
            raw_data = generate_data(100)
            input_data = (pipeline | beam.Create(raw_data))

            transformed_data, transform_fn = (
                    (input_data, raw_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

            elwc_coder = ELWCProtoCoder(context_specs, examples_specs)
            data, metadata = transformed_data

            _ = (data | beam.Map(elwc_coder.encode) | beam.io.WriteToTFRecord(
                file_path_prefix="{}/data".format(output_path),
                file_name_suffix=".tfrecords"))

            _ = (transform_fn | tft_beam.WriteTransformFn(output_path))


if __name__ == '__main__':
    encode()
