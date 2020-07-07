import tensorflow as tf
from tensorflow_serving.apis import input_pb2
from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import Features, Feature


def _encode_value_fn(dtype):
    if dtype.is_integer:
        return lambda value: Feature(int64_list=tf.train.Int64List(value=[value]))
    elif dtype.is_floating:
        return lambda value: Feature(float_list=tf.train.FloatList(value=[value]))
    elif dtype == tf.string:
        return lambda value: Feature(bytes_list=tf.train.BytesList(value=[value.encode()])) if isinstance(value, str) \
            else Feature(bytes_list=tf.train.BytesList(value=[value]))


def _create_example_from_features(features):
    return Example(features=Features(feature=features))


class FeatureHandler(object):
    def __init__(self, name, dtype):
        self._name = name
        self._dtype = dtype
        self.encoding_fn = _encode_value_fn(self._dtype)


class ELWCProtoCoder(object):
    """Coder used to encode a dict of tensor to an ELWC"""

    def __init__(self, context_schema, examples_schema, serialized=True):
        self._context_schema = context_schema
        self._examples_schema = examples_schema
        self._serialized = serialized
        self._context_handlers = {
            name: FeatureHandler(name=name, dtype=feature.dtype)
            for name, feature in self._context_schema.items()
        }
        self._example_handlers = {
            name: FeatureHandler(name=name, dtype=feature.dtype)
            for name, feature in self._examples_schema.items()
        }

    def __reduce__(self):
        return ELWCProtoCoder, (
            self._context_schema,
            self._examples_schema,
            self._serialized
        )

    def encode(self, instance):

        context = self._encode_context(instance)
        examples = self._encode_examples(instance)

        proto = input_pb2.ExampleListWithContext(context=context, examples=examples)

        if self._serialized:
            return proto.SerializeToString()
        else:
            return proto

    def _encode_context(self, instance):
        features = {
            name: handler.encoding_fn(instance[name])
            for name, handler in self._context_handlers.items()
            if instance[name] is not None
        }
        return _create_example_from_features(features)

    def _encode_examples(self, instance):
        k = next(iter(self._example_handlers.keys()))
        features = [dict() for _ in range(len(instance[k]))]
        for name, handler in self._example_handlers.items():
            values = instance[name]
            for i, value in enumerate(values):
                if value is not None:
                    features[i][name] = handler.encoding_fn(value)

        return [_create_example_from_features(f) for f in features]
