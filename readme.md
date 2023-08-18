# FlowTransformer
The framework for transformer based NIDS development

## Usage instructions

FlowTransformer is a modular pipeline that consists of four key components. These components can be swapped as required for custom implementations, or you can use our supplied implementations:

| **Pre-Processing** | **Input Encoding** | **Model** | **Classification Head** |
|--------------------|--------------------|-----------|-------------------------|
| The pre-processing component accepts arbitrary tabular datasets, and can standardise and transform these into a format applicable for use with machine learning models. For most datasets, our supplied `StandardPreprocessing` approach will handle datasets with categorical and numerical fields, however, custom implementations can be created by overriding `BasePreprocessing`                  | The input encoding component will accept a pre-processed dataset and perform the transformations neccescary to ingest this as part of a sequence to sequence model. For example, the embedding of fields into feature vectors.                  | FlowTransformer supports the use of any sequence-to-sequence machine learning model, and we supply several Transformer implementations.         | The classification head is responsible for taking the sequential output from the model, and transforming this into a fixed length vector suitable for use in classification. We recommed using `LastToken` for most applications.                       |

To initialise FlowTransformer, we simply need to provide each of these components to the FlowTransformer class:
```
ft = FlowTransformer(
  pre_processing=<pre processing>,
  input_encoding=<encoding>,
  sequential_model=<model>,
  classification_head=<classification head>,
  params=FlowTransformerParameters(window_size=..., mlp_layer_sizes=[...], mlp_dropout=...)
)
```

The FlowTransformerParameters allows control over the sequential pipeline itself. `window_size` is the number of items to ingest in a sequence, `mlp_layer_sizes` is the number of nodes in each layer of the output MLP used for classification at the end of the pipeline, and the `mlp_dropout` is the dropout rate to apply to this network (0 for no dropout). 

FlowTransformer can then be attached to a dataset, doing this will perform pre-processing on the dataset if it has not already been applied (caching is automatic):

```
ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)
```

Once the dataset is loaded, and the input sizes are computed, a Keras model can be built, which consists of the `InputEncoding`, `Model` and `ClassificationHead` components. To do  this, simply call `build_model` which returns a `Keras.Model`:

```
model = ft.build_model()
model.summary()
```

Finally, FlowTransformer has a built in training and evaluation method, which returns pandas dataframes for the training and evaluation results, as well as the final epoch if early stopping is configured:

```
(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=5, steps_per_epoch=64, early_stopping_patience=5)
```

However, the `model` object can be used in part of custom training loops. 

## Jupyter Notebook

...

## Implementing your own solutions with FlowTransformer

### Ingesting custom data formats

Custom data formats can be easily ingested by FlowTransformer. To ingest a new data format, a `DataSpecification` can be defined, and then supplied to `FlowTransformer`:

```
dataset_spec = DatasetSpecification(
    include_fields=['OUT_PKTS', 'OUT_BYTES', ..., 'IN_BYTES', 'L7_PROTO'],
    categorical_fields=['CLIENT_TCP_FLAGS', 'L4_SRC_PORT', ..., 'L4_DST_PORT', 'L7_PROTO'],
    class_column="Attack",
    benign_label="Benign"
)

flow_transformer.load_dataset(dataset_name, path_to_dataset, dataset_spec) 
```

The rest of the pipeline will automatically handle any changes in data format - and will correctly differentiate between categorical and numerical fields.

### Implementing Custom Pre-processing 

To define a custom pre-processing (which is generally not required, given the supplied pre-processing is capable of handling the majority of muiltivariate datasets), override the base class `BasePreprocessing`:

```
class CustomPreprocessing(BasePreProcessing):

    def fit_numerical(self, column_name:str, values:np.array):
        ...

    def transform_numerical(self, column_name:str, values: np.array):
        ...

    def fit_categorical(self, column_name:str, values:np.array):
        ...

    def transform_categorical(self, column_name:str, values:np.array, expected_categorical_format:CategoricalFormat):
        ...
```

Note, the `CategoricalFormat` here is passed automatically by the `InputEncoding` stage of the pipeline:
- If the `InputEncoding` stage expects categorical fields to be encoded as integers, it will return `CategoricalFormat.Integers`
- If the `InputEncoding` stage expets categorical fields to be one-hot encoded, it will return `CategoricalFormat.OneHot`

Both of these cases must be handled by your custom pre-processing implementation.

### Implementing Custom Encodings 

...

### Implementing Custom Transformers

...

### Implementing Custom Classification Heads

...

### Usage in custom training loops

...

## Currently Supported FlowTransformer Components

Please see the wiki for this Github for a list of the associated FlowTransformer components and their description. Feel free to expand the Wiki with your own custom components after your pull request is accepted.

