# FlowTransformer
The framework for transformer based NIDS development

## Usage instructions

FlowTransformer is a modular pipeline that consists of four key components. These components can be swapped as required for custom implementations, or you can use our supplied implementations:

| **Pre-Processing** | **Input Encoding** | **Model** | **Classification Head** |
|--------------------|--------------------|-----------|-------------------------|
| The pre-processing component accepts arbitrary tabular datasets, and can standardise and transform these into a format applicable for use with machine learning models. For most datasets, our supplied `StandardPreprocessing` approach will handle datasets with categorical and numerical fields, however, custom implementations can be created by overriding `BasePreprocessing`                  | The input encoding component will accept a pre-processed dataset and perform the transformations neccescary to ingest this as part of a sequence to sequence model. For example, the embedding of fields into feature vectors.                  | FlowTransformer supports the use of any sequence-to-sequence machine learning model, and we supply several Transformer implementations.         | The classification head is responsible for taking the sequential output from the model, and transforming this into a fixed length vector suitable for use in classification. We recommed using `LastToken` for most applications.                       |

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

### Implementing Custom Pre-processing 

...

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

