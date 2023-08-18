# FlowTransformer
The framework for transformer based NIDS development

## Usage instructions

FlowTransformer is a modular pipeline that consists of four key components. These components can be swapped as required for custom implementations, or you can use our supplied implementations:

| **Pre-Processing** | **Input Encoding** | **Model** | **Classification Head** |
|--------------------|--------------------|-----------|-------------------------|
| The pre-processing component accepts arbitrary tabular datasets, and can standardise and transform these into a format applicable for use with machine learning models. For most datasets, our supplied `StandardPreprocessing` approach will handle datasets with categorical and numerical fields, however, custom implementations can be created by overriding `BasePreprocessing`                  | B                  | C         | D                       |

## Jupyter Notebook

...

## Implementing your own solutions with FlowTransformer

### Ingesting custom data formats

...

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

