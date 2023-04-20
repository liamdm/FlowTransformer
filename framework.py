#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import os
import time
import warnings
from enum import Enum
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd

from utilities import save_feather_plus_metadata, load_feather_plus_metadata, get_identifier

try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras

from keras import Input, Model
from keras.layers import Dense, Dropout

class CategoricalFormat(Enum):
    """
    The format of variables expected by the model as input
    """
    Integers = 0,
    """
    If categorical values should be dictionary encoded as integers
    """
    OneHot = 1
    """
    If categorical values should be one-hot encoded
    """

class EvaluationDatasetSampling(Enum):
    """
    How to choose evaluation samples from the raw dataset
    """
    LastRows = 0
    """
    Take the last rows in the dataset to form the evaluation dataset
    """
    RandomRows  = 1
    """
    Randomly sample rows to make up the evaluation dataset
    """
    FilterColumn = 2
    """
    Define a column that contains a flag indicating if this row is part of the evaluation set
    """

class Component():
    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def parameters(self) -> dict:
        warnings.warn("Parameters have not been implemented for this class!")
        return {}

class BasePreProcessing(Component):
    def __init__(self):
        pass

    def fit_numerical(self, column_name:str, values:np.array):
        raise NotImplementedError("Please override this base class with a custom implementation")

    def transform_numerical(self, column_name:str, values: np.array):
        raise NotImplementedError("Please override this base class with a custom implementation")

    def fit_categorical(self, column_name:str, values:np.array):
        raise NotImplementedError("Please override this base class with a custom implementation")

    def transform_categorical(self, column_name:str, values:np.array, expected_categorical_format:CategoricalFormat):
        raise NotImplementedError("Please override this base class with a custom implementation")

class ModelInputSpecification:
    def __init__(self, feature_names:List[str], n_numeric_features:int, levels_per_categorical_feature:List[int], categorical_format:CategoricalFormat):
        self.feature_names = feature_names

        self.numeric_feature_names = feature_names[:n_numeric_features]
        self.categorical_feature_names = feature_names[n_numeric_features:]
        self.categorical_format:CategoricalFormat = categorical_format

        self.n_numeric_features = n_numeric_features
        self.levels_per_categorical_feature = levels_per_categorical_feature

class FunctionalComponent(Component):
    def __init__(self):
        self.sequence_length: Optional[int] = None
        self.model_input_specification: Optional[ModelInputSpecification] = None
        self.input_shape: Optional[Tuple[int]] = None

    def apply(self, X, prefix: str = None):
        raise NotImplementedError()

    def build(self, sequence_length:int, model_input_specification:ModelInputSpecification):
        self.sequence_length = sequence_length
        self.model_input_specification = model_input_specification

class DatasetSpecification:
    def  __init__(self, include_fields:List[str], categorical_fields:List[str], class_column:str, benign_label:str, test_column:Optional[str]=None):
        """
        Defines the format of specific NIDS dataset
        :param include_fields: The fields to include as part of classification
        :param categorical_fields: Fields that should be treated as categorical
        :param y_column_multiclass: The column name that includes the class of the flow, eg. DDoS or Benign
        :param benign_label: The label of benign traffic, eg. Benign or 0
        """
        self.include_fields:List[str] = include_fields
        self.categorical_fields:List[str] = categorical_fields
        self.class_column = class_column
        self.benign_label = benign_label
        self.test_column:Optional[str] = test_column

class BaseInputEncoding(FunctionalComponent):
    def apply(self, X:List[keras.Input], prefix: str = None):
        raise NotImplementedError("Please override this with a custom implementation")

    @property
    def required_input_format(self) -> CategoricalFormat:
        raise NotImplementedError("Please override this with a custom implementation")

class BaseSequential(FunctionalComponent):
    def __init__(self):
        super().__init__()

class BaseClassificationHead(FunctionalComponent):
    def __init__(self):
        super().__init__()


    def apply_before_transformer(self, X, prefix:str=None):
        return X

class NamedDatasetSpecifications:
    cse_cic_ids_2018 = DatasetSpecification(
            include_fields=['NUM_PKTS_UP_TO_128_BYTES', 'SRC_TO_DST_SECOND_BYTES', 'OUT_PKTS', 'OUT_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'DST_TO_SRC_AVG_THROUGHPUT', 'DURATION_IN', 'L4_SRC_PORT', 'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'IN_PKTS', 'NUM_PKTS_512_TO_1024_BYTES', 'CLIENT_TCP_FLAGS', 'TCP_WIN_MAX_IN', 'NUM_PKTS_256_TO_512_BYTES', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'LONGEST_FLOW_PKT', 'L4_DST_PORT', 'MIN_TTL', 'DST_TO_SRC_SECOND_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'DURATION_OUT', 'FLOW_DURATION_MILLISECONDS', 'TCP_FLAGS', 'MAX_TTL', 'SRC_TO_DST_AVG_THROUGHPUT', 'ICMP_IPV4_TYPE', 'MAX_IP_PKT_LEN', 'RETRANSMITTED_OUT_BYTES', 'IN_BYTES', 'RETRANSMITTED_IN_BYTES', 'TCP_WIN_MAX_OUT', 'L7_PROTO', 'RETRANSMITTED_OUT_PKTS', 'RETRANSMITTED_IN_PKTS'],
            categorical_fields=['CLIENT_TCP_FLAGS', 'L4_SRC_PORT', 'TCP_FLAGS', 'ICMP_IPV4_TYPE', 'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'L4_DST_PORT', 'L7_PROTO'],
            class_column="Attack",
            benign_label="Benign"
        )

    unified_flow_format = DatasetSpecification(
            include_fields=['NUM_PKTS_UP_TO_128_BYTES', 'SRC_TO_DST_SECOND_BYTES', 'OUT_PKTS', 'OUT_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'DST_TO_SRC_AVG_THROUGHPUT', 'DURATION_IN', 'L4_SRC_PORT', 'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'IN_PKTS', 'NUM_PKTS_512_TO_1024_BYTES', 'CLIENT_TCP_FLAGS', 'TCP_WIN_MAX_IN', 'NUM_PKTS_256_TO_512_BYTES', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'LONGEST_FLOW_PKT', 'L4_DST_PORT', 'MIN_TTL', 'DST_TO_SRC_SECOND_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'DURATION_OUT', 'FLOW_DURATION_MILLISECONDS', 'TCP_FLAGS', 'MAX_TTL', 'SRC_TO_DST_AVG_THROUGHPUT', 'ICMP_IPV4_TYPE', 'MAX_IP_PKT_LEN', 'RETRANSMITTED_OUT_BYTES', 'IN_BYTES', 'RETRANSMITTED_IN_BYTES', 'TCP_WIN_MAX_OUT', 'L7_PROTO', 'RETRANSMITTED_OUT_PKTS', 'RETRANSMITTED_IN_PKTS'],
            categorical_fields=['CLIENT_TCP_FLAGS', 'L4_SRC_PORT', 'TCP_FLAGS', 'ICMP_IPV4_TYPE', 'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'L4_DST_PORT', 'L7_PROTO'],
            class_column="Attack",
            benign_label="Benign"
        )

    mqtt = DatasetSpecification(
        include_fields=['prt_src', 'prt_dst', 'proto', 'fwd_num_pkts', 'bwd_num_pkts', 'fwd_mean_iat', 'bwd_mean_iat',
                   'fwd_std_iat', 'bwd_std_iat', 'fwd_min_iat', 'bwd_min_iat', 'fwd_max_iat', 'bwd_max_iat',
                   'fwd_mean_pkt_len', 'bwd_mean_pkt_len', 'fwd_std_pkt_len', 'bwd_std_pkt_len', 'fwd_min_pkt_len',
                   'bwd_min_pkt_len', 'fwd_max_pkt_len', 'bwd_max_pkt_len', 'fwd_num_bytes', 'bwd_num_bytes',
                   'fwd_num_psh_flags', 'bwd_num_psh_flags', 'fwd_num_rst_flags', 'bwd_num_rst_flags',
                   'fwd_num_urg_flags', 'bwd_num_urg_flags'],
        categorical_fields=['prt_src', 'prt_dst', 'proto'],
        class_column='is_attack',
        benign_label='1',
    )

    nsl_kdd = DatasetSpecification(
        include_fields=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                   'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'],
        categorical_fields=['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login'],
        benign_label='normal',
        class_column='class',
        test_column='is_test'
    )

class FlowTransformerParameters:
    def __init__(self, window_size:int, mlp_layer_sizes:List[int], mlp_dropout:float=0.1):
        self.window_size:int = window_size
        self.mlp_layer_sizes = mlp_layer_sizes
        self.mlp_dropout = mlp_dropout

        self._train_ensure_flows_are_ordered_within_windows = True
        self._train_draw_sequential_windows = False

class SequentialInputEncoding(FunctionalComponent):
    def __init__(self, *steps: FunctionalComponent):
        super().__init__()
        self.steps: Tuple[FunctionalComponent] = steps

    def apply(self, X, prefix: str = None):
        v = X
        for step in self.steps:
            v = step.apply(v)
        return v

    def build(self, sequence_length:int, model_input_specification:ModelInputSpecification):
        for step in self.steps:
            step.build(sequence_length, model_input_specification)


class FlowTransformer:
    retain_inmem_cache = False
    inmem_cache = None

    def  __init__(self, pre_processing:BasePreProcessing,
                  input_encoding:BaseInputEncoding,
                  sequential_model:BaseSequential,
                  classification_head:BaseClassificationHead,
                  params:FlowTransformerParameters,
                  rs:np.random.RandomState=None):

        self.rs = np.random.RandomState() if rs is None else rs
        self.classification_head = classification_head
        self.sequential_model = sequential_model
        self.input_encoding = input_encoding
        self.pre_processing = pre_processing
        self.parameters = params

        self.dataset_specification: Optional[DatasetSpecification] = None
        self.X = None
        self.training_mask = None
        self.model_input_spec: Optional[ModelInputSpecification] = None

        self.experiment_key = {}

    def build_model(self, prefix:str=None):
        if prefix is None:
            prefix = ""

        if self.X is None:
            raise Exception("Please call load_dataset before calling build_model()")

        m_inputs = []
        for numeric_feature in self.model_input_spec.numeric_feature_names:
            m_input = Input((self.parameters.window_size, 1), name=f"{prefix}input_{numeric_feature}", dtype="float32")
            m_inputs.append(m_input)

        for categorical_feature_name, categorical_feature_levels in \
            zip(self.model_input_spec.categorical_feature_names, self.model_input_spec.levels_per_categorical_feature):
            m_input = Input(
                (self.parameters.window_size, 1 if self.model_input_spec.categorical_format == CategoricalFormat.Integers else categorical_feature_levels),
                name=f"{prefix}input_{categorical_feature_name}",
                dtype="int32" if self.model_input_spec.categorical_format == CategoricalFormat.Integers else "float32"
            )
            m_inputs.append(m_input)

        self.input_encoding.build(self.parameters.window_size, self.model_input_spec)
        self.sequential_model.build(self.parameters.window_size, self.model_input_spec)
        self.classification_head.build(self.parameters.window_size, self.model_input_spec)

        m_x = self.input_encoding.apply(m_inputs, prefix)

        # in case the classification head needs to add tokens at this stage
        m_x = self.classification_head.apply_before_transformer(m_x, prefix)

        m_x = self.sequential_model.apply(m_x, prefix)
        m_x = self.classification_head.apply(m_x, prefix)

        for layer_i, layer_size in enumerate(self.parameters.mlp_layer_sizes):
            m_x = Dense(layer_size, activation="relu", name=f"{prefix}classification_mlp_{layer_i}_{layer_size}")(m_x)
            m_x = Dropout(self.parameters.mlp_dropout)(m_x) if self.parameters.mlp_dropout > 0 else m_x

        m_x = Dense(1, activation="sigmoid", name=f"{prefix}binary_classification_out")(m_x)
        m = Model(m_inputs, m_x)
        #m.summary()
        return m

    def _load_preprocessed_dataset(self, dataset_name:str,
                     dataset:Union[pd.DataFrame, str],
                     specification:DatasetSpecification,
                     cache_folder:Optional[str]=None,
                     n_rows:int=0,
                     evaluation_dataset_sampling:EvaluationDatasetSampling=EvaluationDatasetSampling.LastRows,
                     evaluation_percent:float=0.2,
                     numerical_filter=1_000_000_000) -> Tuple[pd.DataFrame, ModelInputSpecification]:

        cache_file_path = None

        if dataset_name is None:
            raise Exception(f"Dataset name must be specified so FlowTransformer can optimise operations between subsequent calls!")

        pp_key = get_identifier(
            {
                "__preprocessing_name": self.pre_processing.name,
                **self.pre_processing.parameters
            }
        )

        local_key = get_identifier({
            "evaluation_percent": evaluation_percent,
            "numerical_filter": numerical_filter,
            "categorical_method": str(self.input_encoding.required_input_format),
            "n_rows": n_rows,
        })

        cache_key = f"{dataset_name}_{n_rows}_{pp_key}_{local_key}"

        if FlowTransformer.retain_inmem_cache:
            if FlowTransformer.inmem_cache is not None and cache_key in FlowTransformer.inmem_cache:
                print(f"Using in-memory cached version of this pre-processed dataset. To turn off this functionality set FlowTransformer.retain_inmem_cache = False")
                return FlowTransformer.inmem_cache[cache_key]

        if cache_folder is not None:
            cache_file_name = f"{cache_key}.feather"
            cache_file_path = os.path.join(cache_folder, cache_file_name)

            print(f"Using cache file path: {cache_file_path}")

            if os.path.exists(cache_file_path):
                print(f"Reading directly from cache {cache_file_path}...")
                model_input_spec: ModelInputSpecification
                dataset, model_input_spec = load_feather_plus_metadata(cache_file_path)
                return dataset, model_input_spec

        if isinstance(dataset, str):
            print(f"Attempting to read dataset from path {dataset}...")
            if dataset.lower().endswith(".feather"):
                # read as a feather file
                dataset = pd.read_feather(dataset, columns=specification.include_fields+[specification.class_column])
            elif dataset.lower().endswith(".csv"):
                dataset = pd.read_csv(dataset, nrows=n_rows if n_rows > 0 else None)
            else:
                raise Exception("Unrecognised dataset filetype!")
        elif not isinstance(dataset, pd.DataFrame):
            raise Exception("Unrecognised dataset input type, should be a path to a CSV or feather file, or a pandas dataframe!")

        assert isinstance(dataset, pd.DataFrame)

        if 0 < n_rows < len(dataset):
            dataset = dataset.iloc[:n_rows]

        training_mask = np.ones(len(dataset),  dtype=bool)
        eval_n = int(len(dataset) * evaluation_percent)

        if evaluation_dataset_sampling == EvaluationDatasetSampling.FilterColumn:
            if dataset.columns[-1] != specification.test_column:
                raise Exception(f"Ensure that the 'test' ({specification.test_column}) column is the last column of the dataset being loaded, and that the name of this column is provided as part of the dataset specification")

        if evaluation_dataset_sampling != EvaluationDatasetSampling.LastRows:
            warnings.warn("Using EvaluationDatasetSampling options other than LastRows might leak some information during training, if for example the context window leading up to a particular flow contains an evaluation flow, and this flow has out of range values (out of range to when pre-processing was applied on the training flows), then the model might potentially learn to handle these. In any case, no class leakage is present.")

        if evaluation_dataset_sampling == EvaluationDatasetSampling.LastRows:
            training_mask[-eval_n:] = False
        elif evaluation_dataset_sampling == EvaluationDatasetSampling.RandomRows:
            index = np.arange(self.parameters.window_size, len(dataset))
            sample = self.rs.choice(index, eval_n, replace=False)
            training_mask[sample] = False
        elif evaluation_dataset_sampling == EvaluationDatasetSampling.FilterColumn:
            # must be the last column of the dataset
            training_column = dataset.columns[-1]
            print(f"Using the last column {training_column} as the training mask column")

            v, c = np.unique(dataset[training_column].values,  return_counts=True)
            min_index = np.argmin(c)
            min_v = v[min_index]

            warnings.warn(f"Autodetected class {min_v} of {training_column} to represent the evaluation class!")

            eval_indices = np.argwhere(dataset[training_column].values == min_v).reshape(-1)
            eval_indices = eval_indices[(eval_indices > self.parameters.window_size)]

            training_mask[eval_indices] = False
            del dataset[training_column]

        numerical_columns = set(specification.include_fields).difference(specification.categorical_fields)
        categorical_columns = specification.categorical_fields

        print(f"Set y to = {specification.class_column}")
        new_df = {"__training": training_mask, "__y": dataset[specification.class_column].values}
        new_features = []

        print("Converting numerical columns to floats, and removing out of range values...")
        for col_name in numerical_columns:
            assert col_name in dataset.columns
            new_features.append(col_name)

            col_values = dataset[col_name].values
            col_values[~np.isfinite(col_values)] = 0
            col_values[col_values < -numerical_filter] = 0
            col_values[col_values > numerical_filter] = 0
            col_values = col_values.astype("float32")

            if not np.all(np.isfinite(col_values)):
                raise Exception("Flow format data had non finite values after float transformation!")

            new_df[col_name] = col_values

        print(f"Applying pre-processing to numerical values")
        for i, col_name in enumerate(numerical_columns):
            print(f"[Numerical {i+1:,} / {len(numerical_columns)}] Processing numerical column {col_name}...")
            all_data = new_df[col_name]
            training_data = all_data[training_mask]

            self.pre_processing.fit_numerical(col_name, training_data)
            new_df[col_name] = self.pre_processing.transform_numerical(col_name, all_data)

        print(f"Applying pre-processing to categorical values")
        levels_per_categorical_feature = []
        for i, col_name in enumerate(categorical_columns):
            new_features.append(col_name)
            if col_name == specification.class_column:
                continue
            print(f"[Categorical {i+1:,} / {len(categorical_columns)}] Processing categorical column {col_name}...")

            all_data = dataset[col_name].values
            training_data = all_data[training_mask]

            self.pre_processing.fit_categorical(col_name, training_data)
            new_values = self.pre_processing.transform_categorical(col_name, all_data, self.input_encoding.required_input_format)

            if self.input_encoding.required_input_format == CategoricalFormat.OneHot:
                # multiple columns of one hot values
                if isinstance(new_values, pd.DataFrame):
                    levels_per_categorical_feature.append(len(new_values.columns))
                    for c in new_values.columns:
                        new_df[c] = new_values[c]
                else:
                    n_one_hot_levels = new_values.shape[1]
                    levels_per_categorical_feature.append(n_one_hot_levels)
                    for i in range(n_one_hot_levels):
                        new_df[f"{col_name}_{i}"] = new_values[:, i]
            else:
                # single column of integers
                levels_per_categorical_feature.append(len(np.unique(new_values)))
                new_df[col_name] = new_values

        print(f"Generating pre-processed dataframe...")
        new_df = pd.DataFrame(new_df)
        model_input_spec = ModelInputSpecification(new_features, len(numerical_columns), levels_per_categorical_feature, self.input_encoding.required_input_format)

        print(f"Input data frame had shape ({len(dataset)},{len(dataset.columns)}), output data frame has shape ({len(new_df)},{len(new_df.columns)}) after pre-processing...")

        if cache_file_path is not None:
            print(f"Writing to cache file path: {cache_file_path}...")
            save_feather_plus_metadata(cache_file_path, new_df, model_input_spec)

        if FlowTransformer.retain_inmem_cache:
            if FlowTransformer.inmem_cache is None:
                FlowTransformer.inmem_cache = {}

            FlowTransformer.inmem_cache.clear()
            FlowTransformer.inmem_cache[cache_key] = (new_df, model_input_spec)

        return new_df, model_input_spec

    def load_dataset(self, dataset_name:str,
                     dataset:Union[pd.DataFrame, str],
                     specification:DatasetSpecification,
                     cache_folder:Optional[str]=None,
                     n_rows:int=0,
                     evaluation_dataset_sampling:EvaluationDatasetSampling=EvaluationDatasetSampling.LastRows,
                     evaluation_percent:float=0.2,
                     numerical_filter=1_000_000_000):
        """
        Load a dataset and prepare it for training

        :param dataset: The path to a CSV dataset to load from, or a dataframe
        :param cache_path: Where to store a cached version of this file
        :param n_rows: The number of rows to ingest from the dataset, or 0 to ingest all
        :param cat_dimensionality_limit: Only encode the most frequent n categorical levels, and assign the rest to an 'other' level
        """

        if not os.path.exists(cache_folder):
            warnings.warn(f"Could not find cache folder: {cache_folder}, attempting to create")
            os.mkdir(cache_folder)

        self.dataset_specification = specification
        df, model_input_spec = self._load_preprocessed_dataset(dataset_name, dataset, specification, cache_folder, n_rows, evaluation_dataset_sampling, evaluation_percent, numerical_filter)

        training_mask = df["__training"].values
        del df["__training"]

        y = df["__y"].values
        del df["__y"]

        self.X = df
        self.y = y
        self.training_mask = training_mask
        self.model_input_spec = model_input_spec

        return df

    def evaluate(self, m:keras.Model, batch_size, early_stopping_patience:int=5, epochs:int=100, steps_per_epoch:int=128):
        n_malicious_per_batch = int(0.5 * batch_size)
        n_legit_per_batch = batch_size - n_malicious_per_batch

        overall_y_preserve = np.zeros(dtype="float32", shape=(n_malicious_per_batch + n_legit_per_batch,))
        overall_y_preserve[:n_malicious_per_batch] = 1.

        selectable_mask = np.zeros(len(self.X), dtype=bool)
        selectable_mask[self.parameters.window_size:-self.parameters.window_size] = True
        train_mask = self.training_mask

        y_mask = ~(self.y.astype('str') == str(self.dataset_specification.benign_label))

        indices_train = np.argwhere(train_mask).reshape(-1)
        malicious_indices_train = np.argwhere(train_mask & y_mask & selectable_mask).reshape(-1)
        legit_indices_train = np.argwhere(train_mask & ~y_mask & selectable_mask).reshape(-1)

        indices_test:np.ndarray = np.argwhere(~train_mask).reshape(-1)

        def get_windows_for_indices(indices:np.ndarray, ordered) -> List[pd.DataFrame]:
            X: List[pd.DataFrame] = []

            if ordered:
                # we don't really want to include eval samples as part of context, because out of range values might be learned
                # by the model, _but_ we are forced to in the windowed approach, if users haven't just selected the
                # "take last 10%" as eval option. We warn them prior to this though.
                for i1 in indices:
                    X.append(self.X.iloc[(i1 - self.parameters.window_size) + 1:i1 + 1])
            else:
                context_indices_batch = np.random.choice(indices_train, size=(batch_size, self.parameters.window_size),
                                                         replace=False).reshape(-1)
                context_indices_batch[:, -1] = indices

                for index in context_indices_batch:
                    X.append(self.X.iloc[index])

            return X

        feature_columns_map = {}

        def samplewise_to_featurewise(X):
            sequence_length = len(X[0])

            combined_df = pd.concat(X)

            featurewise_X = []

            if len(feature_columns_map) == 0:
                for feature in self.model_input_spec.feature_names:
                    if feature in self.model_input_spec.numeric_feature_names or self.model_input_spec.categorical_format == CategoricalFormat.Integers:
                        feature_columns_map[feature] = feature
                    else:
                        # this is a one-hot encoded categorical feature
                        feature_columns_map[feature] = [c for c in X[0].columns if str(c).startswith(feature)]

            for feature in self.model_input_spec.feature_names:
                feature_columns = feature_columns_map[feature]
                combined_values = combined_df[feature_columns].values

                # maybe this can be faster with a reshape but I couldn't get it to work
                combined_values = np.array([combined_values[i:i+sequence_length] for i in range(0, len(combined_values), sequence_length)])
                featurewise_X.append(combined_values)

            return featurewise_X

        print(f"Building eval dataset...")
        eval_X = get_windows_for_indices(indices_test, True)
        print(f"Splitting dataset to featurewise...")
        eval_featurewise_X = samplewise_to_featurewise(eval_X)
        eval_y = y_mask[indices_test]
        eval_P = eval_y
        n_eval_P = np.count_nonzero(eval_P)
        eval_N = ~eval_y
        n_eval_N = np.count_nonzero(eval_N)
        print(f"Evaluation dataset is built!")

        print(f"Positive samples in eval set: {n_eval_P}")
        print(f"Negative samples in eval set: {n_eval_N}")

        epoch_results = []

        def run_evaluation(epoch):
            pred_y = m.predict(eval_featurewise_X, verbose=True)
            pred_y = pred_y.reshape(-1) > 0.5

            pred_P = pred_y
            n_pred_P = np.count_nonzero(pred_P)

            pred_N = ~pred_y
            n_pred_N = np.count_nonzero(pred_N)

            TP = np.count_nonzero(pred_P & eval_P)
            FP = np.count_nonzero(pred_P & ~eval_P)
            TN = np.count_nonzero(pred_N & eval_N)
            FN = np.count_nonzero(pred_N & ~eval_N)

            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            balanced_accuracy = (sensitivity + specificity) / 2

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"Epoch {epoch} yielded predictions: {pred_y.shape}, overall balanced accuracy: {balanced_accuracy * 100:.2f}%, TP = {TP:,} / {n_eval_P:,}, TN = {TN:,} / {n_eval_N:,}")

            epoch_results.append({
                "epoch": epoch,
                "P": n_eval_P,
                "N": n_eval_N,
                "pred_P": n_pred_P,
                "pred_N": n_pred_N,
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "bal_acc": balanced_accuracy,
                "f1": f1_score
            })


        class BatchYielder():
            def __init__(self, ordered, random, rs):
                self.ordered = ordered
                self.random = random
                self.cursor_malicious = 0
                self.cursor_legit = 0
                self.rs = rs

            def get_batch(self):
                malicious_indices_batch = self.rs.choice(malicious_indices_train, size=n_malicious_per_batch,
                                                         replace=False) \
                    if self.random else \
                    malicious_indices_train[self.cursor_malicious:self.cursor_malicious + n_malicious_per_batch]

                legitimate_indices_batch = self.rs.choice(legit_indices_train, size=n_legit_per_batch, replace=False) \
                    if self.random else \
                    legit_indices_train[self.cursor_legit:self.cursor_legit + n_legit_per_batch]

                indices = np.concatenate([malicious_indices_batch, legitimate_indices_batch])

                self.cursor_malicious = self.cursor_malicious + n_malicious_per_batch
                self.cursor_malicious = self.cursor_malicious % (len(malicious_indices_train) - n_malicious_per_batch)

                self.cursor_legit = self.cursor_legit + n_legit_per_batch
                self.cursor_legit = self.cursor_legit % (len(legit_indices_train) - n_legit_per_batch)

                X = get_windows_for_indices(indices, self.ordered)
                # each x in X contains a dataframe, with window_size rows and all the features of the flows. There are batch_size of these.

                # we have a dataframe containing batch_size x (window_size, features)
                # we actually want a result of features x (batch_size, sequence_length, feature_dimension)
                featurewise_X = samplewise_to_featurewise(X)

                return featurewise_X, overall_y_preserve

        batch_yielder = BatchYielder(self.parameters._train_ensure_flows_are_ordered_within_windows, not self.parameters._train_draw_sequential_windows, self.rs)

        min_loss = 100
        iters_since_loss_decrease = 0

        train_results = []
        final_epoch = 0

        last_print = time.time()
        elapsed_time = 0

        for epoch in range(epochs):
            final_epoch = epoch

            has_reduced_loss = False
            for step in range(steps_per_epoch):
                batch_X, batch_y = batch_yielder.get_batch()

                t0 = time.time()
                batch_results = m.train_on_batch(batch_X, batch_y)
                t1 = time.time()

                if epoch > 0 or step > 0:
                    elapsed_time += (t1 - t0)
                    if epoch == 0 and step == 1:
                        # include time for last "step" that we skipped with step > 0 for epoch == 0
                        elapsed_time *= 2

                train_results.append(batch_results + [elapsed_time, epoch])

                batch_loss = batch_results[0] if isinstance(batch_results, list) else batch_results

                if time.time() - last_print > 3:
                    last_print = time.time()
                    early_stop_phrase = "" if early_stopping_patience <= 0 else f" (early stop in {early_stopping_patience - iters_since_loss_decrease:,})"
                    print(f"Epoch = {epoch:,} / {epochs:,}{early_stop_phrase}, step = {step}, loss = {batch_loss:.5f}, results = {batch_results} -- elapsed (train): {elapsed_time:.2f}s")

                if batch_loss < min_loss:
                    has_reduced_loss = True
                    min_loss = batch_loss

            if has_reduced_loss:
                iters_since_loss_decrease = 0
            else:
                iters_since_loss_decrease += 1

            do_early_stop = early_stopping_patience > 0 and iters_since_loss_decrease > early_stopping_patience
            is_last_epoch = epoch == epochs - 1
            run_eval = epoch in [6] or is_last_epoch or do_early_stop

            if run_eval:
                run_evaluation(epoch)

            if do_early_stop:
                print(f"Early stopping at epoch: {epoch}")
                break

        eval_results = pd.DataFrame(epoch_results)

        return (train_results, eval_results, final_epoch)


    def time(self, m:keras.Model, batch_size, n_steps=128, n_repeats=4):
        n_malicious_per_batch = int(0.5 * batch_size)
        n_legit_per_batch = batch_size - n_malicious_per_batch

        overall_y_preserve = np.zeros(dtype="float32", shape=(n_malicious_per_batch + n_legit_per_batch,))
        overall_y_preserve[:n_malicious_per_batch] = 1.

        selectable_mask = np.zeros(len(self.X), dtype=bool)
        selectable_mask[self.parameters.window_size:-self.parameters.window_size] = True
        train_mask = self.training_mask

        y_mask = ~(self.y.astype('str') == str(self.dataset_specification.benign_label))

        indices_train = np.argwhere(train_mask).reshape(-1)
        malicious_indices_train = np.argwhere(train_mask & y_mask & selectable_mask).reshape(-1)
        legit_indices_train = np.argwhere(train_mask & ~y_mask & selectable_mask).reshape(-1)

        indices_test:np.ndarray = np.argwhere(~train_mask).reshape(-1)

        def get_windows_for_indices(indices:np.ndarray, ordered) -> List[pd.DataFrame]:
            X: List[pd.DataFrame] = []

            if ordered:
                # we don't really want to include eval samples as part of context, because out of range values might be learned
                # by the model, _but_ we are forced to in the windowed approach, if users haven't just selected the
                # "take last 10%" as eval option. We warn them prior to this though.
                for i1 in indices:
                    X.append(self.X.iloc[(i1 - self.parameters.window_size) + 1:i1 + 1])
            else:
                context_indices_batch = np.random.choice(indices_train, size=(batch_size, self.parameters.window_size),
                                                         replace=False).reshape(-1)
                context_indices_batch[:, -1] = indices

                for index in context_indices_batch:
                    X.append(self.X.iloc[index])

            return X

        feature_columns_map = {}

        def samplewise_to_featurewise(X):
            sequence_length = len(X[0])

            combined_df = pd.concat(X)

            featurewise_X = []

            if len(feature_columns_map) == 0:
                for feature in self.model_input_spec.feature_names:
                    if feature in self.model_input_spec.numeric_feature_names or self.model_input_spec.categorical_format == CategoricalFormat.Integers:
                        feature_columns_map[feature] = feature
                    else:
                        # this is a one-hot encoded categorical feature
                        feature_columns_map[feature] = [c for c in X[0].columns if str(c).startswith(feature)]

            for feature in self.model_input_spec.feature_names:
                feature_columns = feature_columns_map[feature]
                combined_values = combined_df[feature_columns].values

                # maybe this can be faster with a reshape but I couldn't get it to work
                combined_values = np.array([combined_values[i:i+sequence_length] for i in range(0, len(combined_values), sequence_length)])
                featurewise_X.append(combined_values)

            return featurewise_X


        epoch_results = []


        class BatchYielder():
            def __init__(self, ordered, random, rs):
                self.ordered = ordered
                self.random = random
                self.cursor_malicious = 0
                self.cursor_legit = 0
                self.rs = rs

            def get_batch(self):
                malicious_indices_batch = self.rs.choice(malicious_indices_train, size=n_malicious_per_batch,
                                                         replace=False) \
                    if self.random else \
                    malicious_indices_train[self.cursor_malicious:self.cursor_malicious + n_malicious_per_batch]

                legitimate_indices_batch = self.rs.choice(legit_indices_train, size=n_legit_per_batch, replace=False) \
                    if self.random else \
                    legit_indices_train[self.cursor_legit:self.cursor_legit + n_legit_per_batch]

                indices = np.concatenate([malicious_indices_batch, legitimate_indices_batch])

                self.cursor_malicious = self.cursor_malicious + n_malicious_per_batch
                self.cursor_malicious = self.cursor_malicious % (len(malicious_indices_train) - n_malicious_per_batch)

                self.cursor_legit = self.cursor_legit + n_legit_per_batch
                self.cursor_legit = self.cursor_legit % (len(legit_indices_train) - n_legit_per_batch)

                X = get_windows_for_indices(indices, self.ordered)
                # each x in X contains a dataframe, with window_size rows and all the features of the flows. There are batch_size of these.

                # we have a dataframe containing batch_size x (window_size, features)
                # we actually want a result of features x (batch_size, sequence_length, feature_dimension)
                featurewise_X = samplewise_to_featurewise(X)

                return featurewise_X, overall_y_preserve

        batch_yielder = BatchYielder(self.parameters._train_ensure_flows_are_ordered_within_windows, not self.parameters._train_draw_sequential_windows, self.rs)

        min_loss = 100
        iters_since_loss_decrease = 0

        final_epoch = 0

        last_print = time.time()
        elapsed_time = 0

        batch_times = []


        for step in range(n_steps):
            batch_X, batch_y = batch_yielder.get_batch()

            local_batch_times = []
            for i in range(n_repeats):
                t0 = time.time()
                batch_results = m.predict_on_batch(batch_X)
                t1 = time.time()
                local_batch_times.append(t1 - t0)

            batch_times.append(local_batch_times)

            if time.time() - last_print > 3:
                last_print = time.time()
                print(f"Step = {step}, running model evaluation... Average times = {np.mean(np.array(batch_times).reshape(-1))}")

        return batch_times



