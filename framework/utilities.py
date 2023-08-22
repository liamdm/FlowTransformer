#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import base64
import hashlib
import json
import pickle
from typing import Tuple

import pandas as pd


def get_identifier(d:dict):
    raw_json = json.dumps(d, sort_keys=True, indent=False)
    hash = hashlib.sha1(raw_json.encode("utf8")).digest()
    x = base64.b64encode(hash)
    x = x.decode("ASCII")
    x = x.replace("+", "0").replace("=", "0").replace("/", "0")
    return x

def save_feather_plus_metadata(save_path:str, df:pd.DataFrame, metadata:object):
    metadata_path = save_path + ".metadata.pickle"
    df.to_feather(save_path)
    with open(metadata_path, "wb") as w:
        pickle.dump(metadata, w)

def save_pickle(save_path:str, obj:dict):
    with open(save_path, "wb") as w:
        pickle.dump(obj, w)

def load_pickle(save_path:str):
    with open(save_path, "rb") as r:
        return pickle.load(r, fix_imports=True)

def load_feather_plus_metadata(load_path:str) -> Tuple[pd.DataFrame, object]:
    metadata_path = load_path + ".metadata.pickle"
    with open(metadata_path, "rb") as r:
        metadata = pickle.load(r, fix_imports=True)
    data = pd.read_feather(load_path)
    return data, metadata

