from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def find_specific_variables(features_dict, specific_key, specific_value=None):
    """
    Find keys in a nested dictionary where a specific key exists,
    optionally filtering by a specific value.
    
    Parameters
    ----------
    features_dict : dict
        Dictionary where values are nested dictionaries representing feature attributes.
    specific_key : str
        The key to look for in each sub-dictionary.
    specific_value : optional
        If provided, only return keys where the specific_key has this value.
    
    Returns
    -------
    list
        List of top-level keys where the specific_key is present (and matches the value, if provided).
    """
    keys_with_specific_key = []

    for k, sub_dict in features_dict.items():
        if isinstance(sub_dict, dict):
            if specific_key in sub_dict:
                if specific_value is not None:
                    if sub_dict[specific_key] == specific_value:
                        keys_with_specific_key.append(k)
                else:
                    keys_with_specific_key.append(k)
    return keys_with_specific_key


def get_features_attribute(features, attribute):
    """
    Extract a specific attribute for all features in a dictionary.

    Parameters
    ----------
    features : dict
        Dictionary where keys are feature names and values are dicts of attributes.
    attribute : str
        Attribute name to extract from each feature's metadata.

    Returns
    -------
    dict
        Dictionary mapping feature names to the value of the specified attribute.
    """
    features_to_group = {}

    for k, v in features.items():
        if attribute in v:
            features_to_group[k] = v[attribute]
    return features_to_group
