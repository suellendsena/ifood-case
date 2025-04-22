def find_specific_variables(features_dict, specific_key, specific_value=None):

    keys_with_specific_key=[]

    for k, sub_dict in features_dict.items():
        if isinstance(sub_dict, dict):
            if specific_key in sub_dict:
                if specific_value:
                    if sub_dict[specific_key] == specific_value:
                        keys_with_specific_key.append(k)
                else:
                    keys_with_specific_key.append(k)
    return keys_with_specific_key


def get_features_attribute(features, attribute):
    
    features_to_group={}

    for k, v in features.items():
        if attribute in v:
            features_to_group[k] = v[attribute]
    return features_to_group
