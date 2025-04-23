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


def evaluate_gbt_model(hyperparams, df_with_folds, feature_cols, label_col="label", num_folds=5):
    """
    Perform K-fold cross-validation to evaluate a GBTClassifier model using AUC as the metric.

    Parameters
    ----------
    hyperparams : dict
        Dictionary containing GBT hyperparameters: 'maxDepth', 'maxIter', and 'stepSize'.
    df_with_folds : pyspark.sql.DataFrame
        Input DataFrame containing a column 'fold' indicating the fold assignment for cross-validation.
    feature_cols : list
        List of feature column names used for model training.
    label_col : str, default="label"
        Name of the label column.
    num_folds : int, default=5
        Number of folds to use for cross-validation.

    Returns
    -------
    float
        Average AUC score across all folds.
    """
    evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc_scores = []

    for fold in range(num_folds):
        train_fold = df_with_folds.filter(col("fold") != fold)
        valid_fold = df_with_folds.filter(col("fold") == fold)

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        train_vec = assembler.transform(train_fold).select("features", col(label_col))
        valid_vec = assembler.transform(valid_fold).select("features", col(label_col))

        model = GBTClassifier(
            featuresCol="features",
            labelCol=label_col,
            maxDepth=hyperparams["maxDepth"],
            maxIter=hyperparams["maxIter"],
            stepSize=hyperparams["stepSize"],
            seed=96
        ).fit(train_vec)

        preds = model.transform(valid_vec)
        auc = evaluator.evaluate(preds)
        auc_scores.append(auc)

    return sum(auc_scores) / len(auc_scores)


def objective(trial, df_with_folds, feature_cols, label_col, num_folds):
    """
    Objective function for Optuna hyperparameter tuning of a GBTClassifier using cross-validation.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial object used by Optuna to suggest hyperparameters.
    df_with_folds : pyspark.sql.DataFrame
        DataFrame with pre-assigned folds (must contain a 'fold' column).
    feature_cols : list
        List of feature column names to be used in the model.
    label_col : str
        Name of the label column.
    num_folds : int
        Number of folds for cross-validation.

    Returns
    -------
    float
        The mean AUC score returned by cross-validation, used by Optuna to guide optimization.
    """
    try:
        params = {
            "maxDepth": trial.suggest_int("maxDepth", 3, 8),
            "maxIter": trial.suggest_int("maxIter", 10, 50),
            "stepSize": trial.suggest_float("stepSize", 0.01, 0.3),
        }
        trial.set_user_attr("params", params)

        auc = evaluate_gbt_model(params, df_with_folds, feature_cols, label_col, num_folds)
        return auc

    except Exception as e:
        trial.set_user_attr("error", str(e))
        return 0.0
