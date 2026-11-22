"""One logging format for the whole team. Everyone calls log_experiment()."""
import mlflow

EXPERIMENT_NAME = "TriageAI"
RANDOM_STATE = 42


def setup_mlflow(tracking_uri="sqlite:///mlflow.db", experiment=EXPERIMENT_NAME):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def log_experiment(run_name, model_name, params=None, metrics=None, pipeline_config=None,
                   split_ref="train/val/test v1 (70/15/15, grouped, rs=42)",
                   cv_info=None, tags=None, artifacts=None, model=None):
    """params: classifier hyperparameters
    pipeline_config: dict, e.g. {'ngram_range': '(1, 2)', 'stop_words': None, 'min_df': 2}
    metrics: flat dict, standard names:
      macro_f1, weighted_f1, accuracy, val_macro_f1,
      raw_ece, calibrated_ece, coverage, routing_error_rate, human_review_rate,
      auto_routed_count, auto_routed_errors, threshold
    cv_info: {'splitter': 'StratifiedGroupKFold', 'n_folds': 5, 'groups': 'group_id'}
    artifacts: list of file paths (reliability diagram, confusion matrix, ...)
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("split_reference", split_ref)
        for k, v in (params or {}).items():
            mlflow.log_param(f"clf__{k}", v)
        for k, v in (pipeline_config or {}).items():
            mlflow.log_param(f"feat__{k}", v)
        for k, v in (cv_info or {}).items():
            mlflow.log_param(f"cv__{k}", v)
        for k, v in (metrics or {}).items():
            mlflow.log_metric(k, float(v))
        if tags:
            mlflow.set_tags(tags)
        for path in artifacts or []:
            mlflow.log_artifact(path)
        if model is not None:
            mlflow.sklearn.log_model(model, "model")
