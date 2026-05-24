from typing import Callable
import numpy as np
from lime import lime_tabular
from omnixai.explanations.tabular.feature_importance import FeatureImportance


class LIMEEngineering:
    def __init__(self, predict_fn: Callable,
                 background_data: np.ndarray,
                 feature_names: list = None,
                 mode: str = "regression",
                 kernel_width: float = None,
                 discretize_continuous: bool = False):
        self.predict_fn = predict_fn
        self.feature_names = feature_names or \
            [f"f{i}" for i in range(background_data.shape[1])]
        self.mode = mode

        kwargs = {}
        if kernel_width is not None:
            kwargs['kernel_width'] = kernel_width

        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=background_data,
            feature_names=self.feature_names,
            mode=mode,
            discretize_continuous=discretize_continuous,
            **kwargs
        )

    def explain(self,
                X: np.ndarray,
                num_features: int = 3,
                num_samples: int = 100) -> FeatureImportance:
        explanations = FeatureImportance(self.mode)

        def predict_fn_wrapped(X_):
            y = self.predict_fn(X_)
            return np.array(y).reshape(-1, 1)

        for i in range(X.shape[0]):
            exp = self.explainer.explain_instance(
                X[i],
                predict_fn=predict_fn_wrapped,
                num_features=num_features,
                num_samples=num_samples
            )

            feature_list = exp.as_list()
            if feature_list:
                feature_names_list = []
                feature_values_list = []
                importance_scores_list = []

                for feature_name, score in feature_list:
                    try:
                        idx = self.feature_names.index(feature_name)
                        feature_names_list.append(feature_name)
                        feature_values_list.append(float(X[i][idx]))
                        importance_scores_list.append(float(score))
                    except ValueError:
                        continue

                if feature_names_list:
                    explanations.add(
                        instance=X[i].tolist(),
                        target_label=None,
                        feature_names=feature_names_list,
                        feature_values=feature_values_list,
                        importance_scores=importance_scores_list,
                        sort=True
                    )

        return explanations