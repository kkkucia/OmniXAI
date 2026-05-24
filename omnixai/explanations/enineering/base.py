from typing import List, Dict, Any

import numpy as np

from omnixai.explainers.base import ExplainerBase
from omnixai.explainers.engineering.agnostic.lime import LIMEEngineering
from omnixai.explainers.engineering.agnostic.shap import ShapEngineering


class EngineeringExplainer(ExplainerBase):
    def __init__(self, explainers: List[str], model, data):
        super().__init__()
        self.explainers_list = explainers
        self.model = model
        self.data = data

    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        results = {}

        if "shap" in self.explainers_list:
            shap_exp = ShapEngineering(
                self.model, self.data,
                feature_names=['Kp', 'Ki', 'Kd'])
            results["shap"] = shap_exp.explain(X)

        if "lime" in self.explainers_list:
            lime_exp = LIMEEngineering(
                self.model, self.data,
                feature_names=['Kp', 'Ki', 'Kd'])
            results["lime"] = lime_exp.explain(X)

        return results
