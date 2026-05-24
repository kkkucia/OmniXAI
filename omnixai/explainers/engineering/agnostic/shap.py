from typing import Callable

import numpy as np

from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import ShapTabular as OmniShapExplainer


class ShapEngineering:
    """
    Adapter SHAP dla analizy wplywu parametrow inżynierskich układu.
    Uzywa natywnego ShapTabular z OmniXAI dla pelnej
    kompatybilnosci z mechanizmem Dashboard.
    """

    def __init__(self, predict_fn: Callable,
                 background_data: np.ndarray,
                 feature_names: list = None):
        """
        Args:
            predict_fn:      funkcja predykcyjna
            background_data: macierz danych tła
            feature_names:   nazwy cech, domyslnie ['f0','f1','f2']
        """
        self.predict_fn = predict_fn
        self.feature_names = feature_names or \
                             [f"f{i}" for i in range(background_data.shape[1])]

        # Dane tla jako obiekt Tabular - wymagany przez ShapTabular.
        # Z rownan.
        # E[f(X)] = srednia predykcja po wszystkich probkach tla.
        self.training_data = Tabular(
            background_data,
            feature_columns=self.feature_names
        )

        # Wewnetrzny wrapper - rozwiazuje niezgodnosc formatow:
        #   OmniXAI przekazuje dane jako Tabular,
        #   funkcja uzytkownika oczekuje tablicy NumPy.
        # Konwersja: Tabular -> .to_numpy() -> predict_fn -> skalary.
        def omnixai_predict(tabular_data: Tabular) -> np.ndarray:
            return self.predict_fn(tabular_data.to_numpy())

        # Natywny eksplainer OmniXAI zapewniajacy kompatybilnosc
        # z dashboardem i mechanizmem wizualizacji biblioteki.
        self.explainer = OmniShapExplainer(
            training_data=self.training_data,
            predict_function=omnixai_predict,
            mode="regression"
        )

    def explain(self, X: np.ndarray):
        """
        Oblicza wartosci Shapleya dla konfiguracji w X.
        Zwraca:
            FeatureImportance - obiekt kompatybilny z Dashboard OmniXAI
            zawierajacy phi_j dla kazdej cechy i instancji.
        """
        test_instances = Tabular(X, feature_columns=self.feature_names)
        return self.explainer.explain(test_instances)
