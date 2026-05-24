from control import margin
import numpy as np
from typing import Dict, Optional, Any


class PIDStabilityAnalyzer:
    def __init__(
        self,
        model_wrapper: Any,
        open_loop_system: Optional[Any] = None,
    ):

        self.wrapper = model_wrapper
        self.open_loop = open_loop_system

    def analyze(self, X: Optional[np.ndarray] = None) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "gain_margin_db": "N/A",
            "phase_margin_deg": "N/A",
            "settling_time_s": "N/A",
            "overshoot_percent": "N/A",
            "mse": "N/A",
        }
        t = np.linspace(0, 30, 2000)
        u = np.ones_like(t)

        if X is not None:
            u_flat = np.asarray(X).flatten()
            if len(u_flat) > 0:
                u[:len(u_flat)] = u_flat[:len(t)]

        try:
            y = self.wrapper.predict(u, t=t)
            y = np.asarray(y).flatten()
            if len(y) != len(t):
                raise ValueError(f"Długości nie zgadzają się: y={len(y)}, t={len(t)}")
        except Exception as e:
            print(f"[PIDAnalyzer] Błąd symulacji odpowiedzi: {e}")
            metrics["mse"] = "błąd symulacji"
            return metrics

        n_tail = max(100, len(y) // 5)
        steady_state = float(np.mean(y[-n_tail:]))

        try:
            metrics["mse"] = float(np.mean((1.0 - y) ** 2))
            if not np.isfinite(metrics["mse"]):
                metrics["mse"] = "N/A (inf/nan)"
        except:
            metrics["mse"] = "N/A"

        try:
            y_max = float(np.max(y))
            if abs(steady_state) > 1e-6:
                metrics["overshoot_percent"] = ((y_max - steady_state) / abs(steady_state)) * 100
            else:
                metrics["overshoot_percent"] = 0.0
        except:
            metrics["overshoot_percent"] = "N/A"


        try:
            tol = 0.08 * abs(steady_state) if abs(steady_state) > 0.1 else 0.10

            within_band = np.abs(y - steady_state) <= tol

            if np.any(within_band):
                settling_time = float(t[np.where(within_band)[0][-1]])
            else:
                settling_time = float('inf')

            metrics["settling_time_s"] = settling_time if np.isfinite(settling_time) else ">30 s (nie ustala się)"
        except Exception as e:
            print(f"[PIDAnalyzer] Błąd przy obliczaniu settling time: {e}")
            metrics["settling_time_s"] = "N/A"


        if self.open_loop is not None:
            try:
                gm, pm, wg, wp = margin(self.open_loop)

                if gm is None or not np.isfinite(gm) or gm == np.inf:
                    metrics["gain_margin_db"] = "∞ dB (bardzo duży / brak przecięcia amplitudowego)"
                elif gm > 100:
                    metrics["gain_margin_db"] = f">{20 * np.log10(gm):.1f} dB (bardzo duży)"
                else:
                    metrics["gain_margin_db"] = f"{20 * np.log10(gm):.2f} dB"

                if pm is not None and np.isfinite(pm):
                    metrics["phase_margin_deg"] = f"{pm:.2f}°"
                else:
                    metrics["phase_margin_deg"] = "N/A (brak przecięcia fazy)"

            except Exception as e:
                print(f"[PIDAnalyzer] Błąd margin(): {type(e).__name__}: {e}")
                metrics["gain_margin_db"] = f"błąd: {str(e)}"
                metrics["phase_margin_deg"] = f"błąd: {str(e)}"
        return metrics