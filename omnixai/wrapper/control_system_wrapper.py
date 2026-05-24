from typing import Union, Callable, Optional

import numpy as np
from control import (TransferFunction, StateSpace,
                     forced_response, step_response)


class ControlSystemWrapper:
    def __init__(self,
                 system: Union[TransferFunction, StateSpace, Callable]):
        self.system = system
        self.is_control_sys = isinstance(system,
                                         (TransferFunction, StateSpace))

    def predict(self, input_signal: np.ndarray,
                t: Optional[np.ndarray] = None) -> np.ndarray:
        if input_signal.ndim == 2:
            input_signal = input_signal.flatten()

        if t is None:
            dt = 0.01
            t = np.arange(0, len(input_signal) * dt, dt)

        if self.is_control_sys:
            if np.allclose(input_signal, input_signal[0]):
                # Optymalizacja: skok jednostkowy -> step_response
                _, y = step_response(self.system, T=t)
            else:
                _, y = forced_response(self.system, T=t,
                                       U=input_signal)
            return y.flatten()
        else:
            return self.system(input_signal)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return self.predict(X)
