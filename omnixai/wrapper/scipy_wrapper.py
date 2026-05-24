import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Callable, Optional, Tuple, Union


class SciPyWrapper:
    MODE_TRANSFER_FUNCTION = "transfer_function"
    MODE_ODE               = "ode"

    def __init__(self,
                 system: Union[signal.TransferFunction, Callable],
                 y0: Optional[np.ndarray] = None,
                 method: str = 'RK45',
                 args: Tuple = ()):

        if isinstance(system, signal.TransferFunction):
            self.mode = self.MODE_TRANSFER_FUNCTION
            self.system = system
            self.is_discrete = (system.dt is not None and system.dt > 0)
        elif callable(system):
            if y0 is None:
                raise ValueError(
                    "Tryb ODE wymaga podania stanu poczatkowego y0. "
                    "Przyklad: SciPyWrapper(fun, y0=np.zeros(2))"
                )
            self.mode = self.MODE_ODE
            self.system = system
            self.y0 = np.asarray(y0, dtype=float)
            self.method = method
            self.args = args
        else:
            raise TypeError(
                "Argument 'system' musi byc obiektem "
                "signal.TransferFunction lub funkcja Callable. "
                f"Otrzymano: {type(system)}")

    def _simulate_tf(self, u: np.ndarray,
                     t: np.ndarray) -> np.ndarray:
        if self.is_discrete:
            system_dlti = signal.dlti(
                self.system.num,
                self.system.den,
                dt=self.system.dt
            )
            _, y = signal.dlsim(system_dlti, u)
            return y.flatten()
        else:
            _, y, _ = signal.lsim(self.system, U=u, T=t)
            return y.flatten()

    def _simulate_ode(self, u: np.ndarray, t: np.ndarray) -> np.ndarray:

        u_interp = interp1d(t, u, kind='linear',
                            fill_value="extrapolate",
                            assume_sorted=True)

        def rhs(t_val, y_val):
            return self.system(t_val, y_val, *self.args,
                               u=float(u_interp(t_val)))

        sol = solve_ivp(rhs,
                        t_span=(t[0], t[-1]),
                        y0=self.y0,
                        t_eval=t,
                        method=self.method,
                        rtol=1e-6, atol=1e-8)

        if not sol.success:
            raise RuntimeError(
                f"solve_ivp nie powiodlo sie: {sol.message}")
        return sol.y.T

    def predict(self, u: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
        u = np.asarray(u, dtype=float)

        if t is None:
            if self.mode == self.MODE_TRANSFER_FUNCTION \
                    and self.is_discrete:
                dt = self.system.dt
                t = np.arange(len(u)) * dt
            else:
                t = np.arange(len(u)) * 0.01

        t = np.asarray(t, dtype=float)

        if self.mode == self.MODE_TRANSFER_FUNCTION:
            return self._simulate_tf(u, t)
        else:
            return self._simulate_ode(u, t)

    def __call__(self, X: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
        if X.ndim == 1:
            return self.predict(X, t)
        return np.array([self.predict(x, t) for x in X])

    @staticmethod
    def create_pid_controller(Kp: float, Ki: float, Kd: float,
                              discrete: bool = False,
                              dt: float = 0.01,
                              N: int = 20) -> signal.TransferFunction:

        tau_f = max(Kd / N, 1e-6)
        num = [Kd, Kp, Ki]
        den = [tau_f, 1.0, 0.0]

        C_continuous = signal.TransferFunction(num, den)

        if discrete:
            C_discrete = C_continuous.to_discrete(dt, method='bilinear')
            return C_discrete

        return C_continuous

    @staticmethod
    def create_closed_loop(
            controller: signal.TransferFunction,
            plant: signal.TransferFunction) -> signal.TransferFunction:

        num_open = np.polymul(controller.num, plant.num)
        den_open = np.polymul(controller.den, plant.den)

        num_closed = num_open
        den_closed = np.polyadd(den_open, num_open)

        dt = None
        if hasattr(controller, 'dt') and controller.dt is not None \
                and controller.dt > 0:
            dt = controller.dt
        elif hasattr(plant, 'dt') and plant.dt is not None \
                and plant.dt > 0:
            dt = plant.dt

        return signal.TransferFunction(num_closed, den_closed, dt=dt)