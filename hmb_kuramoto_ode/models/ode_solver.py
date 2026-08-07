"""Fixed-step differentiable Euler and RK4 solvers."""
import math
import torch

def integrate(func, y0: torch.Tensor, t0=0., t1=1., step_size=.1, method="rk4", return_trajectory=False):
    n=max(1,math.ceil((t1-t0)/step_size)); dt=(t1-t0)/n; y=y0; states=[y]; t=torch.as_tensor(t0,device=y.device,dtype=y.dtype)
    for _ in range(n):
        if method == "euler": y=y+dt*func(t,y)
        elif method == "rk4":
            k1=func(t,y); k2=func(t+dt/2,y+dt*k1/2); k3=func(t+dt/2,y+dt*k2/2); k4=func(t+dt,y+dt*k3)
            y=y+dt*(k1+2*k2+2*k3+k4)/6
        else: raise ValueError("method must be 'euler' or 'rk4'")
        if not torch.isfinite(y).all(): raise FloatingPointError(f"ODE state became NaN/Inf at t={float(t):.4g}; reduce step_size/coupling")
        t=t+dt; states.append(y)
    return (y,torch.stack(states)) if return_trajectory else y

