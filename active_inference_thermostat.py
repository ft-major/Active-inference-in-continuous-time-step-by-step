#%% md
# # Example of Thermostat based on Active Inference
# this program is an example of an single agent-based active inference model,.
#%% md
# ## Generative process
# the _generative process_ consist of an agent that, depending on his 1-D position $x$,
# modifies the room temperature in the following manner
# $$
# T(x) = \frac{ T_0 }{ x^2 + 1 }
# $$
# with $T_{0}$ temperature at the origin.
#
# The agent is allowed to sense both the local temperature and its temporal derivative. In particular
# $$
# s = T + z_{s}^{gp} \\
# s' = T' + z_{s'}^{gp} = \frac{ \partial T }{ \partial x } \frac{ \partial x }{ \partial t } + z_{s'}^{gp} = - \frac{ 2x T_{0} }{ (x^2+1)^2 } x' + z_{s'}^{gp}
# $$
# where $z_{s}^{gp}$ and $z_{s'}^{gp}$ are normally distributed noise with zero mean and variances $\Sigma_{s}^{gp}$ and $\Sigma_{s'}^{gp}$
# (the gp superscript indicates that is part of the agent's environment described by the generative process and, for the moment, has nothing to do with the brain model).
#
# Finally, the agent is allowed to set its own velocity by setting it equal to the action variable $a$ as
# $$ x'=a $$
#%% md
# ## Generative model
#
# let's specify the dynamical model of the brain that allows to build a minimization scheme for the VFE, remembering that with the **Laplace approximation**
# $$
# F \equiv \int Q(T) \ln \frac{Q(T)}{P(T,s)}dx \approx L(\mu,s) \equiv - \ln P(\mu,s)
# $$
# we are assuming that the brain represents, through the brain state $\mu$, only the most likely environmental cause T of sensory data s.
#
# Let's start assuming that the agent believes that the world's dynamic is given by an exact thermostat dynamic with differential equation
# $$
# \mu' = f(\mu) + z_{\mu} \text{with} \quad f(\mu) \equiv - \mu + T_{des} \, .
# $$
# Using the local linearity approximation, the agent will represents up to the second order of $\mu$:
# $$
# \mu'' = \frac{ \partial f}{ \partial \mu } \mu' + z_{\mu'} \\
# \mu''' = z_{\mu''}
# $$
# Here the third term is specified to explain that, to consider the dynamic up to the second order,
# the next order is set equal only to a Gaussian noise with large variance $\Sigma_{\mu''}$ so that it can be effectively eliminated from the VFE expression.
#
# Is important to note that in this formulation the agent does not desire to be at $T_{des}$ (the prior $P(\mu)$ is omitted since is a flat distribution),
# but believes in an environment with an equilibrium point at $T_{des}$ that works as attractor.

#%% md
# ## Laplace-encoded Energy
# Now we can write explicitly the joint density $P(\mu,s)$
# $$
# P(\tilde{\mu}, \tilde{s}) \simeq P(s|\mu) P(s'|\mu') P(\mu'|\mu) P(\mu''|\mu') = \mathcal{N}(s;\mu,\Sigma_{s}) \mathcal{N}(s';\mu',\Sigma_{s'}) \mathcal{N}(\mu';\mu-T_{des},\Sigma_{\mu}) \mathcal{N}(\mu'';-\mu',\Sigma_{\mu'})
# $$
# that leads to a Variational Free Energy approximated a the Laplace-encoded Energy
# $$
# L(\tilde{\mu}, \tilde{s}) = \frac{ 1 }{ 2 } \left[ \frac{ \varepsilon_{s}^2 }{ \Sigma_{s} } + \frac{ \varepsilon_{s'}^2 }{ \Sigma_{s'} } +
# \frac{ \varepsilon_{\mu}^2 }{ \Sigma_{\mu} } + \frac{ \varepsilon_{\mu'}^2 }{ \Sigma_{\mu'} } \right] + \frac{ 1 }{ 2 } \ln (\Sigma_{s} \Sigma_{s'} \Sigma_{\mu} \Sigma_{\mu'}) + 2 \ln (2\pi)
# $$
# with
# $$
# \begin{aligned}
# \varepsilon_{s} &= s-\mu \\
# \varepsilon_{s'} &= s'-\mu' \\
# \varepsilon_{\mu} &= \mu'+\mu-T_{des} \\
# \varepsilon_{\mu'} &= \mu''+\mu'
# \end{aligned}
# $$
#%% md
# ## Gradient descent
#
# In the Active Inference framework, the agent uses a gradient descent scheme to minimize VFE. In particular, the brain state variables $\tilde{\mu}$ will be updated following
# $$
# \begin{aligned}
# \mu(t+dt) &= \mu(t) + \mu'(t) dt - k_{\mu} \frac{ \partial L }{ \partial \mu } = \mu(t) + \mu'(t) dt - k_{\mu} \left[ -\frac{ \varepsilon_s }{ \Sigma_{s} } + \frac{ \varepsilon_{\mu} }{ \Sigma_{\mu} } \right] \\
# \mu'(t+dt) &= \mu'(t) + \mu''(t) dt - k_{\mu} \frac{ \partial L }{ \partial \mu' } = \mu'(t) + \mu'(t) dt - k_{\mu} \left[ -\frac{ \varepsilon_{s'} }{ \Sigma_{s'} } + \frac{ \varepsilon_{\mu} }{ \Sigma_{\mu} } + \frac{ \varepsilon_{\mu'} }{ \Sigma_{\mu'} } \right] \\
# \mu''(t+dt) &= \mu''(t) - k_{\mu} \frac{ \partial L }{ \partial \mu'' } = \mu''(t) - k_{\mu} \left[ \frac{ \varepsilon_{\mu'} }{ \Sigma_{\mu'} } \right]
# \end{aligned}
# $$
# with the $k_{\mu}$ parameter to be tuned.
#%% md
# ## Action
# To perform an action the agent has to minimize, always through a gradient denscent, the VFE with respect to the action variable, that in this case is equal to $x'$ since the agent is allowed to set is own velocity.
#
# Here we are assuming that the agent has also an inverse model that allows it to know the effects of its actions on the sensory imputs (i.e. it knows that $\mu(x) = \frac{ T_0 }{ x^2 +1 }$ and $\mu'(x,x')=\frac{ d\mu }{ dx } x' = -T_0\frac{ 2x }{ (x^2+1)^2 } x'$ )
# $$
# \begin{aligned}
# \frac{ ds }{ da } &= \frac{ ds }{ dx' } = \frac{ d }{ dx' } (\mu + z_{\mu}) = 0 \\
# \frac{ ds' }{ da } &= \frac{ ds' }{ dx' } = \frac{ d }{ dx' } (\mu' + z_{\mu'}) = \frac{ d\mu' }{ dx' } = -T_0\frac{ 2x }{ (x^2+1)^2 }
# \end{aligned}
# $$
# Using this inverse model the gradient descent with restpec to action will be
# $$
# x'(t+dt) = x'(t) - k_{a} \left[ \frac{ \partial F }{ \partial x' } \right] = x'(t) - k_{a} \left[ \frac{ \partial F }{ \partial s' } \frac{ \partial s' }{ dx' } \right] = x'(t) - k_{a} \left[ \frac{ \varepsilon_{s'} }{ \Sigma_{s'} } (-T_0\frac{ 2x }{ (x^2+1)^2 }) \right]
# $$

#%% md
# # Code

#%% md
# ## Functions
#%%
# Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% md
# ## Simulation parameters
#%%
simT=100                            # Simulation Time
dt=0.005                            # Time step lenght

steps=range(simT/dt)                # Time step number
action=True                         # Variable to enable action

# Generative process parameters
SigmaGP_s = 0.1                     # Generative process s variance
SigmaGP_s1 = 0.1                    # Generative process s'
T0 = 100                            # Startup temperature

# Generative model parameters
Td = 4                              # Desired temperature
actionTime=0                        # Variable that enable action only after a fixed number of steps
Sigma_s = SigmaGP_s                 # Generative model s variance (in this case we're assuming the agent knows gp variace)
Sigma_s1 = SigmaGP_s1               # Generative model s' variance (in this case we're assuming the agent knows gp variace)
Sigma_mu = 0.1                      # Generative model $\mu$ variance
Sigma_mu1 = 0.1                     # Generative model $\mu'$ variance

#%% md
# ## Classes
#%%
class GenMod:
    # Generative model class
    def __init__(self, rng, x, x1, T, T1, SigmaGP_s, SigmaGP_s1 ):

        self.rng = rng              # np.random.RandomState
        self.x = x                  # Agent position
        self.x1 = x1                # Agent velocity
        self.T = 
