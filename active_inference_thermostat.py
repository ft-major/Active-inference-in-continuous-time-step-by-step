#%% md
# # Example of Thermostat based on Active Inference
# this program is an example of an single agent-based active inference model,.
# ## Generative process
# the _generative process_ consist of an agent that, depending on his 1-D position $x$,
# modifies the room temperature in the following manner
# $$
# T(x) = \frac{ T_0 }{ x^2 + 1 }
# $$
# with $T_{0}$ temperature at the origin.
#
# The agent is allowed to sense both the local temperature and its temporal derivative. In particularly
# $$
# s_{[0]} = T + z_{s_{[0]}}^{gp} \\
# s_{[1]} = T' + z_{s_{[1]}}^{gp} = \frac{ \partial T }{ \partial x } \frac{ \partial x }{ \partial t } + z_{s_{[1]}}^{gp} = - \frac{ 2x T_{0} }{ (x^2+1)^2 } x' + z_{s_{[1]}}^{gp}
# $$
# where $z_{s_{[0]}}^{gp}$ and $z_{s_{[1]}}^{gp}$ are normally distributed noise with zero mean and variances $\Sigma_{s_{[0]}}^{gp} = \Sigma_{s_{[1]}}^{gp} = 0.1$
# (the gp superscript indicates that is part of the agent's environment described by the generative process and, for the moment, has nothing to do with the brain model).
#
# Finally, the agent is allowed to set its own velocity by setting it equal to the action variable $a$ as
# $$ x'=a $$
# ## Generative model
# let's specify the dynamical model of the brain that allows to build a minimization scheme for the VFE.
#
# The first assumption is that the agent knows exactly how sensory data are generated

#%% md
