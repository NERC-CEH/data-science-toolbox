# %% Importing necessary libraries
import pickle 
import numpy as np
import numpyro.distributions as dist
import jax
from jax import random
import jax.numpy as jnp
from tinygp import kernels
from tinygp.kernels.distance import L2Distance
from tqdm import tqdm
import xarray as xr

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
jax.config.update("jax_enable_x64", True)


# %% Loading the data
# with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary.pkl', 'rb') as f:
#     data_dictionary = pickle.load(f)

with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary_new.pkl', 'rb') as f:
    data_dictionary_new = pickle.load(f)

# data_dictionary_new.update({
#     "idata_residual_model_mean": data_dictionary['idata_residual_model_mean'],
# })

def diagonal_noise(coord, noise):
    return jnp.diag(jnp.full(coord.shape[0], noise))


# %% Defining function for generating posterior predictive realisations of the residuals

def generate_truth_predictive_dist(nx, data_dictionary, metric, posterior_param_realisation):
    kern_var_realisation = posterior_param_realisation["kern_var_realisation"]
    lengthscale_realisation = posterior_param_realisation["lengthscale_realisation"]
    noise_realisation = posterior_param_realisation["noise_realisation"]

    bkern_var_realisation = posterior_param_realisation["bkern_var_realisation"]
    blengthscale_realisation = posterior_param_realisation["blengthscale_realisation"]
    bnoise_realisation = posterior_param_realisation["bnoise_realisation"]

    meanfunc_posterior = data_dictionary['meanfunc_posterior']
    omeanfunc_residual_exp = meanfunc_posterior[f'o{metric}_func_residual'].mean(['draw','chain']).data
    omeanfunc_residual_var = meanfunc_posterior[f'o{metric}_func_residual'].var(['draw','chain']).data
    cmeanfunc_residual_exp = meanfunc_posterior[f'c{metric}_func_residual'].mean(['draw','chain']).data

    ox = data_dictionary["ox"]
    cx = data_dictionary["cx"]
    odata = omeanfunc_residual_exp
    odata_var = omeanfunc_residual_var
    cdata = cmeanfunc_residual_exp
    kernelo = kern_var_realisation * kernels.Matern32(lengthscale_realisation,L2Distance())
    kernelb = bkern_var_realisation * kernels.Matern32(blengthscale_realisation,L2Distance())

    noise = noise_realisation + odata_var
    bnoise = bnoise_realisation
    cnoise = noise_realisation + bnoise

    jitter = 1e-5

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], 0)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], 0), jnp.full(cx.shape[0], 0)]
    )
    k11 = kernelo(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([kernelo(nx, ox), kernelo(nx, cx)])
    k21 = jnp.vstack([kernelo(ox, nx), kernelo(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, noise), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, cnoise),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)

    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    k1g2 = k11 - jnp.matmul(jnp.matmul(k12, k22i), k21)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn


def generate_bias_predictive_dist(nx, data_dictionary, metric, posterior_param_realisation):
    kern_var_realisation = posterior_param_realisation["kern_var_realisation"]
    lengthscale_realisation = posterior_param_realisation["lengthscale_realisation"]
    noise_realisation = posterior_param_realisation["noise_realisation"]

    bkern_var_realisation = posterior_param_realisation["bkern_var_realisation"]
    blengthscale_realisation = posterior_param_realisation["blengthscale_realisation"]
    bnoise_realisation = posterior_param_realisation["bnoise_realisation"]

    meanfunc_posterior = data_dictionary['meanfunc_posterior']
    omeanfunc_residual_exp = meanfunc_posterior[f'o{metric}_func_residual'].mean(['draw','chain']).data
    omeanfunc_residual_var = meanfunc_posterior[f'o{metric}_func_residual'].var(['draw','chain']).data
    cmeanfunc_residual_exp = meanfunc_posterior[f'c{metric}_func_residual'].mean(['draw','chain']).data

    ox = data_dictionary["ox"]
    cx = data_dictionary["cx"]
    odata = omeanfunc_residual_exp
    odata_var = omeanfunc_residual_var
    cdata = cmeanfunc_residual_exp
    kernelo = kern_var_realisation * kernels.Matern32(lengthscale_realisation,L2Distance())
    kernelb = bkern_var_realisation * kernels.Matern32(blengthscale_realisation,L2Distance())

    noise = noise_realisation + odata_var
    bnoise = bnoise_realisation
    cnoise = noise_realisation + bnoise

    jitter = 1e-5

    y2 = jnp.hstack([odata, cdata])
    u1 = jnp.full(nx.shape[0], 0)
    u2 = jnp.hstack(
        [jnp.full(ox.shape[0], 0), jnp.full(cx.shape[0], 0)]
    )
    k11 = kernelb(nx, nx) + diagonal_noise(nx, jitter)
    k12 = jnp.hstack([jnp.full((len(nx), len(ox)), 0), kernelb(nx, cx)])
    k21 = jnp.vstack([jnp.full((len(ox), len(nx)), 0), kernelb(cx, nx)])
    k22_upper = jnp.hstack(
        [kernelo(ox, ox) + diagonal_noise(ox, noise), kernelo(ox, cx)]
    )
    k22_lower = jnp.hstack(
        [
            kernelo(cx, ox),
            kernelo(cx, cx) + kernelb(cx, cx) + diagonal_noise(cx, cnoise),
        ]
    )
    k22 = jnp.vstack([k22_upper, k22_lower])
    k22 = k22
    k22i = jnp.linalg.inv(k22)
    u1g2 = u1 + jnp.matmul(jnp.matmul(k12, k22i), y2 - u2)
    l22 = jnp.linalg.cholesky(k22)
    l22i = jnp.linalg.inv(l22)
    p21 = jnp.matmul(l22i, k21)
    k1g2 = k11 - jnp.matmul(p21.T, p21)
    mvn = dist.MultivariateNormal(u1g2, k1g2)
    return mvn

def generate_posterior_predictive_realisations_dualprocess(
    nx,
    data_dictionary,
    metric,
    num_parameter_realisations,
    num_posterior_pred_realisations,
    rng_key
):
    posterior = data_dictionary[f"idata_residual_model_{metric}"].posterior
    truth_posterior_predictive_realisations = []
    bias_posterior_predictive_realisations = []
    iteration = 0
    for i in tqdm(np.random.randint(posterior.draw.shape, size=num_parameter_realisations)):
        posterior_param_realisation = {
            "iteration": i,
            "kern_var_realisation": posterior["kern_var"].data[0, :][i],
            "lengthscale_realisation": posterior["lengthscale"].data[0, :][i],
            "noise_realisation": posterior["noise"].data[0, :][i],
            "bkern_var_realisation": posterior["bkern_var"].data[0, :][i],
            "blengthscale_realisation": posterior["blengthscale"].data[0, :][i],
            "bnoise_realisation": posterior["bnoise"].data[0, :][i],
        }

        truth_predictive_dist = generate_truth_predictive_dist(
            nx, data_dictionary, metric, posterior_param_realisation
        )
        bias_predictive_dist = generate_bias_predictive_dist(
            nx, data_dictionary, metric, posterior_param_realisation
        )
        iteration += 1

        truth_predictive_realisations = truth_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)
        bias_predictive_realisations = bias_predictive_dist.sample(
            rng_key, sample_shape=(num_posterior_pred_realisations,)
        )
        rng_key, rng_key_ = random.split(rng_key)

        truth_posterior_predictive_realisations.append(truth_predictive_realisations)
        bias_posterior_predictive_realisations.append(bias_predictive_realisations)

    truth_posterior_predictive_realisations = jnp.array(
        truth_posterior_predictive_realisations
    )
    bias_posterior_predictive_realisations = jnp.array(
        bias_posterior_predictive_realisations
    )

    residual_pospred_ds = xr.Dataset(
    coords = {'hyperparameter_draws':range(num_parameter_realisations),
              'gp_draws':range(num_posterior_pred_realisations),
              'nx':range(len(nx)),
              },
    # dims = ['hyperparameter_draws','gp_draws','nx'],
    data_vars = {f'unbiased_{metric}_residual_postpred':(['hyperparameter_draws','gp_draws','nx'], truth_posterior_predictive_realisations),
                 f'bias_{metric}_residual_postpred':(['hyperparameter_draws','gp_draws','nx'], bias_posterior_predictive_realisations),
                }
    )
    return residual_pospred_ds

# %% Generating posterior predictive realisations
residual_postpred_mean = generate_posterior_predictive_realisations_dualprocess(
    data_dictionary_new["cx"],
    data_dictionary_new,
    "mean",
    100,
    5,
    rng_key
)

residual_postpred_logvar = generate_posterior_predictive_realisations_dualprocess(
    data_dictionary_new["cx"],
    data_dictionary_new,
    "logvar",
    100,
    5,
    rng_key
)

data_dictionary_new['residual_postpred_mean'] = residual_postpred_mean
data_dictionary_new['residual_postpred_logvar'] = residual_postpred_logvar


# %% Saving the dictionary:
with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary_new.pkl', 'wb') as f:
    pickle.dump(data_dictionary_new, f)
