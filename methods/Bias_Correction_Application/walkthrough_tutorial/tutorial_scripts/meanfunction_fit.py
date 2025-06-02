# %% Importing necessary libraries
import jax
import jax.numpy as jnp
import numpyro
import pickle 
from numpyro import distributions as dist
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import timeit
import arviz as az

rng_key = jax.random.PRNGKey(1)
jax.config.update("jax_enable_x64", True)

# %% Loading the data
# with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary.pkl', 'rb') as f:
#     data_dictionary = pickle.load(f)

with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary_new.pkl', 'rb') as f:
    data_dictionary = pickle.load(f)

# %% Defining the model for predicting the mean and logvar for each dataset as well as the parameters for the meanfunction giving domain-wide behaviour
def meanfunc_model(data_dictionary):
    omean_b0 = numpyro.sample("omean_b0",data_dictionary['omean_b0_prior'])
    omean_b1 = numpyro.sample("omean_b1",data_dictionary['omean_b1_prior'])
    omean_b2 = numpyro.sample("omean_b2",data_dictionary['omean_b2_prior'])
    omean_noise = numpyro.sample("omean_noise",data_dictionary['omean_noise_prior'])
    omean_func = omean_b0 + omean_b1*data_dictionary['oele_scaled'] + omean_b2*data_dictionary['olat_scaled']
    omean = numpyro.sample("omean",dist.Normal(omean_func, omean_noise))

    ologvar_b0 = numpyro.sample("ologvar_b0",data_dictionary['ologvar_b0_prior'])
    ologvar_noise = numpyro.sample("ologvar_noise",data_dictionary['ologvar_noise_prior'])
    ologvar_func = ologvar_b0 * jnp.ones(data_dictionary['ox'].shape[0])
    ologvar = numpyro.sample("ologvar",dist.Normal(ologvar_func, ologvar_noise))
    ovar = jnp.exp(ologvar)

    obs_mask = (jnp.isnan(data_dictionary['odata'])==False)
    numpyro.sample("AWS Temperature", dist.Normal(omean, jnp.sqrt(ovar)).mask(obs_mask), obs=data_dictionary["odata"])

    cmean_b0 = numpyro.sample("cmean_b0",data_dictionary['cmean_b0_prior'])
    cmean_noise = numpyro.sample("cmean_noise",data_dictionary['cmean_noise_prior'])
    cmean_func = cmean_b0 + omean_b1*data_dictionary['cele_scaled'] + omean_b2*data_dictionary['clat_scaled']
    cmean = numpyro.sample("cmean",dist.Normal(cmean_func, cmean_noise))

    clogvar_b0 = numpyro.sample("clogvar_b0",data_dictionary['clogvar_b0_prior'])
    clogvar_noise = numpyro.sample("clogvar_noise",data_dictionary['clogvar_noise_prior'])
    clogvar_func = clogvar_b0 * jnp.ones(data_dictionary['cx'].shape[0])
    clogvar = numpyro.sample("clogvar",dist.Normal(clogvar_func, clogvar_noise))
    cvar = jnp.exp(clogvar)

    numpyro.sample("Climate Temperature", dist.Normal(cmean, jnp.sqrt(cvar)), obs=data_dictionary["cdata"])

# A function to run inference on the model
def run_inference(
    model, rng_key, num_warmup, num_samples, num_chains, *args, **kwargs
):
    """
    Helper function for doing MCMC inference
    Args:
        model (python function): function that follows numpyros syntax
        rng_key (np array): PRNGKey for reproducible results
        num_warmup (int): Number of MCMC steps for warmup
        num_samples (int): Number of MCMC samples to take of parameters after warmup
        data (jax device array): data in shape [#days,#months,#sites]
        distance_matrix_values(jax device array): matrix of distances between sites, shape [#sites,#sites]
    Returns:
        MCMC numpyro instance (class object): An MCMC class object with functions such as .get_samples() and .run()
    """
    starttime = timeit.default_timer()

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
    )

    mcmc.run(rng_key, *args, **kwargs)

    mcmc.print_summary()
    print("Time Taken:", timeit.default_timer() - starttime)
    return mcmc

# %% Setting priors
data_dictionary.update({
    "omean_b0_prior": dist.Normal(-33.0, 10.0),
    "omean_b1_prior": dist.Normal(0.0, 10.0),
    "omean_b2_prior": dist.Normal(0.0, 10.0),
    "omean_noise_prior": dist.Uniform(1e-2, 10.0),
    "ologvar_b0_prior": dist.Normal(5, 5.0),
    "ologvar_noise_prior": dist.Uniform(1e-3, 2.0),
})

data_dictionary.update({
    "cmean_b0_prior": dist.Normal(-39.0, 10.0),
    "cmean_noise_prior": dist.Uniform(1e-2, 10.0),
    "clogvar_b0_prior": dist.Normal(5, 2.0),
    "clogvar_noise_prior": dist.Uniform(1e-3, 2.0),
})
# %% Running the inference
mcmc = run_inference(meanfunc_model, rng_key, 1000, 2000,4, data_dictionary)

idata = az.from_numpyro(mcmc,
                coords={
                "station": data_dictionary['ds_aws_preprocessed']['station'],
                "x": data_dictionary['ds_climate_preprocessed']['x'],
    },
                dims={"clogvar": ["x"],
                      "cmean": ["x"],
                      "ologvar": ["station"],
                      "omean": ["station"],})
meanfunc_posterior = idata.posterior


# Computing the residuals from the mean function model parameters
meanfunc_posterior = meanfunc_posterior.assign_coords({'oele_scaled':('station', data_dictionary['oele_scaled']),
                        'olat_scaled':('station', data_dictionary['olat_scaled']),
                        'cele_scaled':('x', data_dictionary['cele_scaled']),
                        'clat_scaled':('x', data_dictionary['clat_scaled'])})
meanfunc_posterior['omean_func'] = meanfunc_posterior['omean_b0']+meanfunc_posterior['omean_b1']*meanfunc_posterior['oele_scaled']+meanfunc_posterior['omean_b2']*meanfunc_posterior['olat_scaled']
meanfunc_posterior['cmean_func'] = meanfunc_posterior['cmean_b0']+meanfunc_posterior['omean_b1']*meanfunc_posterior['cele_scaled']+meanfunc_posterior['omean_b2']*meanfunc_posterior['clat_scaled']
meanfunc_posterior['ologvar_func'] = meanfunc_posterior['ologvar_b0']
meanfunc_posterior['clogvar_func'] = meanfunc_posterior['clogvar_b0']
meanfunc_posterior['omean_func_residual'] = meanfunc_posterior['omean']-meanfunc_posterior['omean_func']
meanfunc_posterior['cmean_func_residual'] = meanfunc_posterior['cmean']-meanfunc_posterior['cmean_func']
meanfunc_posterior['ologvar_func_residual'] = meanfunc_posterior['ologvar']-meanfunc_posterior['ologvar_func']
meanfunc_posterior['clogvar_func_residual'] = meanfunc_posterior['clogvar']-meanfunc_posterior['clogvar_func']

data_dictionary['meanfunc_posterior'] = meanfunc_posterior

# %% Saving the dictionary:
# with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary.pkl', 'wb') as f:
#     pickle.dump(data_dictionary, f)

with open('/home/jez/Bias_Correction_Application/walkthrough_tutorial/data_dictionary_new.pkl', 'wb') as f:
    pickle.dump(data_dictionary, f)

# %%
print('out')