<img align="left" width=300 src="UKCEH_EDST_Logo.png">

This repository is for a UKCEH Jupyter Book that aligns with the NCUK task of providing a suite of open-source, adaptable analytical methods for the academic community. The idea is to apply **FAIR** principles (Findable, Accessible, Interoperable, and Reusable) to statistical methodologies at UKCEH, building upon ideas from the [EDS Jupyter Book](https://edsbook.org/notebooks/gallery) and the platform Kaggle. The concept is to create a collection of Jupyter Notebooks that present the step-by-step nature of data science and that introduce the sophisticated methodologies used at UKCEH in this context.

The hope is that this resource will act as a springboard that encourags uptake of new statistical methods across multiple areas of application, as well as demonstrating the important components of data science pipelines. Focus is placed upon including methods that are applicable to a wide variety of environmental disciplines and that incorporate **integration of multiple data sets**. 

Timescales: Proof of concept with notebooks for GPs, GBTs and SPDEs by end of Feb.

# Data Science Pipeline:
- Providing context and determining initial goals and performance metrics.
- Data exploration, including examining skew, missing data, feature design and collinearity.
- Model design, starting with simplest model possible and detailing assumptions. Thinking about the data generating process.
- Debugging, training the model and examining initial results, identifying areas for improvement.
- Iteratively building up model complexity and examining results (and/or incorporating additional data).
- Finding balance between interpretability, computational performance and predictive performance.
- Drawing up conclusions and commenting on applicability to other domains.
- Decision making through qualitative reference or optimising cost functions.  


# Current Content Plan:
- Contextual information (Markdown Files): 
    - Purpose of Jupyter Book.
    - Overarching description of data science pipelines, statistical models, performance metrics and other important concepts/caveats. 
    - Useful additional resources (EDS book and Kaggle Notebooks etc.).
- Statistical Methodologies (Jupyter Notebooks - R and Python versions):
    - Gradient Boosted Trees (GBTs).
        - XGBoost / LightGBM / CatBoost ([GBTs Podcast](https://www.superdatascience.com/podcast/gradient-boosting-xgboost-lightgbm-and-catboost-with-kirill-eremenko))
    - Gaussian Processes (GPs).
        - Numpyro / TinyGP / GPFlow / PyMC
    - Stochastic Partial Differential Equations (SPDEs).
        - R-INLA / Joint SPDEs Stan
- Inference Techniques (Jupyter Notebook):
    - Maximum Likelihood Estimation (MLE).
    - Maximum A Posteriori (MAP).
    - Markov Chain Monte Carlo (MCMC).
    - Integrated Nested Laplace Approximations (INLA).
    - Variational Inference.
- Important topics in Environmental Data Science:
    - Data and model integration. 
    - Modelling data over COVID years.
    - Causality.
    - Combatting the reproducibility crisis and pitfalls of data science.
    - Uncertainty quantification in Bayesian and Frequentist settings. 
        - How does it work when considering basic multiple linear regression and what about more complex scenarios with models that integrate data sources at different levels of aggregation etc.? 
    - From science to decision making.
        - Discussion around creating cost functions and using the whole joint posterior distribution to optimise.
    - The limits of bad data.

# Considerations:
- Issues with licenses around access to relevant data sets.
    - Access to the data will encourage engagement as people often like working in their own IDE. 
- Size of data sets and compute required for fitting models.
    - Start with minimum working examples.
- Initial focus is proof of concept with basic, minimal working examples incorporated.
- It's important to be clear on the purpose of the notebooks, that is primarily to demonstrate applications of statistical methodologies in environmental research. Focusing on the applicability of methods to a wide variety of environmental disciplines and the advantages of integrating multiple different data sets. The initial purpose is not to conduct scientific research itself in the notebooks, although they provide a spring board from which additional ideas can be generated and easily tested, leading to papers.  
- It's important to avoid pitfalls learnt from previous attempts at similar projects. The DSNE project for example struggled to get much engagement from PhD students, which may have originated from resistance to being asked to contribute code outside of IDE each individual is familiar with. Hopefully having a GIT repository that allows contributors to engage while staying in their preferred IDE will help. Additionally, not being too focused on specific formats for the notebooks but encouraging creativity and focusing on just getting the content in there quickly. 
- Some templates and initial contextual information on the purpose/vision for the Jupyter Book will be useful. 
- Having a mix of R and Python notebooks would improve engagement. The same notebooks can be generated in both R and Python. 

