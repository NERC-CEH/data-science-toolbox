```{image} EDS_Abstract.jpg 
:alt: abstract-environmental-data-science
:class: bg-primary mb-1 sd-rounded-3
:align: center
```

# UKCEH Data Science Book

This is a collection of data science methods used within UKCEH for a range of environmental applications. The purpose is to apply **FAIR** principles (Findable, Accessible, Interoperable, and Reusable) to statistical methodologies at UKCEH. The hope being that this will encourage **cross-disciplinary use of methods**, enhancing research. Focus is placed upon including methods that are applicable to a wide variety of environmental disciplines and that incorporate **integration of multiple data sets**. 

If you're interested in contributing to this project it would be great to hear from you and you can find details of how to do so via the `CONTRIBUTING.md` page in the root of the repository. 🌞

The current recommended workflow for interactively engaging with the code in the methodology notebooks is to clone the repository linked at the top of each notebook (e.g. {bdg-link-info}`Code Repository (Branch jupyterbook_render) <https://github.com/Jez-Carter/Bias_Correction_Application/tree/jupyterbook_render>`) to get access to the relevant files and then to create a virtual environment and test running different sections of the code in your favourite IDE, such as VS Code.  


| Methods | Key Statistical Concepts | Key Datasets |
| :--- | --- | ---: |
| [Bias Correction of Climate Models](<../notebooks/methods/Bias_Correction_Application/walkthrough_tutorial/Walkthrough Tutorial>) | Gaussian Processes, Bayesian Hierarchical Modelling | Climate Model Output, In-situ Weather Station Measurements 
| Downscaling UK Surface Ozone Concentrations (Planned) | Machine Learning, Gradient Boosted Trees | EMEP4UK Atmospheric Chemistry Transport Model Output, WRF Weather Forecast Model Output, In-situ Ozone Monitoring Network Data
| UK Species Distribution Modelling (Planned) | Stochastic Partial Differential Equations, Integrated Nested Laplace Approximations | UK Butterfly Monitoring Scheme (UKBMS), British Trust for Ornithology (BTO), Environment Agency (EA), Citizen Science Anglers' Riverfly Monitoring Initiative

<br>

| Methods | Generalisability |
| :--- | ---: |
| [Bias Correction of Climate Models](<../notebooks/methods/Bias_Correction_Application/walkthrough_tutorial/Walkthrough Tutorial>) | Combining good-coverage biased datasets with poor-coverage unbiased datasets. Modelling uncertainty when interpolating spatial and/or temporal data. 
| Downscaling UK Surface Ozone Concentrations (Planned) | Using observations and high-resolution predictors to downscale area averaged datasets.
| UK Species Distribution Modelling (Planned) | Combining spatio-temporal datasets  from different observational campaigns.
