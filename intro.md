```{image} images/NatCapUK_colour.png
:alt: natcapuk-logo
:class: bg-primary mb-1
:align: right
:width: 400px
```

```{image} images/eds-abstract-image.jpg 
:alt: abstract-environmental-data-science
:class: bg-primary mb-1 sd-rounded-3
:align: center
```

# UKCEH Data Science Book

This is a collection of data science methods used within UKCEH for a range of environmental applications. The purpose is to apply **FAIR** principles (Findable, Accessible, Interoperable, and Reusable) to statistical methodologies at UKCEH. The hope being that this will encourage **cross-disciplinary use of methods**, enhancing research. Focus is placed upon including methods that are applicable to a wide variety of environmental disciplines and that incorporate **integration of multiple data sets**. 

If you're interested in contributing to this project it would be great to hear from you and you can find details of how to do so via the `CONTRIBUTING.md` page in the root of the repository. 🌞

The current recommended workflow for interactively engaging with the code in the methodology notebooks is to clone the repository linked at the top of each notebook (e.g. {bdg-link-info}`Code Repository (Branch jupyterbook_render) <https://github.com/Jez-Carter/Bias_Correction_Application/tree/jupyterbook_render>`) to get access to the relevant files and then to create a virtual environment and test running different sections of the code in your favourite IDE, such as VS Code.  

| Methods | Key Concepts | Key Datasets |
| :--- | --- | ---: |
| [Bias Correction of Climate Models](<../methods/ds-toolbox-notebook-biascorrection/bias-correction>) {bdg-warning-line}`Ongoing Development`| Gaussian Processes, Bayesian Hierarchical Modelling | Climate Model Output, In-situ Weather Station Measurements 
| [Calculating Risk to Terrestrial Carbon Pool](<../methods/ds-toolbox-notebook-risk-terrestrial-carbon-pool/RiskQuantificationNotebook>) {bdg-warning-line}`Ongoing Development` | Data Access, Data Integration | MODIS Land Cover and Net Primary Production Products, European Space Agency (ESA) Climate Change Initiative (CCI) Soil Moisture Dataset, Global Standardized Precipitation-Evapotranspiration Index (SPEI) Dataset.
| Joint Species Distribution Models in Stan {bdg-warning-line}`Planned` | Stochastic Partial Differential Equations, Integrated Nested Laplace Approximations | UK Butterfly Monitoring Scheme (UKBMS), British Trust for Ornithology (BTO), Environment Agency (EA), Citizen Science Anglers' Riverfly Monitoring Initiative
| Decomposing Error in Multispecies Indicators {bdg-warning-line}`Planned` | Multivariate Methods, Uncertainty, Theory | |
| Accessing EA Data via an API {bdg-warning-line}`Planned` | Data Access, Data Integration | |
| Data Pipelines for JULES Emulation/Portable JULES {bdg-warning-line}`Planned` | | |
| River Utility Tools {bdg-warning-line}`Planned` | Spatio-temporal Integration, Networks | |
| CSV File Checker {bdg-warning-line}`Planned` | Data Quality, Data Integrity | |
| Spatio-temporal Data Integration with INLA {bdg-warning-line}`Tentative` | Spatio-temporal Integration, Bayesian | | 
| Understanding and Modelling Spatio-temporal Lags along Networks {bdg-warning-line}`Tentative` | Spatio-temporal Integration, Networks | |
| State Tagging for Environmental Data QA {bdg-warning-line}`Tentative` | Data Quality, Data Integrity | |

<br>

<!-- | Methods | Generalisability |
| :--- | ---: |
| [Bias Correction of Climate Models](<../notebooks/methods/Bias_Correction_Application/walkthrough_tutorial/Walkthrough Tutorial>) {bdg-warning-line}`Ongoing Development` | Combining good-coverage biased datasets with poor-coverage unbiased datasets. Modelling uncertainty when interpolating spatial and/or temporal data. 
| Joint Species Distribution Models in Stan {bdg-warning-line}`Planned` | 
| Decomposing Error in Multispecies Indicators {bdg-warning-line}`Planned` | 
| Accessing EA Data via an API {bdg-warning-line}`Planned` | 
| Data Pipelines for JULES Emulation/Portable JULES {bdg-warning-line}`Planned` |
| River Utility Tools {bdg-warning-line}`Planned` | 
| CSV File Checker {bdg-warning-line}`Planned` |
| Spatio-temporal Data Integration with INLA {bdg-warning-line}`Tentative` |
| Understanding and Modelling Spatio-temporal Lags along Networks {bdg-warning-line}`Tentative` |
| State Tagging for Environmental Data QA {bdg-warning-line}`Tentative` |  -->