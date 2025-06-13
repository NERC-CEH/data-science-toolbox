<p align="center">
<img width="400" src="images/NatCapUK_colour.png" alt="NCUK Logo">
</p>

<p align="center">
<img width="300" height="300" src="images/ukceh-edsb-logo.png" alt="Environmental Data Science Book Logo">
</p>


> ⚠️ ***PROTOTYPE: This Jupyter Book is a work in progress and is currently in a prototype stage***.

[UKCEH Environmental Data Science Toolbox 🌱](https://NERC-CEH.github.io/data-science-toolbox)

[**Contributing**](#contributing)
| [**Published Book**](#published-book)
| [**User Engagement**](#user-engagement)
| [**Current/Future Content**](#current-and-future-content)
| [**Discussion Topics**](#discussion-topics)

This repository houses the UKCEH Jupyter Book that aligns with the NCUK task of providing a suite of open-source, adaptable analytical methods for the academic community. The concept is to create a collection of user-friendly notebooks demonstrating sophisticated data science methods developed at UKCEH and external contributing organisations. These methods are expected to be generalizable across different areas of application and to have a strong focus on integrative modelling. This resource will enhance collaborative use of data science methods across different environmental themes and will support the UK's national capability in delivering world-leading environmental science. The book is currently in a prototyping stage with methodologies/notebooks expected to evolve over time.

# Contributing

If you're interested in incorporating some methodology into this Jupyter book, or if you're interested in contributing in another way, then please see these [contributing guidelines](/CONTRIBUTING.md) 🌞. Additionally, if you have any questions please either raise them through the [GitHub Discussions](https://github.com/NERC-CEH/data-science-toolbox/discussions) tab or through direct email to [Jeremy Carter](https://github.com/Jez-Carter/) at jercar@ceh.ac.uk. 

# Published Book 

The Jupyter book is currently in very early stages of development. The current published version of the book is available here: [UKCEH Environmental Data Science Toolbox 🌱](https://NERC-CEH.github.io/data-science-toolbox). The book is deployed via [gh-pages](https://jupyterbook.org/en/stable/start/publish.html).

# User Engagement

To engage with the Jupyter book it is advised to visit the published version at: [UKCEH Environmental Data Science Toolbox 🌱](https://NERC-CEH.github.io/data-science-toolbox). The landing page provides context on the different data science methods incorporated and their key components, including datasets, statistical concepts. The notebooks should contain a link to the respective repository housing it and to any papers linked to the method. Additionally, introductory context such as how to set up the environment for running the notebook top-to-bottom will be present. In general, it is expected that users will clone the notebook repository locally, create the virtual environment with relevant packages installed and then run the code in their favourite IDE. 

# Current and Future Content

The current content structure is very simple and consists of: 
- Landing page providing context and including a table summarising features of the current/future methodologies incorporated.
- Data science notebooks labelled by challenge.

In terms of future content plans, there's various additional elements that could be incorporated. This includes for example: guides on UKCEH best practices around research code development and use of computational resources; overarching information around approaches to integrative modelling and creating generalizable methods; notebooks detailing specific elements of data science pipelines such as Bayesian parameter inference techniques and uncertainty quantification; and links to additional resources joining up NC projects.   

# Discussion Topics

1. What to include in the methodology notebooks?
    - The current position is to keep this quite open ended and down to collaborators to incorporate what they feel is most valuable and that meets the projects goals. However, some recommended general structures (such as including a 'Challenge' and 'Approach' short summary at the start of the notebook) have been discussed and are being trialled. The aim is to mix creativity with providing some consistent structure across the notebooks, which will help users navigate and understand the methodologies. Having dropdown sections providing some information around the methods used and their generalizability is recommended, as well as having links to relevant papers and/or repositories at the start of the notebook.

2. Where to house data needed for the notebooks?
    - This is an unresolved issue currently, although there are several options including: in the same repository that holds the notebook; as a published data source made available via Zenodo; in the UKCEH DataLabs infrastructure; and as part of external infrastructures with information provided on how to access. It is recommended to try and keep the size of the data used as small as possible while still relevant for the methodology. Ideally the notebook would utilise open-access data, although if limited-access data is used it's recommended to include information around relevant contact points to get access.

3. What about methodologies that are highly computationally demanding with long run-times? What's the computational back-end of the notebooks?
    - It's recommended that the notebooks should run top-to-bottom without error in <10 mins. In circumstances where it takes longer than this to perform inference on the model, it is recommended to include the code for inference but block comment it out and instead load in the output from a linked dataset. Currently users are expected to clone the repository containing the notebook and run the code locally in their favourite IDE, which means it's useful to limit the computational demands. In the future it's possible specific compute resource will be dedicated to the notebooks, allowing users to run the code in the cloud and without barriers associated with downloading relevant data and installing necessary packages.

> Logo Credits: 
The current Jupyter book logo was generated using OpenAI.

> Accessibility Statement:
This Jupyter Book is being developed with accessibility in mind, following guidelines from the [UKCEH accessibility statement](https://www.ceh.ac.uk/accessibility-statement). We aim to ensure that all content is accessible to a wide range of users, including those with disabilities. If you encounter any accessibility issues or have suggestions for improvement, please let us know through the [GitHub Discussions](https://github.com/NERC-CEH/data-science-toolbox/discussions). 

> Funding Statement:
This research was supported by NERC, through the UKCEH National Capability for UK Challenges Programme NE/Y006208/1

