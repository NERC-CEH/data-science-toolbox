# Method Publication Guidance

**Status**: draft

**Version**: 0.1

**Project/Funding**: Data Science Platform

**Contributors**: Helen Rawsthorne

**Last modified**: 9 March 2026

Please add your name to **Contributors** and update the **Last modified** date whenever you make changes.

## Table of contents <!-- omit in toc -->

- [Methods: what are they?](#methods-what-are-they)
- [Why this guidance exists](#why-this-guidance-exists)
- [Method publication: step-by-step guide](#method-publication-step-by-step-guide)

## Methods: what are they?

In this guidance, **methods** refer to research outputs that take the form of digital objects. These include (but are not limited to):
- computational workflows
- code and software
- algorithms
- models
- computational notebooks

Methods usually take one or more other digital objects as input (e.g. datasets) and produce one or more digital objects as output (e.g. models).

## Why this guidance exists

Say something about which UKCEH methods this applies to.

The aim of this gudiance is to help you make your method:

- discoverable
- citable
- reusable
- archived for long-term access

## Method publication: step-by-step guide

This guide has **3 levels**. You do not need to complete them all at once, but they should be followed in order.

- [Level 1](#level-1-make-your-method-available-on-github) - make your method publicly available and citable
- [Level 2](#level-2-publish-your-method-on-zenodo) - publish your method and obtain a DOI
- [Level 3](#level-3-contribute-your-method-to-the-data-science-toolbox) - make your method reusable through the Data Science Toolbox

### Level 1: make your method available on GitHub

You can either follow the checklist (Scenario 1) or the step-by-step walkthrough (Scenario 2). Scenario 3 is for cases where GitHub cannot be used.

<details>

<summary><h4 style="display:inline-block">Scenario 1: my method is already on GitHub - show me the checklist</h4></summary>

- [ ] My method is in a dedicated repository in the [UKCEH GitHub organisation](https://github.com/NERC-CEH)
- [ ] The repository contains a `README.md` file, which includes:
  - [ ] a clear title
  - [ ] the appropriate [repostatus badge](https://www.repostatus.org/) for my method, and any other relevant badges
  - [ ] a table of contents
  - [ ] a description of my method
  - [ ] contact details for at least one maintainer
  - [ ] information on project funding, including end date if applicable
- [ ] The repository contains an appropriate `LICENSE.md` file
- [ ] The repository contains a `CITATION.cff` file that follows [UKCEH best practice guidelines](https://github.com/NERC-CEH/repo-guidance/blob/main/cff-guidance/citation-cff_guidelines.md)
- [ ] The repository **About** section contains a short description of my method
- [ ] The repository **Topics** section contains relevant tags
- [ ] The repository contains a `CONTRIBUTING.md` file that explains how others may contribute to my method
- [ ] The repository visibility is set to **Public**

</details>

<details>

<summary><h4 style="display:inline-block">Scenario 2: my method is not yet on GitHub - walk me through what I need to do</h4></summary>

1. Create a new repository for your method in the [UKCEH GitHub organisation](https://github.com/NERC-CEH) and not in a personal account. This ensures better visibility, sustainability and institutional support. Contact [UKCEH IT Support](https://cehacuk.sharepoint.com/sites/hub-it/SitePages/Welcome-to-the-Information-Technology-(IT)-Site1.aspx) if you need access.
2. Add a [`README.md` file](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) to your repository, and include at least the following elements:
   - a clear title (the name of your method)
   - the appropriate [repostatus badge](https://www.repostatus.org/) for your method, and any other relevant badges
   - a table of contents
   - a description of your method
   - contact details for at least one maintainer
   - information on project funding, including end date if applicable
3. Add an appropriate [`LICENSE.md` file](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-license-to-a-repository) to your repository. The [MIT license](https://opensource.org/license/mit) is recommended for methods, unless funders or principal investigators request otherwise.
4. Add a [`CITATION.cff` file](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) to your repository that follows [UKCEH best practice guidelines](https://github.com/NERC-CEH/repo-guidance/blob/main/cff-guidance/citation-cff_guidelines.md).
5. Fill in the **About** section of your repository with a short description of your method.
6. Fill in the [**Topics** section](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics) of your repository with tags that are relevant to your method.
7. Add a [`CONTRIBUTING.md` file](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) to your repository that explains how others may contribute to your method.
8. Ensure the visibility of your repository is [set to **Public**](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories#about-repository-visibility).

</details>

<details>

<summary><h4 style="display:inline-block">Scenario 3: my method can not be made publicly available on GitHub</h4></summary>

1. If the issue is with using GitHub itself, consider an alternative Git platform such as [Bitbucket](https://bitbucket.org/), [Codeberg](https://codeberg.org/) or [GitLab](https://about.gitlab.com/).
2. If the issue is that your method cannot be made public, jump to [Level 2, Scenario 2](#l2s2) of this guide.

</details>

### Level 2: publish your method on Zenodo

[Zenodo](https://zenodo.org/) provides long-term archiving and a DOI. The process for archiving your method with Zenodo differs depending on whether your method is on GitHub (Scenario 1) or not (Scenario 2).

<details>

<summary><h4 style="display:inline-block">Scenario 1: my method is already on GitHub</h4></summary>

1. Follow the [instructions issued by GitHub](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content) on how to archive your repository on Zenodo and get a DOI for it.
2. [Create a new release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) of your repository on GitHub and make use of [Semantic Versionning](https://semver.org/) for the version number in the tag.
3. Inspect your new Zenodo record (you can find it on the [GitHub integration page in your Zenodo account](https://zenodo.org/account/settings/github/)), complete with DOI! A new version will be released on Zenodo each time you create a new release on GitHub.
4. Navigate to [edit your record](https://help.zenodo.org/docs/deposit/manage-records/#edit).
   1. Some of your method metadata will automatically have been pulled across from your GitHub repository (license), release (version) and `CITATION.cff` file (title, authors, keywords, etc.). Complete as many of the empty metadata fields in Zenodo as possible.
   2. [Submit your published method](https://help.zenodo.org/docs/share/submit-to-community/#submit) to the [UKCEH Zenodo Community](https://zenodo.org/communities/ukceh/records) (search for "UK Centre for Ecology & Hydrology").
5. Add the DOI of your newly-published method to the `CITATION.cff` file in your GitHub repository.
6. Add a DOI badge to the `README.md` file in your repository. You can find the markdown code for your DOI badge on the [GitHub integration page in your Zenodo account](https://zenodo.org/account/settings/github/).

</details>

<details>

<summary><h4 id="l2s2" style="display:inline-block">Scenario 2: my method is not on GitHub</h4></summary>

1. [Upload your method to Zenodo](https://help.zenodo.org/docs/deposit/create-new-upload/) as a compressed ZIP archive.
   1. Complete as many of the metadata fields in Zenodo as possible.
   2. [Restrict the visibility of your files](https://help.zenodo.org/docs/deposit/create-new-upload/#visibility), if necessary.
2. Navigate to [edit your record](https://help.zenodo.org/docs/deposit/manage-records/#edit) and [submit your published method](https://help.zenodo.org/docs/share/submit-to-community/#submit) to the [UKCEH Zenodo Community](https://zenodo.org/communities/ukceh/records) (search for "UK Centre for Ecology & Hydrology").

</details>

### Level 3: contribute your method to the Data Science Toolbox

Before you get started, checklist to make sure your method is suitable:

- has an open license
- can be used with publicly-available data, or sample data provided
- has potential to be reused by others

Follow the Toolbox contributing guidelines.
