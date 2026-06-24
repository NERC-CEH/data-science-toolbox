# Contributing a Method

Including your method in the Environmental Data Science Toolbox is a great way to improve the outreach of your work and support collaborative science.

The toolbox supports a wide range of methods and coding languages — whatever your idea, we'd love to hear about it!

## Steps

### 1. Open a GitHub issue

Create a [GitHub issue](https://github.com/NERC-CEH/data-science-toolbox/issues) using either the *notebook idea* template (early stage) or the *notebook inclusion* template (notebook ready to go).

### 2. Create a standalone repository

Create a GitHub repository for your notebook following the naming convention `ds-toolbox-notebook-{name}` (e.g. `ds-toolbox-notebook-bias-correction`). Upload your notebook, any necessary supporting files, and a [CITATION.cff](https://citation-file-format.github.io/) file.

### 3. Prepare your notebook

Your notebook must include a frontmatter metadata block. Use the [template notebook](<../notebook_guidance/template_notebook.ipynb>) as a starting point, and refer to the [notebook metadata guidance](../notebook_guidance/notebook_metadata_guidance.md) for full details on the required fields (title, authors, license, funding, etc.).

### 4. Add the repository as a submodule

Request collaborator access to the toolbox repository by emailing [jercar@ceh.ac.uk](mailto:jercar@ceh.ac.uk), then clone the toolbox repo. Inside the `methods/` folder, add your repository as a git submodule:

```bash
git submodule add {url-of-your-repository}
```

### 5. Add your notebook to the table of contents

Update `myst.yml` to include your notebook:

```yaml
- file: methods/ds-toolbox-notebook-{name}/{notebook-filename}.ipynb
  title: Your Method Title
```

### 6. Preview and raise a pull request

Run `jupyter book start` from the repository root to preview the book locally. Once you're happy with how it looks, push your branch and [open a pull request](https://github.com/NERC-CEH/data-science-toolbox/pulls) for review.
