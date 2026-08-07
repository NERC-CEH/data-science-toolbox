# Contributing a Method/Notebook:

Including your method into the Environmental Data Science Toolbox is a fantastic way of improving the outreach of your work and supporting collaborative science 🌟 

The toolbox supports inclusion of a wide range of methods and different coding languages, so whatever your idea we'd love to hear about it and to help promote it! 

The contribution process is very straightforward and requires only a few steps - so don't hold back and start the process as follows:

## 1. Open a GitHub issue
[Create a GitHub issue in the repository](https://github.com/NERC-CEH/data-science-toolbox/issues/new/choose) and select the '*Notebook Inclusion Request*' template. Fill in the initial fields. If you're able to fill in the additional fields section do that as well, although these can also be done subsequently after following the below steps. If struggling with any of the below steps, write a comment about it in your GitHub issue and tag the toolbox reviewers using: *\@NERC-CEH/toolbox-reviewers*.

## 2. Prepare your notebook
Create a notebook utilising the structure in this [template](https://github.com/NERC-CEH/data-science-toolbox/blob/main/notebook_guidance/template_notebook.ipynb), which when rendered in the toolbox will look like this: [template rendered](https://nerc-ceh.github.io/data-science-toolbox/template-notebook). 

Follow the [notebook metadata guidance](https://nerc-ceh.github.io/data-science-toolbox/notebook-metadata-guidance) to correctly complete the required frontmatter fields in the first cell of the notebook.

## 3. Store your notebook in a GitHub repository
Store the notebook file in a separate GitHub repository to the toolbox. This can be an existing repository or you can create a standalone GitHub repository following the naming convention: *ds-toolbox-notebook-notebookname*. Include any necessary files to run it (e.g. referenced images etc.), as well as a [CITATION.cff](https://citation-file-format.github.io/) file.

## 4. Clone the toolbox repository
Either request collaborator access to the [data-science-toolbox repository](https://github.com/NERC-CEH/data-science-toolbox.git) by emailing jercar@ceh.ac.uk, or create a fork. Then run the following in a terminal on your local machine:

```bash
git clone https://github.com/NERC-CEH/data-science-toolbox.git
```

> [!TIP]
> This local repository tracks the remote repository hosted on GitHub and you can run commands such as '*git fetch origin*' and '*git pull origin*' to update your local copy when the remote repository changes (see [Git fetch and merge](https://longair.net/blog/2009/04/16/git-fetch-and-merge/)).

## 5. Add a link to your notebook

Before making any changes create create a branch on the repository. Naming convention for branches is *{yourname}/{branchname}* (e.g. *jez/bias-correction*). Then add the GitHub url of your notebook to *notebooks.yml* and update the `myst.yml` table of contents in the repository:

```yaml
notebooks:
  - name: ds-toolbox-notebook-name
    url: https://github.com/NERC-CEH/ds-toolbox-notebook-name.git
    branch: main
    path: notebook.ipynb
    assets:
      - images/
```

```yaml
- file: methods/ds-toolbox-notebook-name/notebook.ipynb
  title: Notebook Title
```
  
> [!TIP] 
> - Branches are spaces to develop code, edit files and make commits without affecting the parent branch (normally labelled *'main'* or *'master'*). 
> - Remote and local branches exist. Remote branches show up on GitHub and to work on them you'll have to create a linked local branch that tracks the remote one.
> - You can see current local branches via '*git branch*' and can see the available remote branches via '*git branch -r*'. If you've created a new remote branch via GitHub you'll need to run either '*git fetch origin*' or '*git pull origin*' to observe it when running '*git branch -r*'. 
> - If you've got a local branch and want to create a remote branch to link to it, this can be done via '*git push -u origin local_branch*'.

## 6. Render the toolbox with your notebook
To render the toolbox you'll need to create a python virtual environment. We recommend using *uv* and simply running the following from the terminal in the root of the cloned toolbox repository: 

```bash 
uv sync
```

Then to pull in your notebook run:
```bash
uv run python scripts/sync_notebooks.py --manifest notebooks.yml --execute
```

Finally to render the toolbox with your updated notebook run:
```bash
uv run jupyter book start
```
This will create the build files and will provide a link for viewing.

:::{image} ../images/jupyter-start-link.png
:alt: Jupyter Book Start Rendered Link
:class: abstract-image abstract-image--light
:align: center
:width: 50%
:::

## 7. Iteratively improve notebook and submit pull request

Once you can render the notebook in the toolbox you can iteratively improve the notebook and once you're happy:
- Save the updated notebook to the repository you created in step 3.
- On your branch of the cloned toolbox repository Git commit the changes to *notebooks.yml* and *myst.yml* and git push to your remote branch.
- On the GitHub page of the toolbox or your fork, create a pull request to merge your remote branch into main and to ask collaborators for feedback on the changes.

> [!TIP] 
>Taken directly from [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow): 
>
>"*When you create a pull request, include a summary of the changes and what problem they solve. You can include images, links, and tables to help convey this information. If your pull request addresses an issue, link the issue so that issue stakeholders are aware of the pull request and vice versa. If you link with a keyword, the issue will close automatically when the pull request merges. For more information, see "[Basic writing and formatting syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)" and "[Linking a pull request to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).*"
>- Reviewers will typically leave comments/suggestions on the pull request and these can be addressed, see "[Reviewing changes in pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests).
>- Once the pull request is approved the development branch can be merged with the main repository branch and the development branch deleted safely.  
