🌞 Thanks for being interested in contributing to the NC-UK Environmental Data Science Toolbox! 

Here's some useful information for getting started... 🌱 

[**Contributing a Method/Notebook**](#contributing-a-methodnotebook) & [**Taking Part in the Discussion & Co-design**](#contributing-a-methodnotebook) 

## Contributing a Method/Notebook:

Including your method into the Environmental Data Science Toolbox is a fantastic way of improving the outreach of your work and supporting collaborative science 🌟 

The toolbox supports inclusion of a wide range of methods and different coding languages, so whatever your idea we'd love to hear about it and to help promote it! 

The contribution process is very straightforward and requires only a few steps - so don't hold back and start the process as follows:

- Create a [GitHub issue in the repository](https://github.com/NERC-CEH/data-science-toolbox/issues) and use either the early stage 'notebook idea' template or the later stage 'notebook inclusion' template if you've got a notebook ready to go. 

If the notebook is ready to go then continue by:
- Create a standalone GitHub repository for the notebook/method. Upload the notebook/method and any necessary files to run it, as well as a [CITATION.cff](https://citation-file-format.github.io/) file. Follow the naming convention ds-toolbox-notebook-notebookname (e.g. ds-toolbox-notebook-bias-correction). We recommend using the [template notebook](https://nerc-ceh.github.io/data-science-toolbox/template-notebook) as a starting point, and following the [notebook metadata guidance](https://nerc-ceh.github.io/data-science-toolbox/notebook-metadata-guidance) to correctly complete the required frontmatter fields.
- Request collaborator access to the [data-science-toolbox repository](https://github.com/NERC-CEH/data-science-toolbox.git) - email jercar@ceh.ac.uk. 
- [Clone the repository](#clone-repository) to your local machine:
```bash
git clone https://github.com/NERC-CEH/data-science-toolbox.git
```
> [!TIP]
> This local repository tracks the remote repository hosted on GitHub and you can run commands such as '*git fetch origin*' and '*git pull origin*' to update your local copy when the remote repository changes (see [Git fetch and merge](https://longair.net/blog/2009/04/16/git-fetch-and-merge/)).

- [Create a remote branch](#create-branch) on the GitHub page: [data-science-toolbox/branches](https://github.com/NERC-CEH/data-science-toolbox/branches). Naming convention for branches is {yourname}/{branchname} (e.g. jez/bias-correction). Fetch the remote branch to your local machine via '*git fetch origin*' and then create a linked local branch that tracks the remote one via '*git checkout -b branch_name origin/remote_branch_name*'. This also checks out the branch so you can start working on it.
	
> [!TIP] 
> - Branches are spaces to develop code, edit files and make commits without affecting the parent branch (normally labelled *'main'* or *'master'*). 
> - Remote and local branches exist. Remote branches show up on GitHub and to work on them you'll have to create a linked local branch that tracks the remote one.
> - You can see current local branches via '*git branch*' and can see the available remote branches via '*git branch -r*'. If you've created a new remote branch via GitHub you'll need to run either '*git fetch origin*' or '*git pull origin*' to observe it when running '*git branch -r*'. 
> - If you've got a local branch and want to create a remote branch to link to it, this can be done via '*git push -u origin local_branch*'.

- [Create a link to your standalone repository using git submodules](#create-submodule). Change directory into the methods folder and run:
```bash
git submodule add {url-of-repository}
```

- Update the `myst.yml` table of contents in the repository to include a link to your notebook, e.g.:
```yaml
- file: methods/ds-toolbox-notebook-bias-correction/notebook.ipynb
  title: Bias Correction Application
```

- Render the Jupyter book to see how the notebook looks in the published format. First create a virtual environment, install Jupyter Book, then run live preview and build command from the repository root:
```bash 
conda create --name dstoolbox
conda activate dstoolbox
conda install -c conda-forge jupyter-book

jupyter book start
```

10. Iteratively adjust the notebook and once you're happy with how the notebook looks, then commit any uncommited changes and push to the remote branch. Then create a pull request on GitHub to ask collaborators for feedback on the changes and to hopefully merge changes into the main repository branch.

> [!TIP] 
> If the local branch is linked to a remote branch you can push your changes there periodically using '*git push origin branch_name*'. This is useful if collaborating on a development task. If other collaborators have pushed changes to the remote branch so that is ahead of your local branch, then you can absorb these changes either by using '*git pull origin*' or doing it in stages using '*git fetch origin*' and then '*git checkout branch_name*' and '*git merge origin/branch_name*'. Note if your branch falls behind changes made to the main remote branch, these can be merged in using '*git merge origin/main*'.
>
>Taken directly from [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow): 
>>*When you create a pull request, include a summary of the changes and what problem they solve. You can include images, links, and tables to help convey this information. If your pull request addresses an issue, link the issue so that issue stakeholders are aware of the pull request and vice versa. If you link with a keyword, the issue will close automatically when the pull request merges. For more information, see "[Basic writing and formatting syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)" and "[Linking a pull request to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)."*
>- Reviewers will typically leave comments/suggestions on the pull request and these can be addressed, see "[Reviewing changes in pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests).
>- Once the pull request is approved the development branch can be merged with the main repository branch.  


- Delete the development branch after merging with the main repository branch. The commit history of the development branch will be transferred to the main branch and a commit specific to the pull request will remain.
