🌞 Thanks for being interested in contributing to the UKCEH data science toolbox! 

Here's some useful information for getting started... 🌱

If you have a method that you think would be useful to include in the UKCEH Data Science Book, then please do get in touch! The method should be relevant to environmental data science and ideally be applicable to a wide range of environmental disciplines.

[**Contributing a Method/Notebook**](#contributing-a-methodnotebook)
| [**Jupyter Book Workflow**](#jupyter-book-description)
| [**Additional Detail**](#additional-detail)

## Contributing a Method/Notebook:
Here we'll go through the basic steps for contributing a method or notebook to the UKCEH Data Science Book. The process is designed to be simple and straightforward, allowing you to focus on developing your method or notebook without getting bogged down in technical details.

1. [Create a standalone repository for the notebook/method](#create-standalone-repository). Upload the notebook and any necessary files to run it. Do this in the NERC-CEH GitHub organisation and follow the naming convention ds-toolbox-notebook-notebookname (e.g. ds-toolbox-notebook-bias-correction).

2. [Request collaborator access]((#access)) to the [data-science-toolbox repository](https://github.com/NERC-CEH/data-science-toolbox.git) - email jercar@ceh.ac.uk. 

3. [Clone the repository](#clone-repository) to your local machine:
```bash
git clone https://github.com/NERC-CEH/data-science-toolbox.git
```

4. [Create an issue](#create-issue) in the repository detailing what you intend to work on/include: [Current Issues](https://github.com/NERC-CEH/data-science-toolbox/issues).

5. [Create a remote branch](#create-branch) on the GitHub page: [data-science-toolbox/branches](https://github.com/NERC-CEH/data-science-toolbox/branches). Naming convention for branches is {yourname}/{branchname} (e.g. jez/bias-correction).

6. [Fetch the remote branch to your local machine and create a linked local branch](#create-branch) that tracks the remote one via:
```bash
git fetch origin
git checkout -b branch_name origin/remote_branch_name
```

7. [Create a link to your standalone repository using git submodules](#create-submodule). Change directory into the methods folder and run:
```bash
git submodule add {url-of-repository}
```

8. [Update the `_toc.yml` file](#update-toc) in the repository to include a link to your notebook, e.g.:
```yaml
- file: methods/ds-toolbox-notebook-bias-correction/notebook.ipynb
  title: Bias Correction Application
```

9. [Render the Jupyter book](#render-jupyter-book) to see how the notebook looks in the book format. First creat a virtual environment then change directory into folder above data-science-toolbox and create the new build files by running the jupyter-book build command:
```bash 
conda create --name dstoolbox
conda activate jupyter-book
conda install -c conda-forge jupyter-book

jupyter-book build data-science-toolbox
```

10. [Examine the rendered book](#render-jupyter-book) by double clicking the `data-science-toolbox/_build/html/index.html` file. 

11. If you're happy with how the notebook looks, then you commit any uncommited changes and push to the remote branch, then [create a pull request](#pull-changes) on GitHub to ask collaborators for feedback on the changes and to hopefully merge changes into the main repository branch.

12. [Delete the development branch](#close-issue) after merging with the main repository branch. The commit history of the development branch will be transferred to the main branch and a commit specific to the pull request will remain.

## Jupyter Book Description:

A [Jupyter Book](https://jupyterbook.org/en/stable/intro.html) is an organised collection of [Jupyter Notebooks](https://www.dataquest.io/blog/jupyter-notebook-tutorial/) that cover a specific topic. This repository houses the 'UKCEH Data Science Toolbox' Jupyter book, which presents a collection of Jupyter notebook tutorials on specific data science methodologies developed at UKCEH, detailing the full data science pipeline. Here we provide a guideline on developing the Jupyter book's content and suggest useful links and documentation. See this [Jupyter Books 101](https://www.google.com/search?sca_esv=853f175af13f0422&rlz=1C1GCEA_enGB1127GB1127&sxsrf=ADLYWILIDB_FKqa2tEu-BFTAyFkn4C5pZA:1730195044702&q=ghp-import&tbm=vid&source=lnms&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWfbQph1uib-VfD_izZO2Y5sC3UdQE5x8XNnxUO1qJLaQdh3mUfgbiNAX47iHD_lJjnnrtkrknsy6VQXK4-aRHxqnPwuFZlmbREdWKLZFI-gq_UsBLTYJHKqEeHaFb3F8RYl5naC8STX8rrrXVJLtrqfmiz5ev1aurnZdmKum0bTFGUA16&sa=X&ved=2ahUKEwiYhKDoprOJAxWsVkEAHbmtBuoQ0pQJegQIGhAB&biw=1745&bih=828&dpr=1.1&safe=active&ssui=on#fpstate=ive&vld=cid:6619f956,vid:lZ2FHTkyaMU,st:0) for a nice YouTube introduction into the topic. 

> [!NOTE]  
>To be able to re-build the Jupyter book and view your adjustments you'll need to download the Python package, either using PIP ([jupyter-book PyPI](https://pypi.org/project/jupyter-book/)) or Conda ([jupyter-book conda-forge](https://anaconda.org/conda-forge/jupyter-book)), see [Install Jupyter Book](https://jupyterbook.org/en/stable/start/overview.html). It is suggested to use Conda and to run the commands via the Anaconda Prompt if on a windows machine.  

1. The repository contains a basic [anatomy of a jupyter book](https://jupyterbook.org/en/stable/start/create.html#anatomy-of-a-book). The most important files in terms of setting the build options are the table of contents (`_toc.yml`) and book configuration (`_config.yml`) files. It is suggested you will likely not need to edit the `_config.yml` file. However, after creating some content (such as a methodology notebook), for the content to show up in the build, the `_toc.yml` will need updating to include a respective link.  

2. The main type of content files include [markdown](https://jupyterbook.org/en/stable/start/create.html#markdown-files-md) (.md) files and [jupyter notebooks](https://jupyterbook.org/en/stable/start/create.html#jupyter-notebooks-ipynb) (.ipynb) files.
- Markdown files are generally for contextual information (e.g. the repository's landing README.md page). There are various good options for getting the hand of [markdown file syntax](https://www.markdownguide.org/basic-syntax/). One option is to download notemaker apps such as [Obsidian](https://obsidian.md/) or to use the markdown preview option in [VS Code](https://code.visualstudio.com/docs/languages/markdown). It's worth noting that Jupyter book allows a slightly different collection of markdown syntax ([common and MyST flavours](https://jupyterbook.org/en/stable/start/create.html#anatomy-of-a-book)) compared to software such as Obsidian, so if something is not rending correctly then this is probably why. Another good place to pickup the syntax is from contextual files for the Jupyter book in the repository, such as the `intro.md` file that determines content for the landing page of the book.
- Jupyter notebooks are generally for mixing code cells with contextual writing. They're a very good way of explaining complex data science workflows in a step-by-step approach. Notebooks can be run in R, Python and Julia. See the following [methodology notebook template](https://jez-carter.github.io/UKCEH_Data_Science_Book/notebooks/methods/template.html) document for some design ideas, as well as official documentation for [formatting code outputs](https://jupyterbook.org/en/stable/content/code-outputs.html) and [interactive data visualizations](https://jupyterbook.org/en/stable/interactive/interactive.html). To create and edit Jupyter notebooks the [notebook python package](https://anaconda.org/conda-forge/notebook) can be installed via Conda and then the command *'jupyter notebook'* run in the anaconda prompt if on Windows. However, various options exist and in particular it is recommended to use VS Code and the [jupyter extension](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) that is available.

3. After making adjustments to a file or adding a new file to your local copy of the repository, you'll then want to view how the changes impact/look on the published Jupyter book. To do this first make sure the `_toc.yml` file is suitably updated and then you'll need to [recompile the Jupyter book](https://jupyterbook.org/en/stable/start/build.html), updating the `_build` files. Change directory to one above the cloned repository and execute *'jupyter-book build mybookreponame/'* (if on a Windows machine you can run this command in an Anaconda prompt). To then [preview the changes](https://jupyterbook.org/en/stable/start/build.html#preview-your-built-html) find the `_build/index.html` and double click in your file explorer.

4. Follow the GitHub workflow advice above to contribute your changes to the repository. For the changes to appear on the online published book, this is done via [GitHub pages](https://jupyterbook.org/en/stable/start/publish.html#publish-your-book-online-with-github-pages). Please request this directly by emailing Jeremy Carter at jercar@ceh.ac.uk rather than attempting to do it yourself.   

## Additional Detail:

### 1. Create a Standalone Repository for the Notebook/Method <a id='create-standalone-repository'></a>
- Instead of housing the notebook in the data-science-toolbox repository, it is recommended to create a separate repository for the notebook/method. This helps keep the main repository clean and reduces its file size. The new repository can be created on GitHub and relevant files uploaded. Files might include the notebook itself, any modules that are used in the notebook, images and small data files, the yml file for creating the environment for running the notebook. The repository should be created in the NERC-CEH GitHub organisation and follow the naming convention ds-toolbox-notebook-notebookname (e.g. ds-toolbox-notebook-bias-correction).

### 2. Request Collaborator Access or Fork Repository <a id='access'></a>
- If internal to UKCEH and wanting to contribute regularly to the project then request to become a collaborator on the GitHub repository or email me at jercar@ceh.ac.uk. If external to UKCEH it is currently advised to [create a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo), which is a remote copy of the repository to your personal GitHub account.

> [!TIP]
> Adjustments made in the remote forked repository can then be submitted to main repository on GitHub via a pull request.

### 3. Clone Repository <a id='clone-repository'></a>

- Create a local copy of the repository on your machine by [cloning the repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository?tool=webui) or your forked remote repository respectively.

> [!TIP]
> This local repository tracks the remote repository hosted on GitHub and you can run commands such as '*git fetch origin*' and '*git pull origin*' to update your local copy when the remote repository changes (see [Git fetch and merge](https://longair.net/blog/2009/04/16/git-fetch-and-merge/)).

### 4. Create Issue <a id='create-issue'></a>
- [Create an *'issue'*](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue) in the remote repository detailing what you intend to work on/include. See [here](https://github.com/NERC-CEH/data-science-toolbox/issues) for the current list of issues for the repository. 

> [!TIP]
> Creating an issue allows collaborative work from the community and additionally allows you to add updating comments that link to commits and highlight progress, keeping the community informed. You can assign individuals to join work on specific '*issues*', link them in specific comments and create a link to the respective development branch that work is being conducted on.  

### 5/6. Create Branch <a id='create-branch'></a>
- Create a branch on the remote GitHub page:[data-science-toolbox/branches](https://github.com/NERC-CEH/data-science-toolbox/branches). Naming convention for branches is {yourname}/{branchname} (e.g. jez/bias-correction). Fetch the remote branch to your local machine via '*git fetch origin*' and then create a linked local branch that tracks the remote one via '*git checkout -b branch_name origin/remote_branch_name*'. This also checks out the branch so you can start working on it.
	
> [!TIP] 
> - Branches are spaces to develop code, edit files and make commits without affecting the parent branch (normally labelled *'main'* or *'master'*). 
> - Remote and local branches exist. Remote branches show up on GitHub and to work on them you'll have to create a linked local branch that tracks the remote one.
> - You can see current local branches via '*git branch*' and can see the available remote branches via '*git branch -r*'. If you've created a new remote branch via GitHub you'll need to run either '*git fetch origin*' or '*git pull origin*' to observe it when running '*git branch -r*'. 
> - If you've got a local branch and want to create a remote branch to link to it, this can be done via '*git push -u origin local_branch*'.

### 7. Create a link to your standalone repository using git submodules <a id='create-submodule'></a>.
- Change directory into the methods folder of the data-science-toolbox repository and run:
```bash
git submodule add {url-of-repository}
```
- This creates a submodule within the data-science-toolbox repository that points at the standalone repository housing the notebook/method. This is important as it reduces the total file size of the Jupyter book repository and keeps it clean while also allowing you to include additional files related to your notebook in the external repository, such as modules, images and small data files. Changes to your notebook and relevant files will need to be commited to both the standalone repository and the data-science-toolbox repository.

### 8 . Update the `_toc.yml` file <a id='update-toc'></a>
- The repository contains a basic [anatomy of a jupyter book](https://jupyterbook.org/en/stable/start/create.html#anatomy-of-a-book). The most important files in terms of setting the build options are the table of contents (`_toc.yml`) and book configuration (`_config.yml`) files. It is suggested you will likely not need to edit the `_config.yml` file. However, after creating some content (such as a methodology notebook), for the content to show up in the build, the `_toc.yml` will need updating to include a respective link.  

### 9/10. Render the Jupyter book <a id='render-jupyter-book'></a>
- After making adjustments to a file or adding a new file to your local copy of the repository, you'll then want to view how the changes impact/look on the published Jupyter book. To do this first make sure the `_toc.yml` file is suitably updated and then you'll need to [recompile the Jupyter book](https://jupyterbook.org/en/stable/start/build.html), updating the `_build` files. Change directory to one above the cloned repository and execute *'jupyter-book build data-science-toolbox/'* (if on a Windows machine you can run this command in an Anaconda prompt). To then [preview the changes](https://jupyterbook.org/en/stable/start/build.html#preview-your-built-html) find the `_build/index.html` and double click in your file explorer.

### 11. Pull Changes into Main Branch <a id='pull-changes'></a>
- Work on your local branch, making commits at regular intervals. Making regular commits for each specific sub-task within your development goal is a good idea and helps when reviewing the changes ([the right time to commit and branch](https://blog.scottlogic.com/2019/12/19/source-control-when.html#:~:text=Generally%20create%20a%20branch%20for,wherever%20it%20needs%20to%20go.)). 
> [!TIP] 
> If the local branch is linked to a remote branch you can push your changes there periodically using '*git push origin branch_name*'. This is useful if collaborating on a development task. If other collaborators have pushed changes to the remote branch so that is ahead of your local branch, then you can absorb these changes either by using '*git pull origin*' or doing it in stages using '*git fetch origin*' and then '*git checkout branch_name*' and '*git merge origin/branch_name*'. Note if your branch falls behind changes made to the main remote branch, these can be merged in using '*git merge origin/main*'.

- If you're happy with how the notebook looks and all the relevant the work for the issue created is complete then you can submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) on GitHub to ask collaborators for feedback on the changes and to hopefully merge changes into the main repository branch. 

> [!TIP] 
>- Taken directly from [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow): 
>>*When you create a pull request, include a summary of the changes and what problem they solve. You can include images, links, and tables to help convey this information. If your pull request addresses an issue, link the issue so that issue stakeholders are aware of the pull request and vice versa. If you link with a keyword, the issue will close automatically when the pull request merges. For more information, see "[Basic writing and formatting syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)" and "[Linking a pull request to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)."*
>- Reviewers will typically leave comments/suggestions on the pull request and these can be addressed, see "[Reviewing changes in pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests).
>- Once the pull request is approved the development branch can be merged with the main repository branch.  

### 12. Delete Branch and Close Issue <a id='close-issue'></a>
- After merging the given development branch with the main repository branch, the development branch can be deleted. The commit history of the development branch will be transferred to the main branch and a commit specific to the pull request will remain.

> [!TIP] 
> - The remote development branch can be [deleted on GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository#deleting-a-branch). Then to no longer see the remote branch on your local machine you'll need to run '*git fetch origin -p*', which prunes the branches. To delete a local branch that tracked the remote branch we can do *'git branch -d branch-name'*.
