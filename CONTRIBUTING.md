🌞 Thanks for being interested in contributing to the UKCEH data science toolbox! 

Here's some useful information for getting started... 🌱

[**GitHub Workflow**](#github-workflow-for-contributors)
| [**Jupyter Book Workflow**](#jupyter-book-workflow-for-contributors)

## GitHub Workflow for Contributors:
The suggested workflow for contributing to this repository is taken from [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow), general steps involve:

[**1. Examine Current Issues**](#examine-current-issues)
| [**2. Fork/Request Access**](#fork)
| [**3. Clone Repository**](#clone-repository)
| [**4. Create Issue**](#create-issue)
| [**5. Create Branch**](#create-branch)
| [**6. Create Submodule**](#create-submodule)
| [**7. Commit Changes**](#commit-changes)
| [**8. Pull Changes**](#pull-changes)
| [**9. Close Issue**](#close-issue)

The [**Create Submodule**](#create-submodule) advice is specific to contributors wanting to incorporate notebooks. 

### 1. Examine Current Issues <a id='examine-current-issues'></a>
- Examine the remote repository on GitHub and look under the various tabs such as '*Issues*'  to understand current development paths.  

### 2. Request Collaborator Access or Fork Repository <a id='fork'></a>
- If external to UKCEH and only expecting to have minor input to the project then [create a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo), which is a remote copy of the repository to your personal GitHub account. If internal to UKCEH and wanting to contribute regularly to the project then request to become a collaborator on the GitHub repository or email me at jercar@ceh.ac.uk .

> [!TIP] 
> Adjustments made in the remote forked repository can then be submitted to main repository on GitHub via a pull request.

### 3. Clone repository <a id='clone-repository'></a>

- [Clone the repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository?tool=webui) or your forked remote repository respectively, which creates a local copy of the files on your machine. 
- This local repository tracks the remote repository hosted on GitHub and you can run commands such as '*git fetch origin*' and '*git pull origin*' to update your local copy when the remote repository changes (see [Git fetch and merge](https://longair.net/blog/2009/04/16/git-fetch-and-merge/)).

### 4. Create Issue <a id='create-issue'></a>
- [Create an *'issue'*](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue) for any development work, which details what you're working on. See [here](https://github.com/Jez-Carter/UKCEH_Data_Science_Book/issues) for the current list of issues for the repository. Doing this allows early feedback from the community on the idea and additionally allows you to add updating comments that link to commits and highlight progress, keeping the community informed. You can assign individuals to join work on specific '*issues*', link them in specific comments and create a link to the respective development branch that work is being conducted on.   

### 5. Create Branch <a id='create-branch'></a>
- Create a branch either locally ('*git branch new_branch_name*') or on the remote GitHub page ([creating branch via GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository)). A remote branch can be created directly from the 'issue' you created, which also links it. Naming convention for branches is {yourname}/{branchname} (e.g. jez/bias-correction). It is recommended to use local branches for just playing around with the code and remote branches for specific development goals that you hope to share with the community. Ideally, to keep remote branches from deviating too much from the main branch, their scope should remain focused and relatively short term. Make sure you have ['*checked out*'](https://www.atlassian.com/git/tutorials/using-branches/git-checkout) the local branch before starting work on it, this can be done via *'git checkout branch_name'*. 
	
> [!TIP] 
> - Branches are spaces to develop code, edit files and make commits without affecting the parent branch (normally labelled *'main'* or *'master'*). 
> - Remote and local branches exist. Remote branches show up on GitHub and to work on them you'll have to create a linked local branch that tracks the remote one via '*git checkout -b branch_name origin/remote_branch_name*'.   
> - You can see current local branches via '*git branch*' and can see the available remote branches via '*git branch -r*'. If you've created a new remote branch via GitHub you'll need to run either '*git fetch origin*' or '*git pull origin*' to observe it when running '*git branch -r*'. 
> - If you've got a local branch and want to create a remote branch to link to it, this can be done via '*git push -u origin local_branch*'.

### 6. Create Submodule for Notebook Repository <a id='create-submodule'></a>
- If intending to add a notebook into the Jupyter book, store the notebook in a separate external repository and [create a submodule](https://gist.github.com/gitaarik/8735255) within your working branch of the Jupyter book repository that points at this location. This is important as it reduces the total file size of the Jupyter book repository and keeps it clean while also allowing you to include additional files related to your notebook in the external repository, such as modules, images and small data files. 

> [!TIP] 
> - An option for the external repository housing the notebook, is to create a branch named 'jupyterbook-render' from the repository that contains the code used in the notebook.
> - To add the submodule pointing at the particular branch of another repository the following command can be used: '*git submodule add -b {branch-name} {url-of-repository}*'. As an example: '*git submodule add -b jupyterbook_render https://github.com/Jez-Carter/Bias_Correction_Application.git*'. Do this after changing directory to the correct folder of the Jupyter book repository, so probably /notebooks/methods/.  
> - Work on the notebook in the external repository and then pull in changes to the Jupyter book repository by changing directory into the submodule folder and running '*git pull origin*'. When you're in the directory of the submodule, git should automatically be checked-out on the branch of your external repository holding the notebook.
> - After pulling changes from the external repository, the changes can be committed to your branch on the Jupyter book repository. Additionally, you can build the Jupyter book (see: [**Jupyter Book Workflow**](#jupyter-book-workflow-for-contributors)) to see how the notebook renders in that environment.

### 7. Make Changes and Commit <a id='commit-changes'></a>
- Work on your local branch, making commits at regular intervals. Making regular commits for each specific sub-task within your development goal is a good idea and helps when reviewing the changes ([the right time to commit and branch](https://blog.scottlogic.com/2019/12/19/source-control-when.html#:~:text=Generally%20create%20a%20branch%20for,wherever%20it%20needs%20to%20go.)). 
> [!TIP] 
> If the local branch is linked to a remote branch you can push your changes there periodically using '*git push origin branch_name*'. This is useful if collaborating on a development task. If other collaborators have pushed changes to the remote branch so that is ahead of your local branch, then you can absorb these changes either by using '*git pull origin*' or doing it in stages using '*git fetch origin*' and then '*git checkout branch_name*' and '*git merge origin/branch_name*'. Note if your branch falls behind changes made to the main remote branch, these can be merged in using '*git merge origin/main*'.

### 8. Pull Changes into Main Branch <a id='pull-changes'></a>
- If working on a specific development goal then after the work is complete and pushed to the specific remote development branch you can submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) on GitHub to ask collaborators for feedback on the changes and to hopefully merge changes into the main repository branch. 

> [!TIP] 
>- Taken directly from [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow): 
>>*When you create a pull request, include a summary of the changes and what problem they solve. You can include images, links, and tables to help convey this information. If your pull request addresses an issue, link the issue so that issue stakeholders are aware of the pull request and vice versa. If you link with a keyword, the issue will close automatically when the pull request merges. For more information, see "[Basic writing and formatting syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)" and "[Linking a pull request to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)."*
>- Reviewers will typically leave comments/suggestions on the pull request and these can be addressed, see "[Reviewing changes in pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests).
>- Once the pull request is approved the development branch can be merged with the main repository branch.  

### 9. Delete Branch and Close Issue <a id='close-issue'></a>
- After merging the given development branch with the main repository branch, the development branch can be deleted. The commit history of the development branch will be transferred to the main branch and a commit specific to the pull request will remain.

> [!TIP] 
> - The remote development branch can be [deleted on GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository#deleting-a-branch). Then to no longer see the remote branch on your local machine you'll need to run '*git fetch origin -p*', which prunes the branches. To delete a local branch that tracked the remote branch we can do *'git branch -d branch-name'*.


## Jupyter Book Workflow for Contributors:

A [Jupyter Book](https://jupyterbook.org/en/stable/intro.html) is an organised collection of [Jupyter Notebooks](https://www.dataquest.io/blog/jupyter-notebook-tutorial/) that cover a specific topic. This repository houses the 'UKCEH Data Science Toolbox' Jupyter book, which presents a collection of Jupyter notebook tutorials on specific data science methodologies developed at UKCEH, detailing the full data science pipeline. Here we provide a guideline on developing the Jupyter book's content and suggest useful links and documentation. See this [Jupyter Books 101](https://www.google.com/search?sca_esv=853f175af13f0422&rlz=1C1GCEA_enGB1127GB1127&sxsrf=ADLYWILIDB_FKqa2tEu-BFTAyFkn4C5pZA:1730195044702&q=ghp-import&tbm=vid&source=lnms&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWfbQph1uib-VfD_izZO2Y5sC3UdQE5x8XNnxUO1qJLaQdh3mUfgbiNAX47iHD_lJjnnrtkrknsy6VQXK4-aRHxqnPwuFZlmbREdWKLZFI-gq_UsBLTYJHKqEeHaFb3F8RYl5naC8STX8rrrXVJLtrqfmiz5ev1aurnZdmKum0bTFGUA16&sa=X&ved=2ahUKEwiYhKDoprOJAxWsVkEAHbmtBuoQ0pQJegQIGhAB&biw=1745&bih=828&dpr=1.1&safe=active&ssui=on#fpstate=ive&vld=cid:6619f956,vid:lZ2FHTkyaMU,st:0) for a nice YouTube introduction into the topic. 

> [!NOTE]  
>To be able to re-build the Jupyter book and view your adjustments you'll need to download the Python package, either using PIP ([jupyter-book PyPI](https://pypi.org/project/jupyter-book/)) or Conda ([jupyter-book conda-forge](https://anaconda.org/conda-forge/jupyter-book)), see [Install Jupyter Book](https://jupyterbook.org/en/stable/start/overview.html). It is suggested to use Conda and to run the commands via the Anaconda Prompt if on a windows machine.  

1. The repository contains a basic [anatomy of a jupyter book](https://jupyterbook.org/en/stable/start/create.html#anatomy-of-a-book). The most important files in terms of setting the build options are the table of contents (`_toc.yml`) and book configuration (`_config.yml`) files. It is suggested you will likely not need to edit the `_config.yml` file. However, after creating some content (such as a methodology notebook), for the content to show up in the build, the `_toc.yml` will need updating to include a respective link.  

2. The main type of content files include [markdown](https://jupyterbook.org/en/stable/start/create.html#markdown-files-md) (.md) files and [jupyter notebooks](https://jupyterbook.org/en/stable/start/create.html#jupyter-notebooks-ipynb) (.ipynb) files.
- Markdown files are generally for contextual information (e.g. the repository's landing README.md page). There are various good options for getting the hand of [markdown file syntax](https://www.markdownguide.org/basic-syntax/). One option is to download notemaker apps such as [Obsidian](https://obsidian.md/) or to use the markdown preview option in [VS Code](https://code.visualstudio.com/docs/languages/markdown). It's worth noting that Jupyter book allows a slightly different collection of markdown syntax ([common and MyST flavours](https://jupyterbook.org/en/stable/start/create.html#anatomy-of-a-book)) compared to software such as Obsidian, so if something is not rending correctly then this is probably why. Another good place to pickup the syntax is from contextual files for the Jupyter book in the repository, such as the `intro.md` file that determines content for the landing page of the book.
- Jupyter notebooks are generally for mixing code cells with contextual writing. They're a very good way of explaining complex data science workflows in a step-by-step approach. Notebooks can be run in R, Python and Julia. See the following [methodology notebook template](https://jez-carter.github.io/UKCEH_Data_Science_Book/notebooks/methods/template.html) document for some design ideas, as well as official documentation for [formatting code outputs](https://jupyterbook.org/en/stable/content/code-outputs.html) and [interactive data visualizations](https://jupyterbook.org/en/stable/interactive/interactive.html). To create and edit Jupyter notebooks the [notebook python package](https://anaconda.org/conda-forge/notebook) can be installed via Conda and then the command *'jupyter notebook'* run in the anaconda prompt if on Windows. However, various options exist and in particular it is recommended to use VS Code and the [jupyter extension](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) that is available.

3. After making adjustments to a file or adding a new file to your local copy of the repository, you'll then want to view how the changes impact/look on the published Jupyter book. To do this first make sure the `_toc.yml` file is suitably updated and then you'll need to [recompile the Jupyter book](https://jupyterbook.org/en/stable/start/build.html), updating the `_build` files. Change directory to one above the cloned repository and execute *'jupyter-book build mybookreponame/'* (if on a Windows machine you can run this command in an Anaconda prompt). To then [preview the changes](https://jupyterbook.org/en/stable/start/build.html#preview-your-built-html) find the `_build/index.html` and double click in your file explorer.

4. Follow the GitHub workflow advice above to contribute your changes to the repository. For the changes to appear on the online published book, this is done via [GitHub pages](https://jupyterbook.org/en/stable/start/publish.html#publish-your-book-online-with-github-pages). Please request this directly by emailing Jeremy Carter at jercar@ceh.ac.uk rather than attempting to do it yourself.   
