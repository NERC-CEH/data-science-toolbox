🌞 Thanks for being interested in contributing to the UKCEH data science toolbox! 

Here's some useful information for getting started... 🌱

## GitHub Workflow for Contributors:
The suggested workflow for contributing to this repository is taken from [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow), general steps involve:

1. Examine the remote repository on GitHub and look under the various tabs such as '*Issues*'  to understand current development paths.  Examine what remote branches exist. 
	
> [!TIP] 
> - Branches are spaces to develop code, edit files and make commits without affecting the parent branch (normally labelled *'main'* or *'master'*). 
> - Remote and local branches exist. Remote branches show up on GitHub and to work on them you'll have to create a linked local branch, which you push changes from (described later). Remote branches are normally connected with specific developments (e.g. adding a Gaussian process notebook...) and they allow for collaboration. 

2. If external to UKCEH and only expecting to have minor input to the project then [create a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo), which is a remote copy of the repository to your personal GitHub account. If internal to UKCEH and wanting to contribute regularly to the project then request to become a collaborator on the GitHub repository or email me at jercar@ceh.ac.uk .

> [!TIP] 
> Adjustments made in the remote forked repository can then be submitted to main repository on GitHub via a pull request.

3. [Clone the repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository?tool=webui) or your forked remote repository respectively, which creates a local copy of the files on your machine. 
	- This local repository tracks the remote repository hosted on GitHub and you can run commands such as '*git fetch origin*' and '*git pull origin*' to update your local copy when the remote repository changes (see [Git fetch and merge](https://longair.net/blog/2009/04/16/git-fetch-and-merge/)).

4. Create a branch either locally ('*git branch new_branch_name*') or on the remote GitHub page ([creating branch via GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository)). It is recommended to use local branches for just playing around with the code and remote branches for specific development goals that you hope to share with the community. Ideally, to keep remote branches from deviating too much from the main branch, their scope should remain focused and relatively short term.  

> [!TIP] 
> - To work locally on remote branches you need to create a local branch that tracks the remote one via '*git checkout -b branch_name origin/remote_branch_name*'. 
> - You can see current local branches via '*git branch*' and can see the available remote branches via '*git branch -r*'. If you've created a new remote branch via GitHub you'll need to run either '*git fetch origin*' or '*git pull origin*' to observe it when running '*git branch -r*'. 
> - If you've got a local branch and want to create a remote branch to link to it, this can be done via '*git push -u origin local_branch*'.

> [!IMPORTANT] 
> It's highly recommended to [create an *'issue'*](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-an-issue) for any development work, which details what you're working on. See [here](https://github.com/Jez-Carter/UKCEH_Data_Science_Book/issues) for the current list of issues for the repository. Doing this allows early feedback from the community on the idea and additionally allows you to add updating comments that link to commits and highlight progress, keeping the community informed. You can assign individuals to join work on specific '*issues*', link them in specific comments and create a link to the respective development branch that work is being conducted on.   

5. Work on your local branch, making commits at regular intervals. Making regular commits for each specific sub-task within your development goal is a good idea and helps when reviewing the changes ([the right time to commit and branch](https://blog.scottlogic.com/2019/12/19/source-control-when.html#:~:text=Generally%20create%20a%20branch%20for,wherever%20it%20needs%20to%20go.)). 
> [!TIP] 
> If the local branch is linked to a remote branch you can push your changes there periodically using '*git push origin branch_name*'. This is useful if collaborating on a development task. If other collaborators have pushed changes to the remote branch so that is ahead of your local branch, then you can absorb these changes either by using '*git pull origin*' or doing it in stages using '*git fetch origin*' and then '*git checkout branch_name*' and '*git merge origin/branch_name*'. 

6. If working on a specific development goal then after the work is complete and pushed to the specific remote development branch you can submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) on GitHub to ask collaborators for feedback on the changes and to hopefully merge changes into the main repository branch. 

> [!TIP] 
>- Taken directly from [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow): 
>>*When you create a pull request, include a summary of the changes and what problem they solve. You can include images, links, and tables to help convey this information. If your pull request addresses an issue, link the issue so that issue stakeholders are aware of the pull request and vice versa. If you link with a keyword, the issue will close automatically when the pull request merges. For more information, see "[Basic writing and formatting syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)" and "[Linking a pull request to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)."*
>- Reviewers will typically leave comments/suggestions on the pull request and these can be addressed, see "[Reviewing changes in pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests).
>- Once the pull request is approved the development branch can be merged with the main repository branch.  

7. After merging the given development branch with the main repository branch, the development branch can be deleted. The commit history of the development branch will be transferred to the main branch and a commit specific to the pull request will remain. 