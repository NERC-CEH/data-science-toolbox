# Including R Methods

For methods developed in `R`, documentation will often take the form of an `R Markdown` document, `Quarto` document, or package `vignette`. For inclusion in the Environmental Data Science Toolbox, however, methods are presented as Jupyter `.ipynb` notebooks, which can be executed using an R kernel. The approach is used so methods developed in `R`, `Python` and other languages are all visible in the same interface, promoting cross-language method discovery and knowledge sharing.

For methods developed in R the normal approach to documenting the method is to have an rmarkdown or vignette. For inclusion into the toolbox we need an Jupyter .ipynb file, which can be run with an R kernel. The approach is used to allow having methods developed inR and Ypthon and other languages all visible in a similar interface, promoting cross-language method discovery and knowledge sharing.

The guidance below describes how to make an R-based method available as a runnable Jupyter notebook, including the required R and Jupyter setup, dependency management, and registration of an R kernel.

## Requirements

Methods developed in **R** can be included in the Environmental Data Science Toolbox as `.ipynb` notebooks by using an **R Jupyter kernel**.

Before setting up the notebook, make sure you have:
- A working installation of R.
- A working installation of Jupyter.
- Jupyter available from the R session you will use to register the kernel.
- A way of opening and running Jupyter notebooks, such as VS Code with the Jupyter extension, JupyterLab, or another Jupyter-compatible interface.

> [!TIP]
> You can check whether Jupyter is visible from within an R terminal with: `Sys.which("jupyter")`. This should return the path to a Jupyter executable. If it returns an empty string, Jupyter is either not installed or is not available on the PATH inherited by the R session. You can make Jupyter accessible by first activating a UV or Conda environment with Jupyter installed before starting R: `conda activate <jupyter-environment>`.

## 1. Install the package and/or dependencies.
Start an R session and install the required package/dependencies for running the notebook. The exact installation approach may vary between projects. The important requirement is that all packages used by the notebook are available in the R session. 

For reproducibility, best practice is to include a `DESCRIPTION` file or to use `renv`. This will easily allow users to recreate the necessary environment for running the notebook.

## 2. Install and register an R Jupyter kernel.

Register the R environment you're in as an R kernel for use by the notebook. To do this we use `IRkernel`, which you'll have to install through `install.packages('IRkernel')`.

Register the R kernel:

```r
IRkernel::installspec(
    name = "notebookname-r",
    displayname = "R - notebookname"
)
```

This is normally a one-off setup step on each machine.

## 3. Create an .ipynb file and select the R kernel.

While several approaches exist for converting R markdown/quarto documents into Jupyter .ipynb notebooks, we recommend simply starting from fresh with the following [notebook template](https://github.com/NERC-CEH/data-science-toolbox/blob/main/notebook_guidance/template_notebook.ipynb) and copying across code/text. Ultimately, this is more efficient than troubleshooting issues with a direct conversion and it allows you to test the cells all run as expected. 

Open the [notebook template](https://github.com/NERC-CEH/data-science-toolbox/blob/main/notebook_guidance/template_notebook.ipynb) file in a interface such as VS Code with the Jupyter extension, JupyterLab, or another Jupyter-compatible interface. Select the R kernel registered in the previous step:
**Select Kernel → Jupyter Kernel → R - notebookname**

Notebooks use markdown(.md) and R code cells. Include general descriptive text in the markdown cells and include 
R code in specific R code cells. You can see and select the type of cell in the bottom right of the contents.

An `.ipynb` code cell should contain normal R code:

```r
fname <- here::here("inst", "extdata", "data.csv")
dt <- data.table::fread(fname)
```

Do not include R Markdown or Quarto chunk fences such as: *```{r}*.

## 6. Run the notebook cells and test from a clean clone

Before including the notebook in the Toolbox, test the complete workflow from a fresh clone:

```text
clone repository
→ install R dependencies
→ install the local R package
→ register IRkernel
→ open the notebook
→ select the R kernel
→ run all cells
```

This checks that the notebook does not depend on packages or configuration that are only present on the original developer's machine.
