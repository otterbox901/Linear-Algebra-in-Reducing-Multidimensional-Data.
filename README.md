# Linear Algebra in Reducing Multidimensional Data.

A short report and analysis comparing Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) for Iris and MNIST data. The repository contains the R code used to run the analyses and the TeX source for the manuscript/report.

## Overview

This project evaluates how PCA and LDA perform for dimensionality reduction and classification tasks on datasets typically used in ML. The analysis scripts produce summary statistics, figures, and the results included in the manuscript source (TeX).

Goals:
- Compare PCA and LDA in terms of explained variance, separability, and classification performance.
- Provide reproducible R scripts to run the analyses and regenerate the figures and tables.
- Produce a manuscript (PDF) using the included TeX source.


## Requirements

- R >= 4.0
- TeX distribution (TeX Live, MiKTeX, or MacTeX) to compile the manuscript
- Suggested R packages:
  - tidyverse
  - ggplot2
  - MASS (for lda)
  - stats (prcomp)
  - caret (for classification evaluation)
  - knitr / rmarkdown (if there are R Markdown files)
Install packages in R with:
```r
install.packages(c("tidyverse","ggplot2","MASS","caret","knitr","rmarkdown"))
