options(repos = c(CRAN = "https://cloud.r-project.org"))

packages <- c(
  "ggplot2",
  "dplyr",
  "tidyr",
  "readr",
  "tidyverse",
'knitr',
'ggthemes',
'ggrepel',
'dslabs'
)

missing <- packages[!packages %in% installed.packages()[, "Package"]]

if (length(missing) > 0) {
  install.packages(missing)
}