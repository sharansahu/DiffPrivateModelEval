# source_python_classes.R
library(reticulate)
library(here)

# Set up the Python environment
setup_python_environment <- function(envname = "r-reticulate") {
  if (!reticulate::virtualenv_exists(envname)) {
    message("Creating virtual environment: ", envname)
    reticulate::virtualenv_create(envname)
  }
  reticulate::virtualenv_install(envname, packages = c("numpy", "matplotlib"), ignore_installed = TRUE)
  reticulate::use_virtualenv(envname, required = TRUE)
}

# Generic function to source Python files and check for class definitions
source_python_class <- function(python_file, class_name) {
  setup_python_environment()  # Ensure the environment is set up

  # Check if the Python file exists
  python_file_path <- here::here("inst/python", python_file)
  if (!file.exists(python_file_path)) {
    stop(paste("Python file", python_file, "not found in 'inst/python/'."))
  }

  # Source the Python file
  message(paste("Sourcing Python file:", python_file_path))
  reticulate::source_python(python_file_path)
  message("Python file sourced successfully.")

  # Check if the class is defined
  if (!reticulate::py_has_attr(reticulate::import_main(), class_name)) {
    stop(paste(class_name, "class not found after sourcing the Python file."))
  }
  message(paste(class_name, "class successfully loaded."))
}
