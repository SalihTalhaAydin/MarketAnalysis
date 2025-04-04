# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MarketAnalysis Framework"
copyright = "2025, Your Name"  # Update with actual year/name
author = "Your Name"  # Update with actual name
# The short X.Y version
# version = '0.1' # You might want to get this from setup.py or __init__.py
# The full version, including alpha/beta/rc tags
# release = '0.1.0' # You might want to get this from setup.py or __init__.py

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.githubpages",  # Helps with GitHub Pages deployment
    # "myst_parser", # Removed as all files are .rst now
]

templates_path = ["_templates"]
exclude_patterns = []

# The suffix(es) of source filenames. Default is .rst
# source_suffix = ['.rst'] # Removed explicit setting

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# -- Options for MyST Parser -------------------------------------------------
# (Removed as myst_parser extension is removed)
# myst_enable_extensions = [...]
# myst_heading_anchors = 3

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class name.
autodoc_class_signature = "separated"

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# Add path to the project's source code for autodoc
# sys.path.insert(0, os.path.abspath(".")) # Removed adding source dir itself
sys.path.insert(0, os.path.abspath("../../"))  # Point to the project root
# sys.path.insert(0, os.path.abspath('../../market_ml_model')) # Not needed if root is added
