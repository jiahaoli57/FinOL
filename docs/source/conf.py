# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


from datetime import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FinOL'
# copyright = f'2024–{datetime.now().year}, FinOL contributors'
copyright = f'2024, FinOL contributors'
author = 'FinOL contributors'
release = 'MIT'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = ["recommonmark", "sphinx_markdown_tables", "sphinx_rtd_theme"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"  # 'alabaster'

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_static_path = ['_static']
