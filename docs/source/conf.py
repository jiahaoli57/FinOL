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

extensions = [
    # "recommonmark",
    # "sphinx_markdown_tables",
    "sphinx_rtd_theme",
    "sphinx_copybutton"
]

templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = []


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"  # 'alabaster'
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ['_static']
html_css_files = ["css/custom.css"]



# -- Extension configuration -------------------------------------------------

# sphinx_copybutton option to not copy prompt.
copybutton_prompt_text = "$ "
