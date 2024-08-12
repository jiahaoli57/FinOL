# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('../../..'))
# sys.path.insert(0, os.path.abspath('../../finol'))
# sys.path.insert(0, os.path.abspath('../../finol/data_layer'))
# sys.path.insert(0, os.path.abspath('../../finol/data_layer/'))
import finol
from finol import data_layer, model_layer, optimization_layer, evaluation_layer
from finol import utils
# from finol.data_layer import dataset_loader
# from finol.data_layer.dataset_loader import DatasetLoader

# import warnings
# import plotly.io as pio
# from datetime import datetime
# from sphinx_gallery.sorting import FileNameSortKey

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FinOL"
# copyright = f'2024–{datetime.now().year}, {finol.__author__}'
copyright = f"2024, {finol.__author__}"
author = f"{finol.__author__}"
release = "MIT"
# The short X.Y version
version = finol.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # "sphinx.ext.githubpages",
    # "recommonmark",
    # "sphinx_markdown_tables",
    # "sphinx_rtd_theme",
    "sphinx_copybutton",
    # "sphinx_design",
    # "shibuya",
    # "sphinx-jsonschema",
    "sphinxemoji.sphinxemoji",
]
templates_path = ["_templates"]

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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# import sphinx_rtd_theme
# html_theme = "sphinx_rtd_theme"  # alabaster shibuya
html_theme = "shibuya"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "color_mode": "dark",
    "github_url": "https://github.com/jiahaoli57/finol",
    "nav_links": [
        {"title": "Ask GPT-4o", "url": "https://docsbot.ai/chat/yXPhXS1d2dkpcN1wITvD/yJzHla8z2KfIinBnqpzV"},
    ]
}

html_favicon = "../images/finol_logo_icon.png"
html_logo = "../images/finol_logo_pure.png"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
# html_css_files = ["css/optuna.css"]

html_copy_source = False
html_show_sourcelink = False

language = 'en'

html_context = {}

# -- Extension configuration -------------------------------------------------
autosummary_generate = True
autosummary_ignore_module_all = False  # ignore nn.Module if False
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "exclude-members": "with_traceback",
}
# autodoc_mock_imports = ["finol"]
autosummary_imported_members = True

# sphinx_copybutton option to not copy prompt.
copybutton_prompt_text = "$ "

