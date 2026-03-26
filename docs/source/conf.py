import sheet_1

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Quantenschaltung'
copyright = '2026, Alina und Paul'
author = 'Alina und Paul'
release = '28.03.2026'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autosummary', 'nbsphinx', 'myst_nb']

nb_execution_mode = 'off'

templates_path = ['_templates']
exclude_patterns = []

language = 'idx = np.argmax(np.abs(aer_state))'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
