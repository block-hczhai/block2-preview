# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import subprocess

# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'block2'
copyright = '2020-2024, Huanchen Zhai'
author = 'Huanchen Zhai'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "nbsphinx",
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
]

breathe_projects = {"block2": "../build/doxygenxml/"}
breathe_default_project = "block2"
breathe_domain_by_extension = {"hpp": "cpp"}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_theme_options = { 'logo_only': True, }
html_logo = '_static/block2-logo-white.png'

primary_domain = "cpp"
highlight_language = "cpp"

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'preamble': r'''
        \usepackage{charter}
        \usepackage[defaultsans]{lato}
        \usepackage{inconsolata}
    '''
}

sys.path[:0] = [os.path.abspath('../..')]

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__init__'
}

def generate_doxygen_xml(app):
    build_dir = os.path.join(app.confdir, "build")
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)

    try:
        subprocess.call(["doxygen --version"], shell=True)
        retcode = subprocess.call(["mkdir ../build"], cwd=app.confdir, shell=True)
        retcode = subprocess.call(["doxygen ../Doxygen"], cwd=app.confdir, shell=True)
        if retcode < 0:
            sys.stderr.write("doxygen error code: {}\n".format(-retcode))
    except OSError as e:
        sys.stderr.write("doxygen execution failed: {}\n".format(e))

def setup(app):
    # Add hook for building doxygen xml when needed
    app.connect("builder-inited", generate_doxygen_xml)
