from ast import literal_eval
from distutils.core import setup


def get_version(source='src/pyfurcation/__init__.py'):

    with open(source) as f:
        for line in f:
            if line.startswith('__version__'):
                return literal_eval(line.split('=')[-1].lstrip())
        raise ValueError("__version__ not found")

setup(
    name='anhima',
    version=get_version(),
    author='Richard Pearson',
    author_email='rpearson@well.ox.ac.uk',
    package_dir={'': 'src'},
    packages=['pyfurcation'],
    url='https://github.com/hardingnj/pyfurcation',
    license='MIT License',
    description='Production of haplotype bifurcation plots.',
    long_description=open('README.md').read(),
    classifiers=['Intended Audience :: Developers',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Topic :: Software Development :: Libraries :: Python Modules'])