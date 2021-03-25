'''
Created on March 25, 2021

@author: Abhijeet Gupta

NOTE: To create a package run the following line on the terminal:
    python3 setup.py develop
'''

try: 
    from setuptools import setup, find_packages
except:
    from distutils.core import setup, find_packages
        
version = '0.1'

base_requirements = [
            'nose',
            'nosexcover',
            'argparse',
        ]

setup(name='ag_lib',
    version=version,
    description="DEEP LINGUISTIC MODELLING - A01",

    # The classifiers and keywords to apply to this package e.g. on PyPi
    classifiers=[
        "Programming Language :: Python3.7.3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords='ag_lib',

    # Details about the author, license and origin of this package
    author='Abhijeet Gupta',
    author_email='abhijeet.gupta@gmail.com',
    url='http://abhijeetgupta.info',
    license='GPL 3.0',

    # Some technical details that will help setuptools
    # resp. distutils to deal appropriately with this package
    packages=find_packages(where='src', exclude=[]),
    package_dir={'': 'src'},
    namespace_packages=['modules'],
    zip_safe=False,

    # Define what static files and resource need to be included as well
    package_data={
        # All files in the resource directory
        'src.modules': ['resources/*.*'],

    },

    # These are the dependencies that we require to be installed before the
    # package is safe to be used
    
    install_requires=base_requirements,

    # Would like to implement testing eventually through nose tools
    test_suite='nose.collector',

    # The entry points of this package (safe to leave empty)
    entry_points="""
    """,
    )
