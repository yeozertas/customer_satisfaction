from setuptools import setup

setup(
    name='customer_satisfaction',
    version='0.1.0',
    description='Customer Satisfaction Meter',
    author='Yunus Emre Ozertas',
    author_email='yunusemreozertas@yahoo.com.tr',
    install_requires=[
        'transformers==4.29.2',
        'torch==2.0.1',
        'numpy==1.24.3',
        'pyyaml'
    ],

    package_data={'': ['*.txt', '*.json', '*.bin', '*.yaml']},
)
