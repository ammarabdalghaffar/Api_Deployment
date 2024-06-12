from setuptools import setup, find_packages

setup(
    name='knn_api',
    version='0.1.0',
    description='A simple KNN API',
    packages=find_packages(),
    install_requires=[
        'Flask>=1.1.1',
        'scikit-learn>=0.22.0',
        'numpy>=1.18.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
