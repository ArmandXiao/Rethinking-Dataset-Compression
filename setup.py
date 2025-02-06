from setuptools import setup, find_packages

setup(
    name='rethinkdc',
    version='0.3.2',
    description='Rethinkinig Large-scale Dataset Compression',
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    author='Lingao Xiao',
    author_email='xiao_lingao@u.nus.edu',
    include_dirs=['rethinkdc'],
    package_data={
        'rethinkdc': ['dc_config/*.yaml'],
    },
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',  # More flexible requirement
        'torchvision>=0.15.0',
        'torchaudio>=2.0.0',
        'numpy>=2.0',
        'wandb>=0.19',
        'rich>=13.9',
        'huggingface-hub>=0.28',
        'datasets>=3.2',
        'pyyaml>=6.0'
    ],
    entry_points={
        'console_scripts': [
            'rethinkdc=rethinkdc.train_KD:main',
        ],
    },
    license='MIT',
    classifiers=[ 
            'Development Status :: 3 - Alpha', 
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    keywords='dataset compression, deep learning, machine learning',  # Add keywords
    python_requires='>=3.10',
)