from setuptools import setup, find_packages

setup(
    name='wm',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        # e.g., 'numpy', 'pandas', 'torch', etc.
    ],
    author='Yilin Wu',
    author_email='yilin-wu@outlook.com',
    description='A package for Failure Prediction with World Model and VLM.',
    # url='https://github.com/yourusername/your-repo',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)