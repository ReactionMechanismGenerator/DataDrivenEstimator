import os
from distutils.core import setup

scripts = ['scripts/train_cnn.py',
           'scripts/evaluate_cnn.py']

modules = []
for root, dirs, files in os.walk('dde'):
    if 'test_data' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            if 'test' not in file and '__init__' not in file:
                module = 'dde' + root.partition('dde')[-1].replace('/','.') + '.' + file.partition('.py')[0]
                modules.append(module) 

setup(
    name='DataDrivenEstimator',
    version='1.0.0',
    packages=['dde'],
    description='Data Driven Estimator',
    author='Kehang Han, Yi-Pei Li, Colin Grambow',
    author_email='rmg_dev@mit.edu',
    py_modules=modules,
    scripts=scripts
)
