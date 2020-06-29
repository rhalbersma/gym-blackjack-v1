#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from setuptools import setup, find_packages

setup(
    name='gym_blackjack_v1',
    version='0.1.0-dev0',
    description='OpenAI Gym blackjack environment (v1)',
    url='https://github.com/rhalbersma/blackjack',
    author='Rein Halbersma',
    author_email='rhalbersma@gmail.com',
    license='Boost Software License 1.0 (BSL-1.0)',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gym', 'numpy', 'setuptools', 'statsmodels', 'tqdm', 'wheel'
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha'
        'Intended Audience :: Science/Research'
        'License :: OSI Approved :: Boost Software License 1.0 (BSL-1.0)'
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
