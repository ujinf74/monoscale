from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'monoscale_evaluation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ujin Kwon',
    maintainer_email='ujin@tukorea.ac.kr',
    description='Scores a run against ground truth. Not carried on the vehicle.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'odometry_evaluator = monoscale_evaluation.odometry_evaluator:main',
            'grid_evaluator = monoscale_evaluation.grid_evaluator:main',
            'benchmark = monoscale_evaluation.benchmark:main',
            'bag_gate = monoscale_evaluation.bag_gate:main',
        ],
    },
)
