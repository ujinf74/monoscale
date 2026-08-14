from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'monoscale_carla'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ujin Kwon',
    maintainer_email='ujin@tukorea.ac.kr',
    description='The CARLA-side pieces: the assembled camera pair, the truth tap and the scripted drive.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'fisheye_assembler = monoscale_carla.fisheye_assembler:main',
            'carla_ground_truth = monoscale_carla.carla_ground_truth:main',
            'ego_pilot = monoscale_carla.ego_pilot:main',
        ],
    },
)
