from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'monoscale_odometry'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ujin Kwon',
    maintainer_email='ujin@tukorea.ac.kr',
    description='Metric visual-inertial odometry from any number of monocular ground-facing cameras.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'monoscale_odometry = monoscale_odometry.odometry_node:main',
        ],
    },
)
