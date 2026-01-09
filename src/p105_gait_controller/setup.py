from setuptools import find_packages, setup

package_name = 'p105_gait_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['p105_gait_controller/config/gait_controller.yaml']),
        ('share/' + package_name + '/launch', ['launch/gait_controller.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='c222',
    maintainer_email='ttt@ccc.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gait_controller = p105_gait_controller.gait_controller:main',
        ],
    },
)
