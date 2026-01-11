from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        "orchard_row_mapping",
        "orchard_row_mapping.segmentation",
        "orchard_row_mapping.segmentation.vendor",
    ],
    package_dir={"": ""},
)

setup(**d)

