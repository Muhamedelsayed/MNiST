from setuptools import setup

 

with open("README.md","r") as fh:
    long_description = fh.read()
setup(
    name = 'MoSeka',
    version = '2.1',
    description = 'Deep Learning Framework',
    py_modules = ["activations","evaluation_matrix","functional","layers","losses","net","Utils","Activation_util","utils_func","Convolution_util","Pooling_util","RBF_initial_weight","LayerObjects"],
    package_dir = {'':'MoSeka'},
    install_requires = [
        "matplotlib <= 3.1.3",
        "pillow <= 8.0.1",
        "numpy <= 1.17.5",
        "pandas <= 0.25.1",
    ],
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license="MIT",
    url="https://github.com/Muhamedelsayed/MNiST",
)