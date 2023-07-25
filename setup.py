from setuptools import setup, find_packages

setup(
    name="shapefinder",
    description="A tool to find patterns in time series dataset",
    version='0.1.0',
    author="Thomas Schincariol",
    author_email="thomas.schincariol@gmail.com",
    packages=find_packages(),
    install_requires=[
    "dash",
    "dash-bootstrap-components",
    "pandas",
    "matplotlib",
    "dtaidistance",
    "numpy",
    "tkinter",
    "plotly>=4.0.0",
    "dash-table>=4.0.0"
],
    keywords=['python', 'time series', 'find', 'pattern']
)
