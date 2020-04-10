from setuptools import setup
import os

classifiers = ['Development Status :: 4 - Beta',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               ]

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    requires = []
else:
    requires = ["numpy>=1.10",
                "pandas>=0.15",
                "scipy",
                "matplotlib>=1.5",
                "netCDF4",
                "pyproj",
                "pygrib",
                "keras",
                "h5py",
                "cartopy"
                ]

if __name__ == "__main__":
    pkg_description = "Doineann is a deep learning Python package for predicting severe weather."

    setup(name="doineann",
          version="0.0",
          description="Deep learning severe weather forecast system",
          author="Amanda Burke",
          author_email="aburke1@ou.edu",
          long_description=pkg_description,
          license="MIT",
          url="https://github.com/alburke/doineann",
          packages=["util","processing"],
          scripts=["bin/dndata", "bin/dnforecast", "bin/dnfileoutput"],
            #,"bin/dncalibration" 
            #"bin/hsfileoutput", "bin/hsplotter", 
            #    "bin/hswrf3d", "bin/hsstation", "bin/hsncarpatch", "bin/hscalibration","bin/hsdldata",
            #    "bin/hsdlforecast","bin/hsdlfileoutput"],
          data_files=[("mapfiles", ["mapfiles/hrefv2_2018_map.txt"])],
          keywords=["hail","verification","deep learning","weather", "meteorology"],
          classifiers=classifiers,
          include_package_data=True,
          zip_safe=False,
          install_requires=requires)
