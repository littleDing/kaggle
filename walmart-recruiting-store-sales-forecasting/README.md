walmart-recruiting-store-sales-forecasting
======
codes for walmart-recruiting-store-sales-forecasting

Installing
------------------------------------
Codes had been tested under mac-os-x 10.9.2 & ubuntu 10.04

Just checkout and make sure the requirements are well installed:
- python 2.7, scikit-learn, numpy, pandas, scipy
- [RGF](http://stat.rutgers.edu/home/tzhang/software/rgf/)

Configuration
------------------------------------
All configs are in configs.json
- root_dir : the working directory, a base reference of the DATA_DIR, TMP_DIR, etc. If left null, will be the current directory.
- rgf : a *FULL* path of the executable rgf 

Besides, three data files should be downloaed from [KAGGLE](http://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data):
- data/train.csv
- data/test.csv
- data/features.csv

Running
------------------------------------
Basicly, running parameters can be selected in solutions@solutions.py

Running results can be found in directory answers 

Two selected models in competitions could be obtained:

- Single(score public/private 2725.04573/2814.25115) : "python solutions.py Single" 
- Double(score public/private 2673.70317/2764.21952) : "python solutions.py Double" 

Ideas
------------------------------------
The key structure is to use a tree model to capture the nonlinearity and build different features to feed the model.

Feature engineering including time-cycle mapping, medians within different types of groupby, normalization within Store or Date to capture a relative feature, etc. 

There are three main ideas contribuiting huge improvement:
- Missing Value: 
There are many missing values in features.csv, such values are filled with medians of the same Store

- Data Partition: 
Teatures are provided by Store which means no difference exists between Depts in the same Store. 
To capture the nonlinearity provided by Dept, data are partitioned so that there is a model for every single Dept.
Moreover, partitioning data according to Dept&Store will provide larger improvement, but it is proved to be less efficient when more complex feature is provided.

- Kernel Density Estimating: 
A kernel density of WeeklySales along *timeline* is estimated within the same *ID*.
Where *timeline* is some time measure like WeekOfYear, SeasonOfYear.
And *ID* could be Dept, Store and Dept&Store, which provide a similar but generized version of impact like data partition.


Implementation Details
------------------------------------
All codes are written in Python. With the help of numpy and pandas, it is easy to handle data and feature extraction.
But as many features were created during the competition and even more of the combinations, a solution-instance framework is implemented.
The solutions.py handle most of the train-test works while features are created by instances.py.
The two modules are connected using some dynamic parameter pass.

