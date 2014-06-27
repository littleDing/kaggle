kdd-cup-2014-predicting-excitement-at-donors-choose
======

codes for kdd-cup-2014-predicting-excitement-at-donors-choose competition on kaggle

Installing
----------------
Codes had been tested under mac-os-x 10.9.2 & ubuntu 12.04
Just checkout and make sure the requirements are well installed:
- python 2.7, scikit-learn, numpy, pandas, scipy
- [XGBoost](https://github.com/tqchen/xgboost) with its python version compiled and put under ./scripts

Configuration
----------------
All configs are in configs.json
- root\_dir : the working directory, a base reference of the DATA\_DIR, TMP\_DIR, etc. If left null, will be the current directory.

Besides, raw csv data files should be put under ./datas

An summarization of current active feature list could be found in ./scripts/feature\_list.txt

Running
----------------
Basicly, running parameters can be selected in solutions@solutions.py
Run with command "python solutions.py dense5", where dense5 is one of the running parameters sets memtion above.

Integrating
----------------
3 ways to integrate with other models or features:
- write feature\_XXX like others in instances.py, add new feature in solutions and run it
- call instances.make\_dense\_instance\_raw where a DataFrame will be returned with projectid, feature and label
- call instances.dense\_instance\_to\_csv to write DataFrame to a csv file

Warning
----------------
- There cache files in ./caches, their meta data are stored in ./scripts/path\_mapping.cPickle, if you copy cache files from another workspace, remember to copy the meta file as well.
