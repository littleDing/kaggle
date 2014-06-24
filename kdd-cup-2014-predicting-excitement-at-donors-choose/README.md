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

Running
----------------
Basicly, running parameters can be selected in solutions@solutions.py
Run with command "python solutions.py dense5", where dense5 is one of the running parameters sets memtion above.

Integrating
----------------
2 ways to integrate with other models or features:
- write feature\_XXX like others in instances.py, add new feature in solutions and run it
- call instances.make\_dense\_instance\_raw where a DataFrame will be returned with projectid, feature and label

