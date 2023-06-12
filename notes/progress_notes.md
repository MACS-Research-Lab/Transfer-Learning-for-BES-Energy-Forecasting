## **Week 1**

**_:calendar: Thu, Jun 1_**

_Ideas or Plan_

- [x] Recreate my model in keras with lstm instead of having it scikit learn: [View new model](../models/esb1_lstm_model.ipynb)

_Links and papers that could be useful :link:_

- [Transfer learning with deep neural networks for model predictive control of HVAC and natural ventilation in smart buildings](https://www.sciencedirect.com/science/article/pii/S0959652619347365)

- [LSTM RNN for transfer learning on time series data](https://towardsdatascience.com/transfer-learning-for-time-series-prediction-4697f061f000)

- [Online Energy Management in Commercial Buildings using Deep Reinforcement Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8784062&tag=1) (ESB dataset was used)

- [(Tutorial) Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

**_:calendar: Fri, Jun 2_**

_Ideas or Plan_

- [x] Understand how to use the buildings API
- [x] Go through BDX for buildings with similar variables and collect data on 3-4 buildings

_Links and papers that could be useful :link:_

- [Statistical investigations of transfer learning-based methodology for shortterm building energy predictions](https://www.sciencedirect.com/science/article/pii/S0306261920300118?ref=pdf_download&fr=RR-2&rr=7d0b6d0aaba3f7cc)

- [Deep reinforcement learning control for non-stationary building energy management](https://www.sciencedirect.com/science/article/pii/S0378778822007551)

## **Week 2**

**_:calendar: Mon, Jun 5_**

Modification in problem statement: Work only with cooling tower data

_Ideas or Plan_

- [x] Get weather data
- [x] Continue collecting building data

_Links and papers that could be useful :link:_

- National Centers for Environmental Information: [weather data source](https://www.ncei.noaa.gov/cdo-web/search) and [documentation](https://www.ncei.noaa.gov/data/daily-summaries/doc/GHCND_documentation.pdf)

- [Engineering Science Building Data Documentation](https://iahmed.me/EngineeringScienceBuilding/)

**_:calendar: Tue, Jun 6_**

_Ideas or Plan_

- [x] Preprocess ESB data for both cooling towers
- [x] Create some generalized preprocessing functions (missing data removal, outlier removal)
- [ ] Decide on which column to set a target for predictions

**_:calendar: Wed, Jun 7_**

_Ideas or Plan_

- [x] Preprocess Kissam to check variable correlations with ESB
- [x] Compare variables across ESB and Kissam

> **Brainstorming**:<br/>
> Current approach - I've been trying to consolidate data for all cooling towers of a building into a single dataset for that building.<br/>
> Problem - Different buildings have different numbers of cooling towers so I haven't been able to come up with a definite generalized approach to do this yet.<br/>
> New approach idea - Now I'm thinking that instead of trying to approach this as transferring a model for one building onto another, approach it as transferring a model of one cooling tower onto another (so if a building has two cooling towers treat these independently)<br/>
> Potential problems with new approach: A building may alternate between using its cooling towers, so any individual cooling tower would appear to be off for long period thereby leading to misrepresentative data<br/>

_Links and papers that could be useful :link:_

- [Code for wet bulb temperature equation](https://github.com/hazrmard/EngineeringScienceBuilding/blob/master/src/preprocessing/thermo.py)

- [Hyperparameter Optimization With Random Search and Grid Search](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)

**_:calendar: Thu, Jun 8_**

_Ideas or Plan_

- [ ] Go through preprocessing methods previously made for ESB and incorporate
- [x] Preprocess MRB3 to check variable correlations with ESB and Kissam
- [x] Compare variables across ESB and MRB3

**_:calendar: Fri, Jun 9_**

_Ideas or Plan_

- [x] Get more frequently recorded datapoints than 1 hour using the API
- [x] Preprocess new ESB data and create generalized preprocessing functions
- [x] Come up with a timestep dimension creating algorithm

## **Week 3**

**_:calendar: Mon, Jun 12_**

_Ideas or Plan_

- [x] Modify esb1-lstm-model to use new preprocessed ESB data
- [x] Use ESB1_summer_lstm_model trained on ESB cooling tower 1 on ESB cooling tower 2 (without any retraining)
- [ ] Change lstm preprocessing method of handling off data

_Ideas or Plan_

- [ ] Follow transfer learning procedures to use ESB1_summer_lstm_model-lstm-model on ESB cooling tower 2
- [ ] Use wet bulb temperature equation by Roland Stull to improve data imputation techniques
