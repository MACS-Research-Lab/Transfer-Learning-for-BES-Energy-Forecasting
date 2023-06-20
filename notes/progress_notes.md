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
- [x] Learn how to make interactive plots

**_:calendar: Tue, Jun 13_**

_Ideas or Plan_

- [x] Change lstm preprocessing method of handling off data
- [x] Reverse normalize predicted data
- [x] Clean up ESB Summer LSTM preprocessing

> **Disclaimer**:<br/>
> Timesteps jumps have been appropriately removed while handling cases where data was removed due to the cooling tower being off, but not in cases of data removal due to outliers/missing data.<br/>
> Since outliers and missing data for ESB were previously removed in the esb_preprocessing file, timestep jumps due to these will only constitute a very small portion of the data (since outliers and missing data constituted <0.1% of all the data), so I have not changed my outlier and missing data handling methods.<br/>

**_:calendar: Wed, Jun 14_**

- [x] Preprocess Kissam data
- [x] Create LSTM for Kissam tower 1
- [x] Use model of Kissam tower 1 on tower 2 without additional training

**_:calendar: Thu, Jun 15_**

- [x] Generalize model creation process
- [x] Replicate LSTM for all Kissam & ESB towers

**_:calendar: Fri, Jun 16_**

- [x] Improve presentation of results
- [ ] Go through faulty results in replicated LSTM

## **Week 4**

**_:calendar: Mon, Jun 19_**

- [x] Comparison & correlation analysis of ESB/Kissam variables
- [x] Transfer ESB summer onto Kissam summer and vice versa

> **Better preprocessing points**:<br/>
> Make sure that during model transfers, order of columns is the same.<br/>

**_:calendar: Tue, Jun 20_**

- [x] Reread paper on transfer learning for building energy predictions
- [x] Handle zero temperatures as missing data
- [x] Clean up model generation and intra/inter building transfer notebooks
- [ ] Learn about weight-fixed fine tuning

> **Questions**:<br/>
> Variables used in efficiency calculation are enteringWaterTemp, outdoorAirWetBulb and leavingWaterTemp, but I only predict leavingWaterTemp and rather use the first two to predict it. Isn't that a problem?<br/>
> Currently predictions use the past 30 mins of data to make a prediction for the next 5 minute leavingWaterTemp – do I need to make predictions further away?

_Links and papers that could be useful :link:_

- [Statistical investigations of transfer learning-based methodology for shortterm building energy predictions](https://www.sciencedirect.com/science/article/pii/S0306261920300118?ref=pdf_download&fr=RR-2&rr=7d0b6d0aaba3f7cc)
- [A short-term building cooling load prediction method using deep learning algorithms](https://www.sciencedirect.com/science/article/pii/S0306261917302921)
- [Transfer learning with deep neural networks for model predictive control of HVAC and natural ventilation in smart buildings](https://www.sciencedirect.com/science/article/pii/S0959652619347365)

> **Paper notes** for "Statistical investigations of transfer learning-based methodology for shortterm building energy predictions":<br/>
> Compares using the pre-trained model for feature extraction vs weight initialization?<br/>
> Preprocessing: Add categorical variable for dayOfWeek<br/>
