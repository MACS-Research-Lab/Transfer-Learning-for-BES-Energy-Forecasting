# HVAC Model for Vanderbilt ESB Tower 1

<h2> Data-driven predictive model for cooling tower performance </h2>

**Objective:** Create a data-driven predictive model based on time-series for the cooling tower efficiency. <br/>

**:pushpin: Abstract of the project:**</br>
Heating Ventilation and Air Conditioning systems (HVACs) are used to control temperature and humidity inside buildings. They use cooling towers at one end to expel excess heat from the refrigerant into the atmosphere. The heat is expelled through evaporative cooling. <br/>
Cooling towers are used to expel heat from warm water coming out of water-cooled condensers in chiller units. The water has absorbed heat from the chiller’s refrigerant as it condenses. The rate of cooling depends on the cooling tower surface area, humidity, temperature, and speed of water and air. Cooling tower efficiency can be expressed as

$$ μ = {{t_i  - t_o } \over {t_i  - t_w}}\*100 $$

where <br/>
μ = cooling tower efficiency (%) <br/>
$t_i$  = inlet temperature of water to the tower (C, F) <br/>
$t_o$  = outlet temperature of water from the tower (C, F) <br/>
$t_w$  = wet bulb temperature average between current time and next time reading (C,  F) <br/>

The temperature difference between inlet and outlet water ($t_i$  - $t_o$) is normally in the range 10 - 15 F. The units should be consistent for the temperature, but they can be kept as F throughout.

:flashlight: **Go to [analysis.ipynb](analysis.ipynb).**
Model developed using scikit-learn RandomForestRegressor
