# HVAC Model

<h2> Data-driven predictive model for cooling tower performance </h2>

**Objective:** Create a data-driven predictive model based on time-series for the cooling tower efficiency. <br/>

<h4>:pushpin: Abstract of the project:</h4> <br/>
Heating Ventilation and Air Conditioning systems (HVACs) are used to control temperature and humidity inside buildings. They use cooling towers at one end to expel excess heat from the refrigerant into the atmosphere. The heat is expelled through evaporative cooling. <br/>
Cooling towers are used to expel heat from warm water coming out of water-cooled condensers in chiller units. The water has absorbed heat from the chiller’s refrigerant as it condenses. The rate of cooling depends on the cooling tower surface area, humidity, temperature, and speed of water and air. Cooling tower efficiency can be expressed as
$$
μ = (\frac{t_i  - t_o }{t_i  - t_w) })*100
$$
where
μ = cooling tower efficiency (%) <br/>
t_i  = inlet temperature of water to the tower ( o C,  o F) <br/>
t_o  = outlet temperature of water from the tower ( o C,  o F) <br/>
t_w  = wet bulb temperature average between current time and next time reading ( ^o C,  ^o F) <br/>

The temperature difference between inlet and outlet water (t_i  - t_o ) is normally in the range 10 - 15  o F. The units should be consistent for the temperature, but they can be kept as F throughout.


:flashlight: **Go to [analysis.ipynb](analysis.ipynb).**