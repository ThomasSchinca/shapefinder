# shapefinder

shapefinder is a historical pattern matching tool for forecasting purposes. Originially built for social science application like conflict and 
migration flow, it can be adapted for any purpose. This library includes various functions for pattern discovery, dynamic prediction, and data visualization. 
It identifies and matches temporal patterns in data to forecast future trajectories based on similarity to past episodes.

## 📦 Installation

Before installing the package, [create a virtual environment](https://docs.python.org/3/tutorial/venv.html) to avoid conflicts with other packages. 

Install shapefinder through PyPl:
```bash
python -m pip install shapefinder
```

Or using the github version:
```bash
git clone https://github.com/ThomasSchinca/shapefinder.git
cd shapefinder
pip install -r requirements.txt
```

## 🛠️ Usage

### Shape 

```python
from shapefinder import Shape

### Init the Shape object 
shape = Shape()

### Three ways to set your shape wanted 

# 1. Random shape with a given window length 
shape.set_random_shape(window=10)   # in this exemple, the shape has 10 random values between 0 and 1

# 2. Shape drawn by user
shape.draw_shape(window=10) # in this exemple, a pop up appears to let the user draw the shape wanted with a 10 timestamp window. 

# 3. Shape based on values given by user (list, series or numpy format)
shape.set_shape([0,0.5,1,0.5,0])   # in this exemple, the shape has 5 values that draw an "up-and-down" shape. 


### Visualize the shape created 
shape.plot()    # A figure is created 

```

<div align="center">
    <img src="https://github.com/ThomasSchinca/shapefinder/blob/main/docs/assets/Input.png" alt="Input Shape" width="400"/>
</div>

Once the Shape is defined, the finder is set to find the shape in the given dataset. 

### finder

```python
from shapefinder import Shape
from shapefinder import finder
import pandas as pd 
import numpy as np

### Init the Shape object 
shape = Shape()
shape.set_shape([0,0.5,1,0.5,0])   # we take the "up-and-down" shape defined before

### Define the dataset to look into 
# We create a white noise time series 

data = pd.DataFrame(np.random.randn(1000), index=pd.date_range(start='2020-01-01', periods=1000, freq='D'))

### Init the finder object 
find = finder(data,Shape=shape)

### Find the 'up-and-down' pattern in the dataset

# Look for patterns with euclidean distance lower than 0.5
find.find_patterns() 
# Look for patterns with euclidean distance lower than 1
find.find_patterns(min_d=1) 
# Look for patterns with Dynamic Time Warping distance lower than 0.25
find.find_patterns(min_d=0.25,metric='dtw')
# Look for patterns with Dynamic Time Warping distance lower than 0.25 and allow window with 4, 5 and 6 timestamp 
# window to look = 5 +/- dtw_sel(1)
find.find_patterns(min_d=0.25,metric='dtw',dtw_sel=1)
# Same but allowing overlapping windows 
find.find_patterns(min_d=0.25,metric='dtw',dtw_sel=1,select=False)
# Force the minimum number of similar patterns to 5 
find.find_patterns(min_d=0.25,metric='dtw',dtw_sel=1,select=True,min_mat=5)

### Plot the patterns found
# Plot individual figures 
find.plot_sequences(how='units') 
# Plot global figure with all patterns
find.plot_sequences(how='total') 
```
<div align="center">
    <img src="https://github.com/ThomasSchinca/shapefinder/blob/main/docs/assets/Similar_plot.png" alt="Total similar sequences" width="800"/>
</div>
The final step is then to create the future scenario and predict. 

### Scenario and prediction creation 

```python
### Predict the next three timestamps based on the found patterns
# First, we generate the scenarios using an horizon of three and a clustering threshold of 3. 
find.create_sce(horizon=3,clu_thres=3) 
# We can check how do the scenerio look like with their associated probabilities (in the legend).
find.plot_scenario()
```

<div align="center">
  <img src="https://github.com/ThomasSchinca/shapefinder/blob/main/docs/assets/Scenario.png" alt="Scenario with their associated probabilities" width="400"/>
</div>

```python
# We can also just predict a point estimate using the highest probable scenario
pred = find.predict(horizon=3,clu_thres=3)

```

## 📊 Input Data Format

- Shape : list, pandas.Series or numpy.array
- Shape3D : np.ndarray, with shape(x,y,time) 

- finder : pandas.DataFrame with index as Date. 
- finder_3D : np.ndarray with (x,y,time)
- finder_multi : list of pandas.DataFrame with index as Date
- finder_multi_static : list of pandas.DataFrame with index as Date

## 🧪 Examples
More examples specific for :
- Shape_3D, finder_3D : examples/3D_example.ipynb
- finder_multi : examples/Multi_Shape_example.ipynb
- finder_multi_static : examples/Shape_with_covariates_example.ipynb

## 📚 Related Papers

ShapeFinder was developed as part of a research project on conflict forecasting and migration flow using historical pattern matching. 
The following papers use ShapeFinder's methods and code:

- **[Temporal Patterns in Migration Flows : Evidence from South Sudan (2025)](https://doi.org/10.1002/for.3209)**  
  Uses finder_multi to generate forecasts for migration flows in South Sudan.

- **[Temporal Patterns in Conflict Prediction: An improved Shape-Based Approach (2025)](https://journals.sagepub.com/doi/10.1177/00223433251330790)**  
  Uses finder to predict conflict fatalities at the country-month level. 

- **The geometry of conflict : 3D Spatio-temporal patterns in fatalities prediction (Upcomming)**  
  Uses Shape_3D and finder_3D to forecast conflict fatalities at the 0.5x0.5°(called Prio-Grid) month level, 


## 📄 Citation
If you use shapefinder in academic work, please cite:

```pgsql

@misc{shapefinder2025,
  author = {Thomas Schincariol},
  title = {ShapeFinder: Historical Pattern Matching for Forecasting},
  year = {2025},
  howpublished = {\url{https://github.com/ThomasSchinca/shapefinder}}
}
```

## 📬 Contact
For questions, reach out via GitHub Issues or email schincat@tcd.ie.
