# shapefinder

The ShapeFinder library is a collection of useful functions for analyzing and visualizing time series data. 
This library includes various functions for pattern discovery, dynamic prediction, and data visualization. 
The library is built to be versatile and can be used in a wide range of time series analysis tasks.

# How to use 

### Shape 

```python
from shapefinder import Shape

### Init the Shape object 
shape = Shape()

### Three ways to set your shape wanted 

# 1. Random shape with a given window length 
shape.set_random_shape(window=10)   # in this exemple, the shape has 10 random values between 0 and 1

# 2. Shape based on values given by user (list, series or numpy format)
shape.set_shape([0,0.5,1,0.5,0])   # in this exemple, the shape has 5 values that draw an "up-and-down" shape. 

# 3. Shape drawn by user
shape.draw_shape(window=10) # in this exemple, a pop up appears to let the user draw the shape wanted with a 10 timestamp window. 


### Visualize the shape created 
shape.plot()    # A figure is created with 

```

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
# Look for patterns with Dynamic Time Warping distance lower than 0.5
find.find_patterns(min_d=0.5,metric='dtw')
# Look for patterns with Dynamic Time Warping distance lower than 0.5 and allow window with 4, 5 and 6 timestamp 
# window to look = 5 +/- dtw_sel(1)
find.find_patterns(min_d=0.5,metric='dtw',dtw_sel=1)
# Same but allowing overlapping windows 
find.find_patterns(min_d=0.5,metric='dtw',dtw_sel=1,select=False)

### Plot the patterns found
# Plot individual figures 
find.plot_sequences(how='units') 
# Plot global figure with all patterns
find.plot_sequences(how='total') 

### Predict the next timestamp based on the found patterns
# Predict the 4 next timestamp and plot the result 
pred = find.predict(4,plot=True)
# The parameters of inclusion are similar to find_patterns
# Here, we predict using patterns with Dynamic Time Warping distance lower than 1
pred = find.predict(4,plot=True,min_d=1,metric='dtw')

```

