# Import packages 
import plotly.express as px 
import pandas as pd 

# Read in the data 
data = pd.read_csv('AirPassengers.csv') 

# Plot the data （横纵轴）
fig = px.line(data, x='Month', y='#Passengers', 
              labels=({'#Passengers': 'Passengers', 'Month': 'Date'})) 

#标题
fig.update_layout(template="simple_white", font=dict(size=18), 
                  title_text='Airline Passengers', width=650, 
                  title_x=0.5, height=400)