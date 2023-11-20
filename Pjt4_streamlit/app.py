### --- IMPORT LIBRARIES --- ###
# Libraries
from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu
from urllib.request import urlopen
import pickle
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import geopandas as gpd
from geopy.geocoders import Nominatim
import folium
from datetime import datetime, timedelta
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz
import xgboost as xgb 
import re

### --- IMPORT DATA --- ###
## spray_df
spray_path = Path(__file__).parent / "spray_clean.csv"
spray_df = pd.read_csv('spray_clean.csv')

## train_df
train_df_path = Path(__file__).parent / "train_clean.csv"
train_df = pd.read_csv('train_clean.csv')

## train
train_path = Path(__file__).parent / "train.csv"
train = pd.read_csv('train.csv')

## weather_df
weather_path = Path(__file__).parent / "weather_clean.csv"
weather_df = pd.read_csv('weather_clean.csv')

# serious_cases
serious_cases_path = Path(__file__).parent / "weather_clean.csv"
serious_cases = pd.read_csv('wnv_serious_cases.csv')

## Chicago map GEOJSON file
chicago_path = Path(__file__).parent / "boundaries_chicago.geojson"
chicago_city = gpd.read_file('boundaries_chicago.geojson')

# Import model
model_path = Path(__file__).parent / "xgb_model.pkl"
xgb_model = pickle.load(open(model_path, 'rb'))

### --- FUNCTIONS --- ###

# Define a function to determine species that are likely to be present at a given location and date
def predict_species(df, date, latitude, longitude):
    
    # Filter data based on location
    df = df[(df['Latitude'] == latitude) & (df['Longitude'] == longitude)]

    # Filter the weather data for the same date
    df = df[df['Date'] == date]

    # Calculate the probability of each species based on location
    species_probabilities = df['Species'].value_counts(normalize=True)

    # Filter species by the threshold of 30%
    selected_species = species_probabilities[species_probabilities > 0.3].index.tolist()

    # Return the list of species that are likely to be present at the given location and date
    return selected_species

def species_present(df, selected_species):
    # Filter to get only species columns
    species_columns = [col for col in df.columns if col not in ['date', 'latitude', 'longitude']]

    # Iterate over species columns and assign values based on selected_species list
    for species_col in species_columns:
        if species_col in selected_species:
            df[species_col] = 1
        else:
            df[species_col] = 0

    return df

# Define a function to determine the number of mosquitos 
def predict_nummosquitos(df, date, latitude, longitude):
    # Filter the historical weather data based on location
    df = df[(df['latitude'] == latitude) & (df['longitude'] == longitude)]

    # Filter the weather data for the same date
    df = df[df['date'] == date]

    # Calculate the average number of mosquitoes for that date
    mean_mosquitos = df['nummosquitos'].mean()

    return mean_mosquitos

# Define a function that derives all weather datapoints needed for the model based on the current weather data we can get from the API
def derived_weather_data(current_temp, current_dewpoint, wind_speed, date, df):

    # Create an empty dictionary to store weather data
    weather_data = {}

    # Approximate derived weather data with given current weather data, using formulas suggested on weather websites
    weather_data['tmax'] = float(current_temp )+ 5
    weather_data['tmin'] = float(current_temp) - 5
    weather_data['tavg'] = float(current_temp)
    weather_data['depart'] = str(float(current_temp) - 73)
    weather_data['dewpoint'] = current_dewpoint
    weather_data['wetbulb'] = str(float(current_temp) - 5)
    weather_data['heat'] = str(float(current_temp) - 65)
    weather_data['cool'] = str(float(current_temp) - 65)
    weather_data['resultspeed'] = '10'
    weather_data['resultdir'] = '180'
    weather_data['avgspeed'] = wind_speed

    # Calculate 'sunrise' and 'sunset' using mean of the same dates from historical data
    filtered_df = df[df['date'] == date]

    # Use the average 'sunrise', 'sunset', 'preciptotal', 'stnpressure', 'sealevel' from historical data if available
    avg_sunrise = filtered_df['sunrise'].mean()
    avg_sunset = filtered_df['sunset'].mean()
    preciptotal = filtered_df['preciptotal'].mean()
    stnpressure = filtered_df['stnpressure'].mean()
    sealevel = filtered_df['sealevel'].mean()
    weather_data['sunrise'] = avg_sunrise
    weather_data['sunset'] = avg_sunset
    weather_data['preciptotal'] = preciptotal
    weather_data['stnpressure'] = stnpressure
    weather_data['sealevel'] = sealevel

    return weather_data

### --- WEB LAYOUT --- ###
# Configure webpage
st.set_page_config(
    page_title='Chicago West Nile Virus Control',
    page_icon='mosquito',
    layout='wide',
    initial_sidebar_state='expanded'
    )

### --- HEADER --- ###
st.markdown("""
    <h1 style='text-align: center; font-size: 70px;'>Chicago West Nile Virus Control</h1>
""", unsafe_allow_html=True)

### --- TOP NAVIGATION BAR --- ###
selected = option_menu(
    menu_title = None,
    options = ['About', 'Trends', 'Surveillance', 'Control Measures', 'Contact Us'],
    icons = ['eject', 'bar-chart', 'binoculars', 'shield', 'phone'],
    menu_icon = 'mosquito',
    default_index = 0,
    orientation = 'horizontal',
    styles={
        'nav-link-selected': {'background-color': '#89b5ffff'},
        }
)

if selected == 'About':
    st.title('About')
    style = "<div style='background-color:#2C78DA; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)  

    ### WEST NILE VIRUS ###
    # 'West Nile Virus' header
    st.subheader('West Nile Virus')

    # Insert text
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>**West Nile Virus (WNV)**  is a mosquito-borne disease that can infect humans and animals, leading to various symptoms, including fever and, in severe cases, neurological complications. It is transmitted through mosquito bites, and the virus has been reported in several regions worldwide. Vigilance against WNV is crucial as it can lead to health issues, and preventing mosquito breeding and taking protective measures, such as using mosquito repellent, can reduce the risk of infection.</span>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")

    # 'In Chicago' header
    st.subheader('In Chicago')
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>**Chicago** is grappling with a severe West Nile Virus (WNV) situation, standing as the second-worst-hit city in the United States. The presence of the Culex mosquito, a prevalent carrier of the virus, has significantly contributed to the high number of WNV cases in the area. With its dense population and ideal conditions for mosquito breeding, the risk of WNV transmission has escalated.</span>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>The city's residents face a persistent threat to their health due to the alarming spread of this mosquito-borne illness. WNV has caused numerous cases of fever, and in more severe instances, neurological complications, leading to a considerable health burden in the city. </span>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:

        # Causes
        st.markdown("""
    <h1 style='text-align: center; font-size: 30px;'>Causes</h1>
""", unsafe_allow_html=True)
        # Insert image
        image = Image.open('wnv_transmission.png')
        st.image(image)

    with col2:

        # Symptoms
        st.markdown("""
    <h1 style='text-align: center; font-size: 30px;'>Symptoms</h1>
""", unsafe_allow_html=True)
        # Insert image
        image = Image.open('wnv_survey.png')
        st.image(image,caption='Source: Byas, A.D. and Ebel, G.D., 2020')
    if selected != 'About':
        next_tab_button = st.button("Next Tab: Trends")
        if next_tab_button:
            next_section_index = list(sections.keys()).index(selected) + 1
            next_section = list(sections.keys())[next_section_index]
            st.experimental_set_query_params(selected=next_section)
    
if selected == 'Trends':
    st.title('Trends')
    style = "<div style='background-color:#2C78DA; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)  

    # Create a dropdown menu for the user to select a trend
    option = st.selectbox('Select a tend',
                            ('West Nile Fever Cases in Chicago By Community',
                             'West Nile Fever Cases in Chicago Over The Years', 
                             'Number of Mosquitos Trapped Monthly', 
                             'Presence of WNV in Traps in Summer Months', 
                             'Mosquito Species in WNV Positive Traps',
                             'Spray and Trap Locations'))
    
    # translate the english from the option box into the equivalent variable
    if option == 'West Nile Fever Cases in Chicago By Community':
        st.subheader('Mosquito Population By Community')
        with open("mos_heatmap.html", "r") as f:
            html_code = f.read()
        st.components.v1.html(html_code, width=700, height=500)

    elif option == 'West Nile Fever Cases in Chicago Over The Years':
        st.subheader('West Nile Fever Cases in Chicago Over The Years')
        st.markdown("")
        st.markdown("")
        image = Image.open('wnf_chicago.png')
        st.image(image)

    elif option == 'Number of Mosquitos Trapped Monthly':
        st.subheader('Number of Mosquitos Trapped Monthly')
        st.markdown("")
        st.markdown("")
        image = Image.open('monthly_trapped_mos.png')
        st.image(image)

    elif option == 'Presence of WNV in Traps in Summer Months':
        st.subheader('Presence of WNV in Traps in Summer Months')
        st.markdown("")
        st.markdown("")
        image = Image.open('wnv_summer.png')
        st.image(image)

    elif option == 'Mosquito Species in WNV Positive Traps':
        st.subheader('Mosquito Species in WNV Positive Traps')
        st.markdown("")
        st.markdown("")
        image = Image.open('species_wnv_pos.png')
        st.image(image)

    elif option == 'Spray and Trap Locations':
        st.subheader('Spray and Trap Locations')
        st.markdown("")
        st.markdown("")
        with open("spray_trap_locations.html", "r") as f:
            html_code = f.read()
        st.components.v1.html(html_code, width=700, height=500)
    

if selected == 'Surveillance':
    st.title('Surveillance')
    style = "<div style='background-color:#2C78DA; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    # Request url
    url = "https://api.weather.gov/points/41.8781,-87.6298"
    req = requests.get(url)

    # Retrieve forecast data
    forecast_url = req.json()['properties']['forecast']

    # Retrieve weather data for today
    forecast_info = requests.get(forecast_url).json()

    today_temp = str(forecast_info['properties']['periods'][0]['temperature'])
    today_dewpoint = str(round(forecast_info['properties']['periods'][0]['dewpoint']['value'], 2))
    today_humidity = str(forecast_info['properties']['periods'][0]['relativeHumidity']['value'])
    today_windspeed = forecast_info['properties']['periods'][0]['windSpeed']

    # Split wind speed into parts as it is a range
    parts = today_windspeed.split()
    # Remove 'to' and 'mph' from the parts list
    parts = [part for part in parts if part not in ['to', 'mph']]

    # Check if the parts list has 2 elements
    float(parts[0])
    float(parts[1])  

    if len(parts) == 2:
    # Extract the lower and upper bounds
        lower_bound = int(parts[0].split()[0])
        upper_bound = int(parts[1].split()[0])
        
        # Calculate the mid-point
        wind_speed = (lower_bound + upper_bound) / 2
        wind_speed = sum([float(part) for part in parts]) / len(parts)

    ### --- DATE --- ###

    ## 'Date' header
    st.subheader('Date')

    # Get the current time in Chicago's timezone
    chicago_timezone = pytz.timezone('America/Chicago')
    current_time = datetime.now(chicago_timezone)
    month = current_time.month

    # Format the time as a string with the desired format
    formatted_time = current_time.strftime("%d-%m-%Y %H:%M:%S")

    # Style the date display using HTML/CSS
    date_style = f"""
        <div style="font-size: 24px; font-family: Arial, sans-serif; color: black;">
            {formatted_time}
        </div>
    """
    # Display the formatted date
    st.markdown(date_style, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    
    ### --- WEATHER --- ###

    ## Create 'Weather' header
    st.subheader('Weather')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Temperature:')
        st.header(f'{today_temp} °F')
    
    with col2:
        st.subheader('Dewpoint:') 
        st.header(f'{today_dewpoint} °C')

    with col3:
        st.subheader('Wind Speed:') 
        st.header(today_windspeed)

    st.markdown("")
    st.markdown("")
    
    ### --- LOCATION --- ###
    ## Create 'Location' header
    st.subheader('Location')

    # Create a geocoder instance
    geolocator = Nominatim(user_agent="my_geocoder")

    # Input field for address
    address = st.text_input("Enter an address:")
    latitude = 0.0
    longitude = 0.0

    if st.button("Submit"):
        if address:
            try:
                location = geolocator.geocode(address)
                if location:
                    st.success(f"Latitude: {location.latitude}, Longitude: {location.longitude}")
                else:
                    st.warning("Address not found.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a valid address.")

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")

    # Predict number of mosquitos
    nummosquitos = predict_nummosquitos(train_df, formatted_time, latitude, longitude)
    
    # Predict mosquito species
    predict_species = predict_species(train, formatted_time, latitude, longitude)
    species_present = species_present(train_df, predict_species)
    
    # Fill in weather data
    weather_data = derived_weather_data(today_temp, today_dewpoint, today_windspeed, formatted_time, weather_df)

    # Create a dictionary to store the species values
    species_values = {species: 1 if species in species_present else 0 for species in ['pipiens', 'restuans', 'territans']}

    # Assemble the data for prediction
    input_data_df = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'nummosquitos': [nummosquitos],
        'year_x': [current_time.year],
        'month_x': [current_time.month],
        'pipiens': [species_values['pipiens']],
        'restuans': [species_values['restuans']],
        'territans': [species_values['territans']],
        'tmax': [weather_data['tmax']],
        'tmin': [weather_data['tmin']],
        'tavg': [weather_data['tavg']],
        'depart': [weather_data['depart']],
        'dewpoint': [weather_data['dewpoint']],
        'wetbulb': [weather_data['wetbulb']],
        'heat': [weather_data['heat']],
        'cool': [weather_data['cool']],
        'sunrise': [weather_data['sunrise']],
        'sunset': [weather_data['sunset']],
        'preciptotal': [weather_data['preciptotal']],
        'stnpressure': [weather_data['stnpressure']],
        'sealevel': [weather_data['sealevel']],
        'resultspeed': [weather_data['resultspeed']],
        'resultdir': [weather_data['resultdir']],
        'avgspeed': [weather_data['avgspeed']],
        'year_y': [current_time.year],
        'month_y': [current_time.month]
    })


    # Make predictions
    prediction = xgb_model.predict(input_data_df)

    # Display the prediction result
    st.write("Predicted Probability of WNV:", prediction[0])
    st.write("Your area is under **high** risk. You may contact <insert hyperlink> for assistance.")


if selected == 'Control Measures':

    ### --- PREVENTIVE MEASURES --- ###
    st.title('Control Measures')
    style = "<div style='background-color:#2C78DA; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    # Text
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>The Chicago Department of Public Health (CDPH) has a robust mosquito control program, which includes treating 70,000 catch basins in Chicago with larvicide, collecting and testing mosquito samples every week, and spraying in specific geographic areas if indicated.</span>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")

    st.subheader("Advice for Residents")
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>Chicago residents are encouraged to take the following precautions against mosquitoes:</span>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>- Use insect repellant that contains DEET, Picaridin, IR3535, or Oil of Lemon Eucalyptus according to label instructions.</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>- Eliminate standing water. Empty water from any outdoor containers, such as flowerpots, gutters, tires, toys, pet water dishes, and birdbaths once weekly.</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>- Keep grass and weeds short to eliminate hiding places for adult mosquitoes.</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>- When outside between dusk and dawn, wear loose-fitting, light-colored clothing, long pants, long-sleeved shirts, socks, and shoes.</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>- Make sure that all screens, windows, and doors are tight-fitting and free of holes. Repair or replace screens that have tears or other openings.</span>", unsafe_allow_html=True)
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>- Check on neighbors regularly, especially those who are older, live alone, or need additional assistance.</span>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")

    st.subheader("Upcoming Events")
    st.markdown("<span style='font-size: 18px; font-family: Arial, sans-serif;'>No upcoming events. Check again next month. </span>", unsafe_allow_html=True)


if selected == 'Contact Us':

    ### --- CONTACT US --- ###
    st.title('Contact Us')
    style = "<div style='background-color:#2C78DA; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    # Text
    st.write("<span style='font-size: 18px; font-family: Arial, sans-serif;'>If you believe you may be at risk of West Nile virus or have questions about this infectious disease, don't hesitate to reach out for assistance and guidance. The City of Chicago and the Chicago Department of Public Health are here to support you and provide valuable information. </span>", unsafe_allow_html=True) 
    st.write("<span style='font-size: 18px; font-family: Arial, sans-serif;'>We encourage you to contact us through the following means:</span>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")

    # Contact Information
    st.header("City of Chicago")
    
    ## Create columns
    col1, col2 = st.columns(2)

    with col1:

        # Contact of the City of Chicago
        st.subheader("Phone:")
        st.write("<span style='font-size: 18px; font-family: Arial, sans-serif;'>**WNV Concerns:** 312.744.5000</span>", unsafe_allow_html=True)
        st.write("<span style='font-size: 18px; font-family: Arial, sans-serif;'>**WNV Information:** 312.747.9884</span>", unsafe_allow_html=True)
        
        st.subheader("Email:")
        st.write("<span style='font-size: 18px; font-family: Arial, sans-serif;'>healthychicago@cityofchicago.org</span>", unsafe_allow_html=True) 
        #st.write("<span style='font-size: 18px; font-family: Arial, sans-serif;'>healthychicago@cityofchicago.org</span>", unsafe_allow_html=True)

    with col2:
        st.subheader("Address:")
        st.write("<span style='font-size: 18px; font-family: Arial, sans-serif;'>City Hall, 121 N. LaSalle Street, Chicago, Illinois 60602</span>", unsafe_allow_html=True)        
        
        st.subheader("Website:")
        st.write("<span style='font-size: 18px; font-family: Arial, sans-serif;'>https://www.chicago.gov/city/en.html</span>", unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("")
    st.markdown("")

    # Final text
    st.write("<span style='font-size: 18px; font-family: Arial, sans-serif;'>Your health and well-being are of paramount importance to us. Please do not hesitate to get in touch if you have any questions, concerns, or require further guidance regarding West Nile virus prevention and management. We are here to help you stay safe and informed.</span>", unsafe_allow_html=True)    
