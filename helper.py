import streamlit as st
import folium
from geopy.distance import geodesic
from streamlit_folium import folium_static
import re
import pandas as pd

def extract_timings(timings_list):
    extracted_timings = {}
    
    days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for day in days_of_week:
        extracted_timings[day] = None
    
    for timing in timings_list:
        # Extract day, start time, and end time using regular expressions
        match = re.match(r'(\w+)\s(\d{1,2}:\d{2} [ap]m) - (\d{1,2}:\d{2} [ap]m)', timing.replace("'",""))
        if match:
            day = match.group(1)
            start_time = match.group(2)
            end_time = match.group(3)
            extracted_timings[day] = f"{start_time} - {end_time}"
    return extracted_timings

def availability(df, day):
    start =[]
    end = []
    doctors = []

    for index, row in df.iterrows():

      if str(row[day]) == 'None':
        start.append(None)
        end.append(None)

      else:
        start.append(str(row[day]).split(' - ')[0])
        end.append(str(row[day].split(' - ')[1]))  

      doctors.append(str(row['Doctor']))

    df_name = str(day)+'_df'

    df_name = pd.DataFrame({'Doctor': doctors, 'start': start, 'end': end})
    df_name['start'] = pd.to_datetime(df_name['start'], format='%I:%M %p')
    df_name['end'] = pd.to_datetime(df_name['end'], format='%I:%M %p')
    
    return df_name

def get_similar(text_list, encoder, faiss, query_text):
    vectors = encoder.encode(text_list)
    vector_dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(vectors)
    index.add(vectors)


    # Encode the query text
    query_vector = encoder.encode([query_text])

    # Search for similar text
    k = 5  # Number of nearest neighbors to retrieve
    distance, indices = index.search(query_vector, k=10)
    # Retrieve the similar text based on the indices
    similar_text = [text_list[i] for i in indices[0]]

    for i in similar_text:
        i = i.replace("]",'')
        i = i.replace("[",'')
        i = i.replace("'", '')
    
    return similar_text, vectors, query_vector

def create_map(final_df):
    # Create a map centered at Santa Clara University
    map_center = (37.349, -121.939)
    m = folium.Map(location=map_center, zoom_start=8)

    # Add Santa Clara University marker with two icons
    santa_clara_marker = folium.Marker(
        location=(37.349, -121.939),
        popup='Santa Clara University',
        icon=folium.Icon(color='red', icon='university'),
    )
    santa_clara_marker.add_child(folium.Icon(color='red', icon='star'))
    santa_clara_marker.add_to(m)

    # Add clinic markers in blue color and connect with lines
    for i,doctor in final_df.iterrows():
        # Create HTML for doctor name with clickable link
        doctor_name = f"<a href='{doctor['url']}' target='_blank' style='text-decoration: underline; color: #333; font-weight: bold;'>{doctor['Doctor']}</a>"

        clinic_marker = folium.Marker(
            location=(doctor['Latitude'], doctor['Longitude']),
            popup=doctor_name,
            icon=folium.Icon(color='blue', icon='clinic-medical', prefix='fa'),
        ).add_to(m)

        distance = calculate_distance(map_center, (doctor['Latitude'], doctor['Longitude']))

        folium.PolyLine(
            locations=[map_center, (doctor['Latitude'], doctor['Longitude'])],
            color='blue',
            weight=1.5,
            opacity=1,
            tooltip=f"Distance: {distance} miles",
        ).add_to(m)

    return m

def calculate_distance(point1, point2):
    # Calculate distance between two points in miles using geodesic distance
    return round(geodesic(point1, point2).miles, 2)
