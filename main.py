import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import DPRContextEncoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from streamlit_folium import folium_static
from helper import create_map, get_similar, availability, extract_timings
import os
import re

#detect similar diseases/conditions treated
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache_resource
def load_model():
        encoder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
        return encoder

encoder = load_model()

df = pd.read_csv('updated_output.csv')

st.title("Welcome to MedQuest !")
st.header("Lets get you treated..")
query_text = st.text_input("Enter disease name")

if query_text != '':
        
        
        df['Speciality'] = df['Speciality'].apply(lambda x: x.split(', '))
        merged_list = list(set([item for sublist in df['Speciality'] for item in sublist]))

        similar_text, vectors, query_vector = get_similar(merged_list, encoder, faiss, query_text)


        st.text(" ")

        st.header("Similar Diseases that nearby doctors treat: ")
        i = 0
        for i in range(0,len(similar_text)-3,3):
                disease_str_1 = f"**<span style='background-color: lightgreen;'>{similar_text[i]}</span>**"
                disease_str_2 = f"**<span style='background-color: lightgreen;'>{similar_text[i+1]}</span>**"
                disease_str_3 = f"**<span style='background-color: lightgreen;'>{similar_text[i+2]}</span>**"
                st.markdown(f"{disease_str_1}   {disease_str_2}    {disease_str_3}", unsafe_allow_html=True)
                #st.text(str(similar_text[i]) + ' ' + str(similar_text[i+1]) + ' ' + str(similar_text[i+2]))

        st.text(" ")
        st.text(" ")
        st.text(" ")

        st.header("Visualise Word Embeddings")        
        st.markdown('''A word embedding is a learned numeric representation for text where words that have the same meaning have a similar representation. It is this approach to representing words and documents that makes it easier for machines to interpret human languages.''')
        st.text("")
        option = st.radio(
                "What do you want to visualise",
                ('Top Similar Diseases', 'More Similar Diseases', 'All'))
        very_similar = similar_text[0:5]
        less_similar = similar_text[6:10]

        final_vector = np.append(vectors, query_vector, axis=0)

        #embeddings extract via TSNE
        tsne = TSNE(n_components=3, perplexity=2)
        embeddings = tsne.fit_transform(final_vector)
        merged_list.append(query_text)
        for i in merged_list:
                i = i.strip("[]")

        #plot scatterplot of embeddings
        plot_df = pd.DataFrame({'X': embeddings[:, 0], 'Y': embeddings[:, 1], 'Z': embeddings[:, 2], 'Element': merged_list})
        plot_df['Color'] = plot_df['Element'].apply(lambda x: 'blue' if x in very_similar else 'red' if x == query_text else 'green' if x in less_similar else 'grey')

        if option == 'Top Similar Diseases':
                option_df = plot_df[(plot_df['Element'].isin(very_similar)) | (plot_df['Element'] == query_text)]

        elif option == 'More Similar Diseases':
                option_df = plot_df[(plot_df['Element'].isin(very_similar)) | (plot_df['Element'] == query_text) | (plot_df['Element'].isin(less_similar))]

        elif option == 'All':
                option_df = plot_df


        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(go.Scatter3d(
        x=option_df['X'],
        y=option_df['Y'],
        z=option_df['Z'],
        mode='markers',
        marker=dict(
            color=option_df['Color'],
            opacity=0.7
        ),
        text=option_df['Element'],
        hoverinfo='text'
        ))

        # Connect black element with all red elements
        black_element = option_df[option_df['Color'] == 'red']
        red_elements = option_df[option_df['Color'] == 'blue']

        for index, row in black_element.iterrows():
                for red_index, red_row in red_elements.iterrows():
                    fig.add_trace(go.Scatter3d(
                        x=[row['X'], red_row['X']],
                        y=[row['Y'], red_row['Y']],
                        z=[row['Z'], red_row['Z']],
                        mode='lines',
                        line=dict(
                            color='black',
                            width=2
                        ),
                        showlegend=True
                    ))

        # Update layoutx
        fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(l=0, r=0, b=0, t=0)
        )

        st.plotly_chart(fig)

        #creating empty df
        final_df = pd.DataFrame(columns=['Doctor', 'url', 'Speciality', 'Address','Distance(miles)','Timings','Insurance'])
        #filter based on similar diseases
        for i in similar_text:
                temp_df = df[df['Speciality'].apply(lambda x: i in x)]
                final_df = final_df.append(temp_df)
                final_df = final_df.drop_duplicates(subset=['Doctor'])
                final_df = final_df[final_df['Timings'] != "[]"]
                final_df = final_df[0:5]

        st.text(" ")
        st.text(" ")
        st.title('List of Doctors')
        st.markdown("Lets search doctors that treat the conditions we matched using word embeddings:")
        st.write('---')
        for i, row in final_df.iterrows():
                doctor_name = row['Doctor']
                speciality = row['Speciality']
                clinic_distance = row['Distance(miles)']

                # Convert the list of specialities to a string
                speciality_str = ', '.join(speciality)
                # Remove the square brackets and single quotes
                # speciality_str = speciality_str.replace("[", "").replace("]", "").replace("'", "")


                # Highlight keywords in the speciality
                for keyword in similar_text:
                    if keyword in speciality_str:
                        speciality_str = speciality_str.replace(keyword, f"**<span style='background-color: lightgreen;'>{keyword}</span>**")

                # Display the doctor's details
                st.markdown(f"**{doctor_name}**")
                st.markdown(f"**Speciality:** {speciality_str}", unsafe_allow_html=True)
                st.write('**Distance (miles):**', clinic_distance)
                st.write('---')

        # Display the map in Streamlit
        st.title('Doctor Map')
        st.write('Map showing Santa Clara University and clinic locations')

        map = create_map(final_df)
        folium_static(map)

        st.text(" ")
        #plot timeline code
        st.title("When are the doctors Available ?")
        final_df['Timings'] = final_df['Timings'].str.strip('[]').str.split(', ')
        final_df['Extracted_Timings'] = final_df['Timings'].apply(extract_timings)
        final_df = pd.concat([final_df.drop('Extracted_Timings', axis=1), final_df['Extracted_Timings'].apply(pd.Series)], axis=1)
        # Reorder the columns
        column_order = ['Doctor', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        avail_df = final_df[column_order]

        day = st.radio(
                "Find Availability:",
                ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')) 

        df_name = availability(avail_df, day)

        fig = px.timeline(df_name, x_start="start", x_end="end", y="Doctor")
        st.plotly_chart(fig)

        st.header("Precaution better than cure")
        url = 'https://bhavnasjain.github.io/data-visualization/'

        st.markdown(f'''
        <a href={url}><button style="background-color:GreenYellow;">Stay Fit !</button></a>
        ''',
        unsafe_allow_html=True)

