import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

@st.cache
def data_load():
    data = pd.read_csv('streamlit_data.csv')
    return data

def main():
    data = data_load()
    page = st.sidebar.selectbox("Choose a Page", ["Homepage","Exploration and analysis"])

    if page == 'Homepage':
        
        st.header('Data Exploration and analysis')
        st.write("head of the data  ")
        st.write(data.head())
    elif page == 'Exploration and analysis':
        map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])

        st.map(map_data)

       

if __name__ == "__main__":
    main()
