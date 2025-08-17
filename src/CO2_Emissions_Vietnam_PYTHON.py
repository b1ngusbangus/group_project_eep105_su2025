import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import seaborn as sns
import pickle
from pathlib import Path
from PIL import Image

#getting data through reading their csvs
co2pp = pd.read_csv("yearly_co2_emissions_1000_tonnes.csv")
vietnam_disasters = pd.read_csv("disasters_vietnam.csv")
temp = pd.read_csv("temperature_vietnam.csv")
gdp = pd.read_csv("gdp_pcap.csv")
energy = pd.read_csv('energy_per_person.csv')

with st.sidebar:
    selected = option_menu(
      menu_title = "Main Menu",
      options = ["Global Emissions","Top 10 Countries","Tile Plot","Facet Plot","Scatterplots"],
      menu_icon = "cast",
      default_index = 0
    )

