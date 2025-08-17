import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
      menu_title = "Vietnam CO2 Emissions",
      options = ["Line Plots","Tile Plots","Facet Plots","Scatterplots"],
      menu_icon = "cast",
      default_index = 0
    )
if selected == "Line Plots":
    st.header('Line Plots')
    lp1,lp2 = st.columns(2)
    with st.container():
        lp1.write("Global CO2 Emissions Per Person (1800-2019)")
        lp2.write("Top 10 Emissions-producing Countries in 2010 (1900–2014)")
    with lp1:
        image_url1 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/main/png/World_CO2.png"
        st.image(image_url1, caption="Global CO2 Emissions Per Person (1800-2019)")
        st.caption("CO2 emissions have risen exponentially since the 20th century... Perhaps this change relates to the Industrial Revolution, which has forever changed the way the world moves around and produces things.")
    with lp2:
        image_url2 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/main/png/Top10_CO2_lineplot.png"
        st.image(image_url2, caption="Top 10 Emissions-producing Countries in 2010 (1900–2014)")
        st.caption("It is important to note that the traditional industrial nations like Germany, the UK, and Russia show earlier growth patterns, with peaks occurring mid-century before stabilizing or declining. The graph illustrates the global shift in emissions from traditional Western industrial powers to emerging Asian economies, particularly China and India.")
if selected == "Tile Plots":
    st.header('Tile Plots')
    tp = st.columns(1)[0]
    with st.container():
        tp.write("Top 10 CO₂ Emission-producing Countries\nOrdered by Emissions Produced in 2014")
    with tp:
        image_url3 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/main/png/Top10_CO2_tile.png"
        st.image(image_url3, caption="Top 10 CO₂ Emission-producing Countries\nOrdered by Emissions Produced in 2014")
        st.caption("ADD DESCRIPTION")

if selected == "Facet Plots":
    st.header('Facet Plots')
    fp = st.columns(1)[0]
    with st.container():
        fp.write("Distribution of Indications by Year and Value")
    with fp:
        image_url4 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/main/png/Agg_facet_plot.png"
        st.image(image_url4, caption="Distribution of Indications by Year and Value")
        
if selected == "Scatterplots":
    st.header('Scatterplots')
    sc1, sc2 = st.columns(2)
    sc3 = st.columns(1)[0]
    with st.container():
        sc1.write("Vietnam CO2 Emissions and Temperature, separated")
        sc2.write("Vietnam CO2 Emissions and Temperature, Unscaled (1980-2014)")
        sc3.write("Vietnam CO2 Emissions and Temperature, Scaled (1980–2014)")
    with sc1:
        image_url5 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/main/png/CO2_temp_Vietnam_facet.png"
        st.image(image_url5, caption="Vietnam CO2 Emissions and Temperature, separated")
    with sc2:
        image_url6 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/main/png/lin_reg_unscaled.png"
        st.image(image_url6, caption="Vietnam CO2 Emissions and Temperature, Unscaled (1980-2014)")
    with sc3:
        image_url7 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/main/png/Vietnam_emissions_temp_scaled.png"
        st.image(image_url7, caption="Vietnam CO2 Emissions and Temperature, Scaled (1980–2014)")

