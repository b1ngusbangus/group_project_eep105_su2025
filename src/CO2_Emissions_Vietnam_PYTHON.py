import streamlit as st

with st.sidebar:
    selected = option_menu(
      menu_title = "Main Menu",
      options = ["Line Plots","Tile Plots","Facet Plots","Scatterplots"],
      menu_icon = "cast",
      default_index = 0
    )
if selected == "Line Plots":
    st.header('Line Plots')
    lp1,lp2 = columns(2)
    with st.container():
        lp1.write("Global CO2 Emissions Per Person (1800-2019)")
        lp2.write("Top 10 Emissions-producing Countries in 2010 (1900–2014)")
    with lp1:
        image_url1 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/blob/main/png/World_CO2.png"
        st.image(image_url1, caption="Global CO2 Emissions Per Person (1800-2019)")
    with lp2:
        image_url2 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/blob/main/png/Top10_CO2_lineplot.png"
        st.image(image_url2, caption="Top 10 Emissions-producing Countries in 2010 (1900–2014)")
if selected == "Tile Plots":
    tp = columns(1)
    with st.container():
        tp.write("Top 10 CO₂ Emission-producing Countries\nOrdered by Emissions Produced in 2014")
    with tp:
        image_url3 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/blob/main/png/Top10_CO2_tile.png"
        st.image(image_url3, caption="Top 10 CO₂ Emission-producing Countries\nOrdered by Emissions Produced in 2014")
        
if selected == "Facet Plots":
    fp = columns(1)
    with st.container():
        tp.write("Distribution of Indications by Year and Value")
    with fp:
        image_url4 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/blob/main/png/Agg_facet_plot.png"
        st.image(image_url4, caption="Distribution of Indications by Year and Value")
        
if selected == "Scatterplots":
    sc1, sc2 = columns(2)
    sc3 = columns(1)
    with st.container():
        sc1.write("Vietnam CO2 Emissions and Temperature, separated")
        sc2.write("Vietnam CO2 Emissions and Temperature, Unscaled (1980-2014)")
        sc3.write("Vietnam CO2 Emissions and Temperature, Scaled (1980–2014)")
    with sc1:
        image_url = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/blob/main/png/CO2_temp_Vietnam_facet.png"
        st.image(image_url, caption="Vietnam CO2 Emissions and Temperature, separated")
    with sc2:
        image_url = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/blob/main/png/lin_reg_unscaled.png"
        st.image(image_url, caption="Vietnam CO2 Emissions and Temperature, Unscaled (1980-2014)")
    with sc3:
        image_url = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/blob/main/png/Vietnam_emissions_temp_scaled.png"
        st.image(image_url, caption="Vietnam CO2 Emissions and Temperature, Scaled (1980–2014)")

