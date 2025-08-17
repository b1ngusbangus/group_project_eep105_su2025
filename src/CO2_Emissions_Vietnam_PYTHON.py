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
        image_url1 = ""
        st.image(image_url1, caption="My Image")
    with lp2:
        image_url2 = ""
        st.image(image_url2, caption="My Image")
if selected == "Tile Plots"
    tp = columns(1)
    with st.container():
        tp.write("Top 10 CO₂ Emission-producing Countries\nOrdered by Emissions Produced in 2014")
    with tp:
        image_url3 = "https://raw.githubusercontent.com/username/repository_name/branch/path/to/your_image.png"
        st.image(image_url3, caption="My Image")
        
if selected == "Facet Plots"
    fp = columns(1)
    with st.container():
        tp.write("Distribution of Indications by Year and Value")
    with fp:
        image_url4 = "https://raw.githubusercontent.com/username/repository_name/branch/path/to/your_image.png"
        st.image(image_url4, caption="My Image")
        
if selected == "Scatterplots"
    sc1, sc2 = columns(2)
    sc3 = columns(1)
    with st.container():
        sc1.write("Vietnam Emissions and Temperature")
        sc2.write("Vietnam Emissions and Temperature (1980-2014)")
        sc3.write("Vietnam CO2 Emissions and Temperature (1980–2014)")
    with sc1:
        image_url = "https://raw.githubusercontent.com/username/repository_name/branch/path/to/your_image.png"
        st.image(image_url, caption="My Image")
    with sc2:
        image_url = "https://raw.githubusercontent.com/username/repository_name/branch/path/to/your_image.png"
        st.image(image_url, caption="My Image")
    with sc3:
        image_url = "https://raw.githubusercontent.com/username/repository_name/branch/path/to/your_image.png"
        st.image(image_url, caption="My Image")

