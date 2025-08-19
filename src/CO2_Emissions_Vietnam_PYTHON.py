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
        st.caption("We see that Germany had very low emission rates at the end of World War II. We also see that the US has consistently had high emission rates since 1900, but the emission rates in China recently surpassed that of the US. South Korea is notable for having very low CO2 emissions up until the end of WWII. South Korea has gone from rubble to incredible prosperity since the Korean War and we can see this in the years where environmental concerns weren't a big thing in the world yet.")

if selected == "Facet Plots":
    st.header('Facet Plots')
    fp = st.columns(1)[0]
    with st.container():
        fp.write("Distribution of Indications by Year and Value")
    with fp:
        image_url4 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/main/png/Agg_facet_plot.png"
        st.image(image_url4, caption="Distribution of Indications by Year and Value")
        st.caption("As seen in this facet plot, the GDP file does not have that large of an increase in GDP compared to other countries. This could be because Vietnam is a developing country. For Emissions, Vietnam does contribute quite a lot to emissions. This could be as a result to growing production for development. For Energy, Its exponential growth pattern mirrors the CO2 emissions data, remaining minimal until 1850, then showing steady growth through 1950, followed by dramatic acceleration in recent decades.")
        
if selected == "Scatterplots":
    st.header('Scatterplots')
    sc1, sc2 = st.columns(2)
    sc3 = st.columns(1)[0]
    sc4, sc5 = st.columns(2)
    sc6 = st.columns(1)[0]
    with st.container():
        sc1.write("Vietnam CO2 Emissions and Temperature, separated")
        sc2.write("Vietnam CO2 Emissions and Temperature, Unscaled (1980-2014)")
        sc3.write("Vietnam CO2 Emissions and Temperature, Scaled (1980–2014)")
        sc4.write("(Individual finding) US CO2 Emissions and Temperature, separated")
        sc5.write("US CO2 Emissions and Temperature, Unscaled (1980-2014)")
        sc6.write("US CO2 Emissions and Temperature, Scaled (1980–2014)")
    with sc1:
        image_url5 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/main/png/CO2_temp_Vietnam_facet.png"
        st.image(image_url5, caption="Vietnam CO2 Emissions and Temperature, separated")
        st.caption("We notice that temperature has fluctuated in Vietnam but generally increased over the years as its CO2 emissions have risen exponentially several decades following the Vietnam War. This demonstrates how Vietnam is becoming more of a developed country, producing lots of goods but also are not worried about environmental issues since they're primarily focused on developing the country, whatever it takes to achieve that.")
    with sc2:
        image_url6 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/main/png/lin_reg_unscaled.png"
        st.image(image_url6, caption="Vietnam CO2 Emissions and Temperature, Unscaled (1980-2014)")
        st.caption("We see a positive correlation between CO2 emissions and temperature in Vietnam. This confirms the fact that CO2 emissions are contributing to temperature warming which can be traced back to how Vietnam is becoming more of a developed country keen on slowly making its way to the global superpowers. Because of this, it doesn't take into account the negative environmental effects it's causing.")
    with sc3:
        image_url7 = "https://raw.githubusercontent.com/b1ngusbangus/group_project_eep105_su2025/main/png/Vietnam_emissions_temp_scaled.png"
        st.image(image_url7, caption="Vietnam CO2 Emissions and Temperature, Scaled (1980–2014)")
        st.caption("We see here that the datapoints are scattered but still shows a generally positive direction. Our t-value that we calculated is 2.73, meaning statistical significance at the 5% level. Therefore, we are able to reject the null hypothesis that temperature and CO2 emissions aren't correlated, which makes sense since CO2 emissions have largely shown to contribute to rising temperatures. Given that Vietnam is becoming more developed and is producing more CO2 emissions, we would expect to see its average temperatures rising in the coming years.")
    with sc4: 
        // ADD PNGS HERE

