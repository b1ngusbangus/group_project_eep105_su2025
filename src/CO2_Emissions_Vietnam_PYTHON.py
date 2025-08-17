import os
import matplotlib.pyplot as plt
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

# DATA WRANGLING

# 1. CO2
melted_CO2 = pd.melt(co2pp, id_vars = ['country'],
                                   var_name = 'Year',
                                    value_name = 'Emissions')

#renaming the columns, converting Year to numeric and adding a Label column
melted_CO2 = melted_CO2.rename(columns={'country': 'Country'})
melted_CO2['Year'] = pd.to_numeric(melted_CO2['Year'], errors = 'coerce')
melted_CO2['Label'] = 'CO2 Emissions (Metric Tons)'

# 2. GDP
melted_gdp_growth =pd.melt(
    gdp,
    id_vars = ['country'],
    var_name = 'Year',
    value_name = 'gdp')
melted_gdp_growth = melted_gdp_growth.rename(columns={'country': 'Country'})
melted_gdp_growth['Year'] = pd.to_numeric(melted_gdp_growth['Year'], errors = 'coerce')
melted_gdp_growth['Label'] = 'GDP Growth/Capita (%)'
melted_gdp_growth = melted_gdp_growth.rename(columns = {'gdp' : 'GDP'})
melted_gdp_growth = melted_gdp_growth[['Country', 'Year', 'GDP', 'Label']]

melted_gdp_growth = melted_gdp_growth[(melted_gdp_growth['Year'] <= 2014)]

# 3. Energy use
melted_energy_use =pd.melt(
    energy,
    id_vars = ['country'],
    var_name = 'Year',
    value_name = 'energy')
melted_energy_use = melted_energy_use.rename(columns={'country': 'Country'})
melted_energy_use['Year'] = pd.to_numeric(melted_energy_use['Year'], errors = 'coerce')
melted_energy_use['Label'] = 'Energy Use (kg, oil-eq./capita)'
melted_energy_use = melted_energy_use.rename(columns = {'energy' : 'Energy'})

melted_energy_use = melted_energy_use[(melted_energy_use['Year'] <= 2014)]

# Vietnam specific data
# 4. Disasters
#Getting the disaster year
vietnam_disasters['Year'] = vietnam_disasters['DisNo.'].astype(str).str[:4]
#Getting the disaster count grouped by year
disaster_counts = vietnam_disasters.groupby('Year').size().reset_index(name='Disaster_Count')

disaster_counts['Year'] = disaster_counts['Year'].astype(int)
disaster_counts = disaster_counts[(disaster_counts['Year'] <= 2014)]

disaster_counts['Country'] = 'Vietnam'
disaster_counts['Label'] = 'Number of Disasters'
disaster_counts['Indicator'] = 'Disasters'

# 5. Temperature
temp.rename(columns={'Category': 'Year', 'Annual Mean': 'Value'}, inplace=True)
temp = temp[(temp['Year'] <= 2014)]

temp['Country'] = 'Vietnam'
temp['Label'] = 'Temperature (Celsius)'
temp['Indicator'] = 'Temperature'

# JOINING THE DATA
data_wide = (
    pd.merge(melted_CO2, melted_gdp_growth, on=["Country", "Year", "Label"], how="outer")
    .merge(melted_energy_use, on=["Country", "Year", "Label"], how="outer")
)

#creating the long version of the data
data_long = pd.melt(
    data_wide,
    id_vars=['Country', 'Year', 'Label'],
    var_name='Indicator',
    value_name='Value'
)

data_long = pd.concat([data_long, disaster_counts, temp],
        axis = 0,
        ignore_index = True)
sorted_data_long = data_long.sort_values(['Label', 'Country'])

#Creating a column Region to indicate whether the Country is Vietnam or Rest of the World
data_long['Region'] = np.where(data_long['Country'] == "Vietnam",
                             "Vietnam",
                             "Rest of the World")
data_long = data_long.sort_values(by = ['Country', 'Year'])
#Selecting columns and getting first 6 rows
f_data_long = data_long.filter(items=['Country', 'Year', 'Label', 'Indicator', 'Value', 'Region'])

#Creating dataframe that is sorted by country name
data_long_with_miss = data_long.sort_values('Country')
#Making sure that the order of "Label" is what we want
label_priority = {
    'CO2 Emissions (Metric Tons)': 0,
    'GDP Growth/Capita (%)': 1,
    'Energy Use (kg, oil-eq./capita)': 2
}
f_data_long = f_data_long.sort_values(
    by=['Country', 'Year', 'Label'],
    key=lambda x: x.map(label_priority)
).reset_index(drop=True)
#Dropping NA values and sorting values by country and getting first 6 rows
f_data_long = f_data_long.dropna(subset=['Value'])
f_data_long.sort_values(by = ['Country'])

# DATA VISUALIZATION
# CO2 emissions
# 1. Aggregated sum of all countries, line plot: 1751-2014
f_data_long = f_data_long.dropna(subset = ['Value'])
data_long_co2 = f_data_long[f_data_long['Indicator'] == 'Emissions']
sum_co2 = data_long_co2.groupby(['Year']).sum()
sum_co2 = sum_co2.drop(columns = ["Label", "Country", "Indicator", "Region"], axis=1)
sum_co2 = sum_co2.rename(columns = {'Value':'Emissions'}).reset_index()
sum_co2['Emissions'] = pd.to_numeric(sum_co2['Emissions'], errors='coerce')

# PLOT THE FIRST FIGURE!
sb.set_style("darkgrid")
plt.figure(figsize=(9, 6))
fig1, ax = plt.subplots(figsize=(10, 6))
ax = sb.lineplot(data=sum_co2, x="Year", y="Emissions", color="#304d6d")
ax.set_ylabel("Emissions (Metric Tonnes)")
plt.figtext(
    0.75, -0.03,                       # x and y position (0-1 range in figure coords)
    "Limited to reporting countries",
    ha="center", fontsize=9, color="gray"
)
ax.set_title("Global CO2 Emissions Per Person (1800-2019)");

# Create the directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# save png for later use
# fig1.savefig("plots/World_CO2.png", dpi = 300, bbox_inches = 'tight')

# 2. Top 10 countries in 2010 (1900-2014): Line plot
top_10_countries = data_long_co2[data_long_co2["Year"] == 2014]
top_10_countries = top_10_countries.sort_values(by="Value", ascending=False).head(10)

#Creating the top 10 list of countries
top_10_list = top_10_countries["Country"].tolist()

#Creating a dataframe where it shows the top 10 counntries
top_10_df = data_long_co2[(data_long_co2["Year"] <= 2014) &
                                    (data_long_co2["Country"].isin(top_10_list)) &
                                    (data_long_co2["Year"] >= 1900)
                                    ]
#Making the items in the dataframe a category type
top_10_df = top_10_df.copy()
top_10_df["Country"] = top_10_df["Country"].astype('category')
top_10_df["Country"] = top_10_df["Country"].cat.remove_unused_categories()

# PLOT THE SECOND FIGURE!
#styling the type of graph
sb.set_style("darkgrid")
#Creating the figure for the tope 10 emitting countries
plt.figure(figsize=(12, 6))
ax = sb.lineplot(
    data=top_10_df[top_10_df["Country"].isin(top_10_list)],
    x="Year",
    y="Value",
    hue="Country",
    palette="viridis",
    estimator=None
)
#Adding titles and labels
ax.set_title("Top 10 Emissions-producing Countries in 2010 (1900–2014)", fontsize=16)
ax.set_ylabel("Emissions (Metric Tonnes)")

plt.legend(loc='upper left')
plt.figtext(0.75, -0.03, "Ordered by Emissions Produced in 2014", ha="center", fontsize=9, color="gray");

# 3. Top 10 countries in 2010 (1900-2014): Tile plot
# filter the data accordingly
filtered_data = (data_long_co2
                 [data_long_co2['Country'].isin(top_10_list)]
                 [data_long_co2['Year'] >= 1900])

# get ordering of countries by their last (2014) value
country_order = (filtered_data
                 [filtered_data['Year'] == filtered_data['Year'].max()]
                 .sort_values('Value', ascending=False)['Country'].tolist())

# create pivot table for heatmap
heatmap_data = filtered_data.pivot(index='Country', columns='Year', values='Value')
# reorder rows according to the country_order
heatmap_data = heatmap_data.reindex(country_order)

# ensure your data is properly formatted
# reset index if Country is currently the index
if 'Country' not in heatmap_data.columns:
    heatmap_data_working = heatmap_data.reset_index()
else:
    heatmap_data_working = heatmap_data.copy()

# set Country as index if it's not already
if 'Country' in heatmap_data_working.columns:
    heatmap_data_working = heatmap_data_working.set_index('Country')

# ensure all columns are numeric
# convert all columns to numeric, replacing any non-numeric values with NaN
for col in heatmap_data_working.columns:
    heatmap_data_working[col] = pd.to_numeric(heatmap_data_working[col], errors='coerce')

print("Data shape:", heatmap_data_working.shape)
print("Data types:", heatmap_data_working.dtypes.unique())

# handle problematic values for log transformation
# replace NaN, zero, and negative values
log_data = heatmap_data_working.copy()

# replace zeros and negative values with small positive number
log_data = log_data.where(log_data > 0, np.nan)  # replace <=0 with NaN
log_data = log_data.fillna(1)  # fill NaN with 1 (log(1) = 0)

# apply log transformation using pandas
log_data = log_data.apply(lambda x: np.log(x))

# PLOT THE THIRD FIGURE!
plt.figure(figsize=(15, 8))
fig2, ax2 = plt.subplots(figsize=(15, 8))

# create heatmap
mask = log_data.isna()  # mask any remaining NaN values
ax2 = sns.heatmap(log_data,
                  cmap='viridis',
                  mask=mask,
                  cbar_kws={'label': 'Ln(CO₂ Emissions (Metric Tonnes))'})

# customize the plot
plt.title('Top 10 CO₂ Emission-producing Countries\nOrdered by Emissions Produced in 2014',
          fontsize=16, pad=20)

# set x-axis ticks to show every 5 years from 1900 to 2014
years = list(range(1900, 2015, 5))
year_positions = []
year_labels = []

# find positions of years that exist in the data
for year in years:
    # check both integer and string versions
    year_candidates = [year, str(year)]
    for year_candidate in year_candidates:
        if year_candidate in log_data.columns:
            try:
                pos = log_data.columns.get_loc(year_candidate)
                year_positions.append(pos + 0.5)  # center the tick
                year_labels.append(str(year))
                break
            except:
                continue

plt.xticks(year_positions, year_labels, rotation=90, fontsize=12)
plt.yticks(np.arange(len(log_data.index)) + 0.5, log_data.index, rotation=0, fontsize=12)

# remove axis labels
plt.xlabel('')
plt.ylabel('')

# adjust layout and save
plt.tight_layout()
# fig2.savefig("plots/Top10_CO2_tile.png", dpi=300, bbox_inches='tight')
plt.show();

# CO2, GDP, & EMISSIONS: COMPARE VIETNAM WITH THE REST OF THE WORLD
# 1. Faceted plot

# define a function to clean the data
def clean_value_column(df):
    """
    Clean the 'Value' column by converting string representations to numbers
    Handles formats like: '10.3k', '1.2M', '500', etc.
    """
    df = df.copy()

    def convert_value(val):
        if pd.isna(val):
            return np.nan

        # convert to string and clean
        val_str = str(val).strip().lower()

        # handle empty strings
        if val_str == '' or val_str == 'nan':
            return np.nan

        # handle already numeric values
        try:
            return float(val_str)
        except ValueError:
            pass

        # handle k, m, b suffixes
        multipliers = {
            'k': 1000,
            'm': 1000000,
            'b': 1000000000,
            't': 1000000000000
        }

        # check if it ends with a multiplier
        for suffix, multiplier in multipliers.items():
            if val_str.endswith(suffix):
                try:
                    number = float(val_str[:-1])
                    return number * multiplier
                except ValueError:
                    return np.nan

        # if we can't convert it, return NaN
        return np.nan

    # apply the conversion
    df['Value'] = df['Value'].apply(convert_value)
    return df

# clean the data first
print("Cleaning data...")
filtered_data = data_long[~data_long['Indicator'].isin(['Disasters', 'Temperature'])]

# clean the Value column
filtered_data = clean_value_column(filtered_data)
# remove rows with NaN values if needed
filtered_data = filtered_data.dropna(subset=['Value'])

# get unique indicators and regions for faceting
indicators = filtered_data["Indicator"].unique()
regions = filtered_data['Region'].unique()
n_indicators = len(indicators)
n_regions = len(regions)

# PLOT THE TILE PLOT FIGURE!
fig, axes = plt.subplots(n_indicators, n_regions,
                        figsize=(5 * n_regions, 4 * n_indicators),
                        sharex=True)

# Handle cases where there's only one indicator or region
if n_indicators == 1 and n_regions == 1:
    axes = [[axes]]
elif n_indicators == 1:
    axes = [axes]
elif n_regions == 1:
    axes = [[ax] for ax in axes]

# Plot each combination of Indicator and Region
for i, indicator in enumerate(indicators):
    for j, region in enumerate(regions):
        ax = axes[i][j]

        # Filter data for this specific indicator and region
        subset = filtered_data[
            (filtered_data['Indicator'] == indicator) &
            (filtered_data['Region'] == region)
        ]

        if not subset.empty:
            # Plot lines for each country in this subset
            for country in subset['Country'].unique():
                country_data = subset[subset['Country'] == country]

                # Sort by year to ensure proper line connection
                country_data = country_data.sort_values('Year')

                # Additional check to ensure we have numeric data
                if len(country_data) > 0 and country_data['Value'].notna().any():
                    try:
                        ax.plot(country_data['Year'], country_data['Value'],
                               linewidth=1, alpha=0.7, label=country)
                    except Exception as e:
                        print(f"Error plotting {country} in {region}, {indicator}: {e}")
                        continue

        # Add strip labels (equivalent to strip.text)
        # Column headers (Region) at the top
        if i == 0:
            ax.set_title(region, fontsize=16, fontweight='bold', pad=20)

        # Row headers (Indicator) on the right
        if j == n_regions - 1:
            ax.text(1.02, 0.5, indicator, transform=ax.transAxes,
                   rotation=90, verticalalignment='center',
                   fontsize=16, fontweight='bold')

        # Apply custom theme styling (equivalent to my_theme)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        # Remove individual subplot labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Format y-axis to show values in a readable format
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Set overall title and labels
fig.suptitle('Distribution of Indicators by Year and Value',
            fontsize=18, y=0.98)

# Add shared axis labels
fig.text(0.5, 0.02, 'Year', ha='center', fontsize=14)
fig.text(0.02, 0.5, 'Indicator Value', va='center', rotation='vertical',
         fontsize=14)

# Adjust layout to accommodate labels and titles
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.85)

plt.show();

# CO2 EMISSIONS AND TEMPERATURE IN VIETNAM
# Scatterplots
# 1. CO2 Emissions and Temperature, separately

# ensure all values under Year are int type
year_int = data_long["Year"].astype(int)
data_long["Year"] = year_int

# ENSURE YEAR IS OF INT TYPE
data_long["Year"] = data_long["Year"].astype(int).tolist()

# filter data to only include the Emissions and Temperature indicators from 1980-2014
filtered_data_long = data_long[
    (data_long['Year'] >= 1980) &
    (data_long['Year'] <= 2014) &
    (data_long['Country'] == 'Vietnam') &
    (data_long['Indicator'].isin(['Emissions', 'Temperature']))
]

# PLOT THE FACETED FIGURE!
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Get unique indicators for faceting
indicators = filtered_data_long['Indicator'].unique()

for i, indicator in enumerate(indicators):
    # Filter data for current indicator
    subset = filtered_data_long[filtered_data_long['Indicator'] == indicator]

    # Plot points (equivalent to geom_point)
    axes[i].scatter(subset['Year'], subset['Value'], alpha=0.7)

    # Add smooth line (equivalent to geom_smooth with loess)
    # Using lowess (locally weighted scatterplot smoothing) as pandas equivalent to loess
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(subset['Value'], subset['Year'], frac=0.3)
    axes[i].plot(smoothed[:, 0], smoothed[:, 1], color='#304d6d', linewidth=2)

    # Set y-axis to free scale (equivalent to scales = "free_y")
    # This happens automatically with subplots

    # Set subplot title using Label column (equivalent to facet_wrap Label)
    if 'Label' in subset.columns:
        axes[i].set_title(subset['Label'].iloc[0], fontsize=14)
    else:
        axes[i].set_title(indicator, fontsize=14)

    # Remove axis titles (equivalent to axis.title = element_blank())
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

    # Customize tick labels
    axes[i].tick_params(axis='both', which='major', labelsize=12, colors='black')

# Set x-axis breaks and labels (equivalent to scale_x_continuous)
x_ticks = range(1980, 2015, 5)
plt.xticks(x_ticks, x_ticks, rotation=90, fontsize=12, color='black')

# Set main title
fig.suptitle('Vietnam Emissions and Temperatures (1980-2014)', fontsize=16)

# Apply classic theme styling
plt.style.use('classic')
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Adjust layout
plt.tight_layout()
# save graph before showing
# fig.savefig('plots/CO2_temp_Vietnam_facet.png', dpi=300, bbox_inches='tight')
plt.show();

# 2. CO2 Emissions and Temperature, together
wide_viet = (
    data_long
    .query("Country == 'Vietnam' & Year >= 1980 & Year <= 2014")
    # .drop(columns=['Label'])
    .pivot_table(index=['Country', 'Year'],
                 columns='Indicator',
                 values='Value',
                 aggfunc='first',
                 observed=True)
    .reset_index()
)

# create a clean copy and convert to numeric
wide_viet_clean = wide_viet.copy()

# convert columns to numeric, replacing any non-numeric values with NaN
wide_viet_clean['Emissions'] = pd.to_numeric(wide_viet_clean['Emissions'], errors='coerce')
wide_viet_clean['Temperature'] = pd.to_numeric(wide_viet_clean['Temperature'], errors='coerce')

# remove rows with NaN values
wide_viet_clean = wide_viet_clean.dropna(subset=['Emissions', 'Temperature'])

# ensure we have valid numeric data
wide_viet_clean = wide_viet_clean[
    (wide_viet_clean['Emissions'].notna()) &
    (wide_viet_clean['Temperature'].notna()) &
    (np.isfinite(wide_viet_clean['Emissions'])) &
    (np.isfinite(wide_viet_clean['Temperature']))
]

# PLOT THE LINEAR REGRESSION FIGURE
sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(9, 6))

# create scatter plot
ax.scatter(wide_viet_clean['Emissions'], wide_viet_clean['Temperature'], c="#304d6d")

# set labels and title
ax.set_xlabel('Emissions (Metric Tonnes)', fontsize=14)
ax.set_ylabel('Temperature (Fahrenheit)', fontsize=14)
ax.set_title('Vietnam Emissions and Temperature (1980-2014)', fontsize=16)

# customize tick labels
if len(wide_viet_clean) > 0:
    ax.set_xticks(np.linspace(wide_viet_clean['Emissions'].min(),
                              wide_viet_clean['Emissions'].max(),
                              4))

# remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# add regression line
if len(wide_viet_clean) >= 2:
    sns.regplot(
        data=wide_viet_clean,
        x="Emissions",
        y="Temperature",
        scatter=False,
        ci=None,
        ax=ax,
        color="#B1303F"
    )

# show the plot
plt.tight_layout()
plt.show()

# SCALED EMISSIONS & TEMPERATURE RELATIONSHIP
# convert to numeric and clean
wide_viet_clean = wide_viet.copy()

# convert to numeric, coercing any non-numeric values to NaN
wide_viet_clean['Emissions'] = pd.to_numeric(wide_viet_clean['Emissions'], errors='coerce')
wide_viet_clean['Temperature'] = pd.to_numeric(wide_viet_clean['Temperature'], errors='coerce')

# remove rows with NaN values
wide_viet_clean = wide_viet_clean.dropna(subset=['Emissions', 'Temperature'])

print(f"Data points: {len(wide_viet_clean)}")

if len(wide_viet_clean) >= 2:
    # scale the data like R's scale()
    wide_viet_clean['Emissions_scaled'] = (
        (wide_viet_clean['Emissions'] - wide_viet_clean['Emissions'].mean()) /
        wide_viet_clean['Emissions'].std(ddof=0)
    )
    wide_viet_clean['Temperature_scaled'] = (
        (wide_viet_clean['Temperature'] - wide_viet_clean['Temperature'].mean()) /
        wide_viet_clean['Temperature'].std(ddof=0)
    )

# scale the data like R's scale()
wide_viet_clean['Emissions_scaled'] = (wide_viet_clean['Emissions'] - wide_viet_clean['Emissions'].mean()) / wide_viet_clean['Emissions'].std(ddof=0)
wide_viet_clean['Temperature_scaled'] = (wide_viet_clean['Temperature'] - wide_viet_clean['Temperature'].mean()) / wide_viet_clean['Temperature'].std(ddof=0)

# create scatter plot + regression line
plt.figure(figsize=(8,6))
sns.regplot(
    x='Emissions_scaled',
    y='Temperature_scaled',
    data=wide_viet_clean,
    ci=None,  # se = FALSE in R
    scatter_kws={'s': 40}  # point size
)

# labels and title
plt.title(r"Vietnam CO$_2$ Emissions and Temperature (1980–2014)", fontsize=16)
plt.xlabel("Scaled Emissions (Metric Tonnes)", fontsize=14)
plt.ylabel("Scaled Temperature (Fahrenheit)", fontsize=14)

# tick label sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# grid style
plt.grid(True, linestyle='-', linewidth=0.5, color='#696969', alpha=0.3)

plt.show()








