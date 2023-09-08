# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


# Load & read the dataset
data= pd.read_csv("/Users/susoeresia-eke/Desktop/F1Drivers_Dataset.csv")
df = pd.DataFrame(data)

# Data Exploration

# Print the shape (number of rows, number of columns)
print(df.shape)
# Check the first few rows of the dataset
print(df.head())
# Check summary statistics for numerical columns and round them out
print(df.describe().round())
# Check data types and missing values
print(df.info())

#Check null and duplicate values
print(data.isnull().sum())
print(data.duplicated().sum())

# Check data types
data_types = df.dtypes

# Seasons data type is object, Extract years and convert to integers
# Load the dataset with custom parsing for 'Season' column
def parse_seasons(seasons_str):
    # Remove square brackets and split by comma, then convert to integers
    return [int(year.strip()) for year in seasons_str.strip('[]').split(',')]

# Load the CSV file with custom parsing for 'Season' column
data = pd.read_csv("/Users/susoeresia-eke/Desktop/F1Drivers_Dataset.csv", converters={'Seasons': parse_seasons})

# Now, the 'Seasons' column contains lists of integers
print(data['Seasons'].head())

# Display the DataFrame
print(data)

# Explore the distribution of driver nationalities
plt.figure(figsize=(10, 8))
sns.countplot(data=df, x='Nationality', palette='viridis')
plt.title('Distribution of Driver Nationalities')
plt.xlabel('Nationality')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Preliminary Analysis
# Relationship between Championships and Race Wins
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Championships', y='Race_Wins', hue='Nationality', data=df, palette='Set2')
plt.title('Relationship between Championships and Race Wins')
plt.xlabel('Championships')
plt.ylabel('Race Wins')
plt.legend(title='Nationality', loc='upper right')
plt.show()

# Insights and Recommendations
# Calculate the average win rate for champions
avg_win_rate_champions = df[df['Champion'] == 1]['Win_Rate'].mean()
print(f'Average Win Rate for Champions: {avg_win_rate_champions:.2%}')
# Average Win Rate for Champions: 15.55%

# Use Case 3: Analysis of Race Win Rate
data['Win_Rate'] = (data['Race_Wins'] / data['Race_Entries']) * 100
plt.figure(figsize=(10, 8))
sns.histplot(data['Win_Rate'], bins=20, kde=True)
plt.title("Distribution of Race Win Rate")
plt.xlabel("Win Rate (%)")
plt.ylabel("Frequency")
plt.show()

# Deductions
# 1. Most drivers do not have championships, indicating that winning a championship is rare.
# 2. There is a positive correlation between championships and race wins.
# 3. The distribution of race win rates is right-skewed, with a few drivers having high win rates.



# Use Case 1: Nationality Analysis
#Nationality Analysis: Analyzing the distribution of the top 10 nationalities among Formula One drivers.

nationality_counts = data['Nationality'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=nationality_counts.values, y=nationality_counts.index, palette='Blues_d')
plt.title("Top 10 Nationalities in Formula One")
plt.xlabel("Count")
plt.ylabel("Nationality")
plt.show()
# UK and US nationalities are the most common- by a significant amount- among Formula One drivers in the dataset. This might indicate the popularity of the sport is highest in those two countries, along with the availability of infrastructure to train F1 drivers.



# Use Case 2: Decade-wise Analysis
# Decade-wise Analysis
#Decade-wise Analysis: Examining how race wins, pole positions, and podiums have evolved over the decades.


# Calculate the average number of Race Wins, Podiums, and Pole Positions for each decade
decade_stats = data.groupby('Decade').agg({
    'Race_Wins': 'mean',
    'Podiums': 'mean',
    'Pole_Positions': 'mean'
}).reset_index()

# Data Visualization
plt.figure(figsize=(10, 6))
plt.plot(decade_stats['Decade'], decade_stats['Race_Wins'], marker='o', label='Average Race Wins')
plt.plot(decade_stats['Decade'], decade_stats['Podiums'], marker='o', label='Average Podiums')
plt.plot(decade_stats['Decade'], decade_stats['Pole_Positions'], marker='o', label='Average Pole Positions')
plt.xlabel('Decade')
plt.ylabel('Average Count')
plt.title('Decade-wise Analysis of F1 Driver Performance')
plt.legend()
plt.grid(True)
plt.show()

# The analysis reveals a historical trend of continuous performance improvement among Formula 1 drivers from the 1950s to the early 2010s. This sustained growth in race wins, podiums, and pole positions reflects the cumulative effect of advancements in car technology, training regimes, and race strategies.
#Additionally,there is an observable decline in performance between the 2010s and 2020. This decline may be attributed to various factors, including regulations aimed at reducing car performance advantages, stricter engine and aerodynamic rules, and the emergence of dominant teams and drivers who temporarily reduced competitiveness across the field.
#Decade-wise Analysis: Examining how race wins and podiums have evolved over the decades.



# Use Case 3: Driver Success vs. Race Entries
#Driver Success vs. Race Entries: Investigating whether there's a clear relationship between the number of race entries and championships won.

# Data Cleaning
data['Championships'] = data['Championships'].fillna(0).astype(int)
data['Win_Rate'] = (data['Race_Wins'] / data['Race_Entries']) * 100

# Use Case 3: Driver Success vs. Race Entries
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Race_Entries', y='Race_Wins', data=data, alpha=0.5)
plt.title("Driver Success vs. Race Entries")
plt.xlabel("Race Entries")
plt.ylabel("Race Wins")
plt.show()

# Calculate Pearson correlation between Race Entries and Race Wins
correlation, p_value = pearsonr(data['Race_Entries'], data['Race_Wins'])
print(f"Pearson Correlation Coefficient: {correlation:.2f}")
#The correlation coefficient of 0.6 suggests that there is a moderate positive relationship between the number of race entries and the number of race wins.
# As drivers participate in more races (higher race entries), they tend to achieve a higher number of race wins, on average.



# Use Case 4: Championship and Win Rate Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Win_Rate', y='Championships', data=data, alpha=0.5)
plt.title("Championships vs. Win Rate")
plt.xlabel("Win Rate (%)")
plt.ylabel("Championships")
plt.show()

# 4.1 - Calculate Pearson correlation between Win Rate and Championships
correlation, p_value = pearsonr(data['Win_Rate'], data['Championships'])
print(f"Pearson Correlation: {correlation:.2f}")
print(f"P-Value: {p_value:.2f}")

#In this case, the correlation coefficient is positive and relatively strong (0.73), which means there is a positive linear relationship between the two variables.
#Win Rate and Championships: The scatter plot shows a positive correlation between a driver's win rate and the number of championships they have won. This suggests that drivers with higher win rates are more likely to achieve championships. The Pearson correlation coefficient confirms this positive relationship.
#ie. Drivers with higher win rates are more likely to become champions, indicating the importance of consistent race performance in winning championships.

# More Data Cleaning
data['Championships'] = data['Championships'].fillna(0).astype(int)
data['Win_Rate'] = (data['Race_Wins'] / data['Race_Entries']) * 100

# Use Case #5: Impact of Seasons Played on Performance
data['Seasons_Played'] = data['Seasons'].apply(len)

# Calculate average performance metrics based on the number of seasons played
seasons_performance = data.groupby('Seasons_Played').agg({
    'Race_Wins': 'mean',
    'Podiums': 'mean',
    'Championships': 'mean'
}).reset_index()

# Plotting the impact of seasons played on performance
plt.figure(figsize=(10, 6))
sns.barplot(x='Seasons_Played', y='Race_Wins', data=seasons_performance, palette='viridis')
plt.title("Impact of Seasons Played on Average Race Wins")
plt.xlabel("Seasons Played")
plt.ylabel("Average Race Wins")
plt.show()

#5-1 # Calculate Pearson correlation coefficient
correlation, p_value = pearsonr(seasons_performance['Seasons_Played'], seasons_performance['Race_Wins'])
print(f"Pearson Correlation Coefficient: {correlation:.2f}")
#The analysis shows a positive correlation between the number of seasons played and the average race wins for drivers.
# Drivers who have competed in more seasons tend to have a higher average number of race wins.