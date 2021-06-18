import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import random
import csv
import statistics

df = pd.read_csv('savings_data.csv')

fig = px.scatter(df, y = 'quant_saved', color = 'female')

fig.show()

with open('savings_data.csv', newline = '') as f:
    reader = csv.reader(f)
    savings_data = list(reader)

savings_data.pop(0)

total_entries = len(savings_data)

total_female = 0

for data in savings_data:
    if int(data[1]) == 1:
        total_female += 1

fig = go.Figure(go.Bar(x = ['Female', 'Male'], y = [total_female, [total_entries - total_female]]))

fig.show()

all_savings = []

for data in savings_data:
    all_savings.append(float(data[0]))

print(f"Mean of Savings - {statistics.mean(all_savings)}")

print(f"Mode of Savings - {statistics.mode(all_savings)}")

print(f"Median of Savings - {statistics.median(all_savings)}")

female_savings = []

male_savings = []

for data in savings_data:
  if int(data[1]) == 1:
    female_savings.append(float(data[0]))
  else:
    male_savings.append(float(data[0]))

print(f"Mean of Female Savings - {statistics.mean(female_savings)}")

print(f"Mode of Female Savings - {statistics.mode(female_savings)}")

print(f"Median of Female Savings - {statistics.median(female_savings)}")

print(f"Mean of Male Savings - {statistics.mean(male_savings)}")

print(f"Mode of Male Savings - {statistics.mode(male_savings)}")

print(f"Median of Male Savings - {statistics.median(male_savings)}")

print(f"Standard Deviation of Female Savings - {statistics.stdev(female_savings)}")

print(f"Standard Deviations of Male Savings - {statistics.stdev(male_savings)}")

print(f"Standard Deviations of All Savings - {statistics.stdev(all_savings)}")

wealthy = []

savings = []

for data in savings_data:
    if float(data[3]) != '0':
        wealthy.append(float(data[3]))
        savings.append(float(data[0]))

correlation = np.corrcoef(wealthy, savings)

print(f"Correlation between the wealth of the person and the savings - {correlation[0, 1]}")

fig = ff.create_distplot([df['quant_saved'].tolist()], ['Savings'], show_hist = False)

fig.show()

import seaborn as sns
import random

sns.boxplot(data = df, x = df['quant_saved'])

q1 = df['quant_saved'].quantile(0.25)
q3 = df['quant_saved'].quantile(0.75)
iqr = q3 - q1

print(f'Q1 - {q1}')
print(f'Q3 - {q3}')
print(f'IQR - {iqr}')

lower_whisker = q1 - 1.5 * iqr

upper_whisker = q3 + 1.5 * iqr

print(f'Upper Whisker - {upper_whisker}')

print(f'Lower Whisker - {lower_whisker}')

new_df = df[df['quant_saved'] < upper_whisker]

all_savings = new_df['quant_saved'].tolist()

print(f'Mean of Savings - {statistics.mean(all_savings)}')

print(f'Median of Savings - {statistics.median(all_savings)}')

print(f'Mode of Savings - {statistics.mode(all_savings)}')

print(f'Standard Deviaiton of Savings - {statistics.stdev(all_savings)}')

fig = ff.create_distplot([new_df['quant_saved'].tolist()], ['savings'], show_hist = False)

fig.show()

sampling_mean_list = []

for i in range(0, 1000):
  temp_list = []

  for j in range(100):
    temp_list.append(random.choice(all_savings))

    sampling_mean_list.append(statistics.mean(temp_list))

mean_sampling = statistics.mean(sampling_mean_list)

fig = ff.create_distplot([sampling_mean_list], ['Savings (Sampling)'], show_hist = False)

fig.show()