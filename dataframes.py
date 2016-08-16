import pandas as pd
import urllib.request

# import pylab

# https://github.com/jadianes/data-science-your-way/blob/master/01-data-frames/README.md
# start

# tb_deaths_url_csv = 'https://docs.google.com/spreadsheets/d/12uWVH_IlmzJX_75bJ3IH5E-Gqx6-zfbDKNvZqYjUuso/pub?gid=0&output=CSV'
# tb_existing_url_csv = 'https://docs.google.com/spreadsheets/d/1X5Jp7Q8pTs3KLJ5JBWKhncVACGsg5v4xu6badNs4C7I/pub?gid=0&output=csv'
# tb_new_url_csv = 'https://docs.google.com/spreadsheets/d/1Pl51PcEGlO9Hp4Uh0x2_QM0xVb53p2UDBMPwcnSjFTk/pub?gid=0&output=csv'

local_tb_deaths_file = 'data/tb_deaths_100.csv'
local_tb_existing_file = 'data/tb_existing_100.csv'
local_tb_new_file = 'data/tb_new_100.csv'
#
# deaths_f = urllib.request.urlretrieve(tb_deaths_url_csv, local_tb_deaths_file) # country names -> row labels
# existing_f = urllib.request.urlretrieve(tb_existing_url_csv, local_tb_existing_file) # cells as numbers
# new_f = urllib.request.urlretrieve(tb_new_url_csv, local_tb_new_file)

deaths_df = pd.read_csv(local_tb_deaths_file, index_col = 0, thousands  = ',').T
existing_df = pd.read_csv(local_tb_existing_file, index_col = 0, thousands  = ',').T
new_df = pd.read_csv(local_tb_new_file, index_col = 0, thousands  = ',').T

# print(existing_df.head()) # first few lines
# print(existing_df.columns) # column names
# print(existing_df.index) # row names

deaths_df.index.names = ['year']
deaths_df.columns.names = ['country']
existing_df.index.names = ['year']
existing_df.columns.names = ['country']
new_df.index.names = ['year']
new_df.columns.names = ['country']
# print(existing_df) # assign proper names to rows and columns

# print(existing_df['United Kingdom'])
# print(existing_df.Spain)
# print(existing_df[['Spain', 'United Kingdom']])
# print(existing_df.Spain['1990'])
# print(existing_df.Spain.'1990') # wrong syntax
# print(existing_df[['Spain', 'United Kingdom']][2:7])
# print(existing_df[0:5])

# print(existing_df.iloc[0:2]) # positional index access
# print(existing_df.loc['1992':'2005']) # label access
# print(existing_df.loc[['1992','1998','2005'],['Spain','United Kingdom']])

# print(existing_df > 10)
# print(existing_df['United Kingdom'] > 10) # returns bool
# print(existing_df.Spain[existing_df['United Kingdom'] > 10]) # returns spain's #numbers at uk's true positions
# print(existing_df[existing_df > 10]) # returns NaN for fields of rows if <= 10
# print(existing_df.where(existing_df > 10, 0)) # prints 0 instead of NaN

# print(existing_df.sum())
# print(existing_df.sum(axis=1)) # sum by y axis (year)
# print(existing_df.apply(lambda x: x/10) # same result as applymap
# print(existing_df.applymap(lambda x: x/10)) # applymap is intended to be used with elements

# mean_cases_by_period = existing_df.groupby(lambda x: int(x) > 1999).mean() # group by condition
# mean_cases_by_period.index = ['1990-1999', '2000-2007']
# print(mean_cases_by_period)
# print(mean_cases_by_period[['United Kingdom', 'Spain', 'Colombia']])

# https://github.com/jadianes/data-science-your-way/blob/master/01-data-frames/README.md
# end


# https://github.com/jadianes/data-science-your-way/blob/master/02-exploratory-data-analysis/README.md
# start

# df_summary = existing_df.describe()
# print(df_summary)
# print(df_summary[['Spain','United Kingdom']])

# tb_pct_change_spain = existing_df.Spain.pct_change()
# print(tb_pct_change_spain) # percentage change starting from [0]
# print(tb_pct_change_spain.max())
# print(existing_df['United Kingdom'].pct_change().max())

# print(existing_df['Spain'].pct_change().argmax()) # index (year) of max()
# print(existing_df['Spain'].pct_change().idxmax()) # for newer pandas

# existing_df[['United Kingdom', 'Spain', 'Colombia']].plot()
# existing_df[['United Kingdom', 'Spain', 'Colombia']].boxplot()
# pylab.show()

# print(existing_df.apply(pd.Series.argmax, axis=1)) # max value's column name for each row

# deaths_total_per_year_df = deaths_df.sum(axis=1)
# existing_total_per_year_df = existing_df.sum(axis=1)
# new_total_per_year_df = new_df.sum(axis=1)
# world_trends_df = pd.DataFrame({
#            'Total deaths per 100K' : deaths_total_per_year_df,
#            'Total existing cases per 100K' : existing_total_per_year_df,
#            'Total new cases per 100K' : new_total_per_year_df},
#        index=deaths_total_per_year_df.index)
# print(world_trends_df)
# world_trends_df.plot(figsize=(12,6)).legend(
#     loc='center left',
#     bbox_to_anchor=(1, 0.5))
# pylab.show()

# deaths_by_country_mean = deaths_df.mean()
# deaths_by_country_mean_summary = deaths_by_country_mean.describe()
# print(deaths_by_country_mean_summary)
# existing_by_country_mean = existing_df.mean()
# existing_by_country_mean_summary = existing_by_country_mean.describe()
# print(existing_by_country_mean_summary)
# new_by_country_mean = new_df.mean()
# new_by_country_mean_summary = new_by_country_mean.describe()
# print(new_by_country_mean_summary)

# deaths_by_country_mean.order().plot(kind='bar', figsize=(24,6))
# pylab.show()

# countries beyond 1.5 times the inter quartile range (50%)
# deaths_outlier = deaths_by_country_mean_summary['50%'] * 1.5
# print(deaths_outlier)
# existing_outlier = existing_by_country_mean_summary['50%'] * 1.5
# print(existing_outlier)
# new_outlier = new_by_country_mean_summary['50%'] * 1.5
# print(new_outlier)

# Now compare with the outlier threshold
# outlier_countries_by_deaths_index = \
#     deaths_by_country_mean > deaths_outlier
# outlier_countries_by_existing_index = \
#    existing_by_country_mean > existing_outlier
# outlier_countries_by_new_index = \
#     new_by_country_mean > new_outlier

# print(outlier_countries_by_deaths_index)
# print(outlier_countries_by_existing_index)
# print(outlier_countries_by_new_index)

# num_countries = len(deaths_df.T)
# print(sum(outlier_countries_by_deaths_index) / num_countries)
# print(sum(outlier_countries_by_existing_index) / num_countries) # prevalence
# print(sum(outlier_countries_by_new_index) / num_countries) # incidence

# outlier_deaths_df = deaths_df.T[ outlier_countries_by_deaths_index ].T
# outlier_existing_df = existing_df.T[ outlier_countries_by_existing_index ].T
# outlier_existing_df = new_df.T[ outlier_countries_by_new_index ].T

# print(outlier_deaths_df)
# print(outlier_existing_df)
# print(outlier_existing_df)

# greater than 5 times the median value
# deaths_super_outlier = deaths_by_country_mean_summary['50%'] * 5
# existing_super_outlier = existing_by_country_mean_summary['50%'] * 5
# new_super_outlier = new_by_country_mean_summary['50%'] * 5
#
# super_outlier_countries_by_deaths_index = \
#     deaths_by_country_mean > deaths_super_outlier
# super_outlier_countries_by_existing_index = \
#     existing_by_country_mean > existing_super_outlier
# super_outlier_countries_by_new_index = \
#     new_by_country_mean > new_super_outlier

# print(super_outlier_countries_by_new_index)
# print(super_outlier_countries_by_existing_index)
# print(super_outlier_countries_by_deaths_index)

# print(sum(super_outlier_countries_by_deaths_index) / num_countries)

# super_outlier_deaths_df = \
#     deaths_df.T[ super_outlier_countries_by_deaths_index ].T
# super_outlier_existing_df = \
#     existing_df.T[ super_outlier_countries_by_existing_index ].T
# super_outlier_new_df = \
#     new_df.T[ super_outlier_countries_by_new_index ].T

# print(super_outlier_new_df)

# super_outlier_new_df.plot(figsize=(12,4)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
# pylab.show()

#　We have 22 countries where the number of new cases on an average year is greater
# than 5 times the median value of the distribution. Let's create a country that
# represents on average these 22
# average_super_outlier_country = super_outlier_new_df.mean(axis=1)
# print(average_super_outlier_country)

# let's create a country that represents the rest of the world.
# average_better_world_country = \
#     new_df.T[ - super_outlier_countries_by_new_index ].T.mean(axis=1)
# print(average_better_world_country)

# two_world_df = \
#     pd.DataFrame({
#             'Average Better World Country': average_better_world_country,
#             'Average Outlier Country' : average_super_outlier_country},
#         index = new_df.index)
# two_world_df.plot(title="Estimated new TB cases per 100K",figsize=(12,8))
# pylab.show()

# two_world_df.pct_change().plot(title="Percentage change in estimated new TB cases", figsize=(12,8))
# pylab.show()

# For example, TB and HIV are frequently associated, together with poverty levels.
# It would be interesting to join datasets and explore tendencies in each of them.
# We challenge the reader to give them a try and share with us their findings.

# https://github.com/jadianes/data-science-your-way/blob/master/02-exploratory-data-analysis/README.md
# end

# https://github.com/jadianes/data-science-your-way/blob/master/03-dimensionality-reduction-and-clustering/README.md
# start

# PCA

# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2)
# pca.fit(existing_df)
# existing_2d = pca.transform(existing_df)
# existing_df_2d = pd.DataFrame(existing_2d)
# existing_df_2d.index = existing_df.index
# existing_df_2d.columns = ['PC1','PC2']
# print(existing_df_2d.head())
# print(pca.explained_variance_ratio_)

# ax = existing_df_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8))
#
# for i, country in enumerate(existing_df.index):
#     ax.annotate(
#         country,
#         (existing_df_2d.iloc[i].PC2, existing_df_2d.iloc[i].PC1)
#     )
# pylab.show()

# from sklearn.preprocessing import normalize
#
# existing_df_2d['country_mean'] = pd.Series(existing_df.mean(axis=1), index=existing_df_2d.index)
# country_mean_max = existing_df_2d['country_mean'].max()
# country_mean_min = existing_df_2d['country_mean'].min()
# country_mean_scaled = \
#     (existing_df_2d.country_mean - country_mean_min) / country_mean_max
# existing_df_2d['country_mean_scaled'] = pd.Series(
#         country_mean_scaled,
#         index=existing_df_2d.index)
# print(existing_df_2d.head())
# existing_df_2d.plot(
#     kind='scatter',
#     x='PC2',
#     y='PC1',
#     s=existing_df_2d['country_mean_scaled']*100,
#     figsize=(16,8))
# pylab.show()

# existing_df_2d['country_sum'] = pd.Series(
#     existing_df.sum(axis=1),
#     index=existing_df_2d.index)
# country_sum_max = existing_df_2d['country_sum'].max()
# country_sum_min = existing_df_2d['country_sum'].min()
# country_sum_scaled = \
#     (existing_df_2d.country_sum-country_sum_min) / country_sum_max
# existing_df_2d['country_sum_scaled'] = pd.Series(
#         country_sum_scaled,
#         index=existing_df_2d.index)
# existing_df_2d.plot(
#     kind='scatter',
#     x='PC2', y='PC1',
#     s=existing_df_2d['country_sum_scaled']*100,
#     figsize=(16,8))
# pylab.show()

# existing_df_2d['country_change'] = pd.Series(
#     existing_df['2007'] - existing_df['1990'],
#     index=existing_df_2d.index)
# country_change_max = existing_df_2d['country_change'].max()
# country_change_min = existing_df_2d['country_change'].min()
# country_change_scaled = \
#     (existing_df_2d.country_change - country_change_min) / country_change_max
# existing_df_2d['country_change_scaled'] = pd.Series(
#         country_change_scaled,
#         index=existing_df_2d.index)
# print(existing_df_2d[['country_change','country_change_scaled']].head()) # KeyError: '2007'

# existing_df_2d.plot(
#     kind='scatter',
#     x='PC2', y='PC1',
#     s=existing_df_2d['country_change_scaled']*100,
#     figsize=(16,8))
# pylab.show()

# k-clustering

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit(existing_df)

existing_df_2d['cluster'] = pd.Series(clusters.labels_, index=existing_df_2d.index)

import numpy as np

# existing_df_2d.plot(
#         kind='scatter',
#         x='PC2',y='PC1',
#         c=existing_df_2d.cluster.astype(np.float),
#         figsize=(16,8))
# pylab.show()
