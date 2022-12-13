import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
from datetime import datetime

df = pd.read_csv('IN_youtube_trending_data.csv')
df.head(3)
df.tail(3)
df.shape

df['trending_date'] = pd.to_datetime(df['trending_date'], format="%Y-%m-%dT%H:%M")
df['publishedAt'] = pd.to_datetime(df['publishedAt'], format="%Y-%m-%dT%H:%M")
df.head(3)
df.nunique()

for col in df.columns:
    a = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,a))

trends = df.drop(['video_id', 'categoryId', 'comments_disabled', 'ratings_disabled'], axis=1)

plot.scatter(x=df['view_count'], y=df['likes'])
plot.title('view_count vs likes')
plot.xlabel('view_count')
plot.ylabel('likes')
sns.regplot(x='view_count', y='likes', data=df, scatter_kws={"color":"#FD841F"}, line_kws={"color":"#E14D2A"})
plot.show()

correlation = df.corr(method='pearson')
fig = plot.figure(figsize=(10, 8))
sns.heatmap(correlation,xticklabels = correlation.columns, yticklabels = correlation.columns, annot = True, cmap='RdPu', annot_kws={'weight':'bold'})
plot.title('Correlation matrix for Numeric Features')
plot.xlabel('Movie Features')
plot.ylabel('Movie Features')
plot.show()

colors = ["#1C315E"]
sns.set_theme(style="darkgrid")
sns.set_palette(sns.color_palette(colors))
sns.lmplot(x = 'likes', y = 'view_count', data = trends)
plot.title('View Count vs. Likes')
plot.show()

sns.set(rc={'figure.figsize':(12,10)})
sns.lineplot(x='trending_date', y='view_count', data=df, errorbar=('ci', False), color='#E14D2A')
plot.title('View Count vs. Trending Date')
plot.show()

sns.set(rc={'figure.figsize':(8,5)})
sns.lineplot(x='publishedAt', y='likes', data=df, errorbar=('ci', False), color='#001253')
plot.title('Likes vs Published Date')
plot.show()

sns.set(rc={'figure.figsize':(15,10)})
by_channel = df.groupby("title").size().reset_index(name="view_count").sort_values("view_count", ascending=False).head(20)
sns.barplot(x="view_count", y="title", data=by_channel,palette=sns.cubehelix_palette(n_colors=22, reverse=True))
plot.title('Most Viewed Videos')
plot.xlabel("View")
plot.ylabel("Video Title")
plot.show()

df.loc[df['categoryId'] ==1, 'category'] = 'Film & Animation'
df.loc[df['categoryId'] ==2, 'category'] = 'Autos & Vehicles'
df.loc[df['categoryId'] ==10, 'category'] = 'Music'
df.loc[df['categoryId'] ==15, 'category'] = 'Pets & Animals'
df.loc[df['categoryId'] ==17, 'category'] = 'Sports'
df.loc[df['categoryId'] ==19, 'category'] = 'Travel & Events'
df.loc[df['categoryId'] ==20, 'category'] = 'Gaming'
df.loc[df['categoryId'] ==22, 'category'] = 'People & Blogs'
df.loc[df['categoryId'] ==23, 'category'] = 'Comedy'
df.loc[df['categoryId'] ==24, 'category'] = 'Entertainment'
df.loc[df['categoryId'] ==25, 'category'] = 'News & Politics'
df.loc[df['categoryId'] ==26, 'category'] = 'Howto & Style'
df.loc[df['categoryId'] ==27, 'category'] = 'Education'
df.loc[df['categoryId'] ==28, 'category'] = 'Science & Technology'
df.loc[df['categoryId'] ==29, 'category'] = 'Nonprofits & Activism'
df.head(5)

sns.set(rc={'figure.figsize':(10,10)})
sns.barplot(x='likes', y='category', data=df, errorbar=('ci', False), palette='magma')
plot.title('Likes vs. Category')
plot.xlabel("Likes")
plot.ylabel("Category")
plot.show()

sns.set(rc={'figure.figsize':(10,15)})
sns.barplot(x='view_count', y='category', data=df, palette='rocket', errorbar=('ci', False), hue='ratings_disabled')
plot.show()

sns.set(rc={'figure.figsize':(5,5)})
sns.barplot(x='comments_disabled', y='view_count', data=df, palette='magma', errorbar=('ci', False))
plot.title('View Count vs. Comments Disabled')
plot.show()

sns.set(rc={'figure.figsize':(10,6)})
df.likes.plot(kind='line',color='#93089A', label = 'Likes', linestyle = '--')
df.dislikes.plot(color='black', label = 'Dislikes')
plot.title('Likes vs. Dislikes')
plot.xlabel('Number of videos')
plot.ylabel('Likes and Dislike')
plot.show()