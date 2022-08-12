import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import base64
import streamlit as st
import sys
import holoviews as hv
from holoviews import opts, dim
sys.path.insert(0, '/spotify_data_project/')
st.set_option('deprecation.showPyplotGlobalUse', False)

def get_access_token(id, secret):
  url = 'https://accounts.spotify.com/api/token'
  encoded = base64.b64encode((id + ':' + secret).encode('ascii'))
  form = {'grant_type':'client_credentials'}
  headers = {
    'Authorization': 'Basic ' + encoded.decode('ascii')
  }
  response = requests.post(url, headers=headers, data=form)
  return response.json()['access_token']

@st.cache
def make_playlist_track_df(df):
  playlists_exploded = df.explode('tracks').reset_index()
  tracks = pd.DataFrame(list(playlists_exploded['tracks']))
  return pd.concat([playlists_exploded, tracks], axis=1).drop(['tracks'], axis = 1)

def get_duration_lists(artist_selectbox,track_df,df):
  #playlists_indexes_with_artist = playlist_track_df.loc[(playlist_track_df['index'].isin(playlist_track_df.loc[playlist_track_df['artist_name'] == artist_selectbox]['index'].unique()))]['index'].#unique()
  playlists_indexes_with_artist = track_df.loc[track_df['artist_name'] == artist_selectbox]['index'].unique()
  artist_duration_list = df.loc[df.index.isin(playlists_indexes_with_artist)]['duration_ms'].to_numpy()
  not_artist_duration_list = df.loc[df.index.isin(playlists_indexes_with_artist) == False]['duration_ms'].to_numpy()
  artist_duration_list = artist_duration_list / 60000
  not_artist_duration_list = not_artist_duration_list / 60000
  return (artist_duration_list, not_artist_duration_list)

# Stores TF IDF Values for each artist that shares a playlist with searched artist into a dataframe
def get_tf_idf_df(artist_selectbox, track_df, df):

  # Make matrix (Artists X Playlist) with the amount of times the artist appeared in the playlist
  tracks_in_playlist_with_artist = track_df.loc[(track_df['index'].isin(track_df.loc[track_df['artist_name'] == artist_selectbox]['index'].unique())) & (track_df['artist_name'] != artist_selectbox)]
  artist_frequency_df = tracks_in_playlist_with_artist.groupby(['index'])['artist_name'].value_counts().clip(upper=5)
  artist_frequency_df = artist_frequency_df.unstack()
  artist_frequency_df[artist_frequency_df.isna()] = 0
  artist_frequency_df = artist_frequency_df.T
  artist_frequency = artist_frequency_df.values

  artist_list = artist_frequency_df.index
  playlist_amount = len(df)

  # Finds amount of playlists each artist appears in, and uses that to calculate IDF
  inverse_playlist_frequency = track_df.loc[:,['index','artist_name']].drop_duplicates()['artist_name'].value_counts().sort_index(ascending=True)
  inverse_playlist_frequency = np.log((1 + playlist_amount) / (1 + inverse_playlist_frequency)) + 1
  inverse_playlist_frequency = inverse_playlist_frequency.loc[inverse_playlist_frequency.index.isin(artist_list)]

  tf_idf = artist_frequency * inverse_playlist_frequency.to_numpy()[:,None]
  tf_idf_sum = tf_idf.sum(axis=1)
  return pd.DataFrame({'artist':artist_list,'value':tf_idf_sum}).sort_values(by='value',ascending=False)

def get_similar_artist_list(artist, track_df):
  playlists_with_artist_list = track_df.loc[track_df['artist_name'] == artist]['index'].unique()
  return track_df.loc[track_df['index'].isin(playlists_with_artist_list)].drop_duplicates(subset=['index','artist_name'])['artist_name'].value_counts().drop([artist],axis=0).to_frame()

def get_chord_data(artist_selectbox, track_df,df):
  similar_artist_list = get_tf_idf_df(artist_selectbox,track_df,df)[0:5]['artist'].to_numpy()
  tracks_in_playlist_with_artist_by_similar_artists = track_df.loc[(track_df['index'].isin(track_df.loc[track_df['artist_name'] == artist_selectbox]['index'].unique())) & (track_df['artist_name'] != artist_selectbox) & (track_df['artist_name'].isin(similar_artist_list))][['index','artist_name']].drop_duplicates()
  artist_matrix = pd.DataFrame(0, index=similar_artist_list, columns=similar_artist_list)
  for artist in similar_artist_list:
    artists_playlists_list = tracks_in_playlist_with_artist_by_similar_artists.loc[tracks_in_playlist_with_artist_by_similar_artists['artist_name'] == artist]['index'].to_numpy()
    for index, row in tracks_in_playlist_with_artist_by_similar_artists.iterrows():
      if row['index'] in (artists_playlists_list):
        artist_matrix[artist][row['artist_name']] += 1
        if artist == row['artist_name']:
          artist_matrix[artist][row['artist_name']] = 0
  data = artist_matrix.reset_index().melt(id_vars="index").rename(columns={"index":"from"})
  nodes = hv.Dataset(pd.DataFrame({'artist':similar_artist_list}).reset_index())
  return (data, nodes)

def show_followers_hist(df):
  fig,ax = plt.subplots()
  ax.set_xlabel('Number of Followers')
  ax.set_ylabel('Frequency')
  ax.set_title('Histogram of Playlist Follower Counts')
  ax.hist(df['num_followers'].values,20,range=(0,20))
  return fig

def show_playlist_length_hist(df):
  fig,ax = plt.subplots()
  ax.set_xlabel('Number of Songs')
  ax.set_ylabel('Frequency')
  ax.set_title('Histogram of Playlist Length')
  ax.hist(df['num_tracks'].values, range=(5,250))
  return fig

def show_chord_plot(data):
  hv.extension('bokeh')
  hv.output(size=200)
  chord = hv.Chord((data[0], data[1])).select(value=(5, None))
  chord.opts(opts.Chord(cmap='Category20', edge_cmap='Category20', labels='artist', edge_color=dim('from').str(), node_color=dim('artist').str()))
  return chord

def show_violin_plot(data, artist):
  fig,ax = plt.subplots()
  ax.set_title('Playlist Duration')
  ax.set_ylabel('Duration (minutes)')
  ax.set_xticks(np.arange(1, 3), labels=[f"Playlists with {artist}", f"Playlists without {artist}"])
  ax.violinplot([data[0], data[1]])
  return fig

# Display Horizontal Bar Chart for Similar Artists Data
def show_simple_barchart(data):
  fig,ax = plt.subplots()
  indexes = data.index
  values = data['artist_name']
  y_pos = np.arange(len(indexes))
  bars = ax.barh(y_pos, values)
  ax.set_yticks(y_pos, labels=indexes)
  ax.set_xlabel('Frequency')
  ax.set_title('Similar Artists')
  ax.bar_label(bars)
  return fig

# Display Horizontal Bar Chart for TF IDF Data
def show_tf_idf_barchart(data):
  fig,ax = plt.subplots()
  indexes = data['artist']
  values = data['value']
  y_pos = np.arange(len(indexes))
  bars = ax.barh(y_pos, values)
  ax.set_yticks(y_pos, labels=indexes)
  ax.set_xlabel('TF IDF Frequency')
  ax.set_title('Similar Artists (TF IDF)')
  ax.bar_label(bars)
  return fig