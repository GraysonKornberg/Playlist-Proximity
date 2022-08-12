from random import sample
import pandas as pd
import streamlit as st
import os
import json
import sys
from tqdm import tqdm
import holoviews as hv
import requests
sys.path.insert(0, '/spotify_data_project/')
st.set_option('deprecation.showPyplotGlobalUse', False)
import clientSecrets
from functions import *

# Loads 500,000 playlists from json (first 500,000 if first = true, else last 500,000)
@st.cache
def load_json(first):
  json_path = 'spotify_million_playlist_dataset/data' if first else 'spotify_million_playlist_dataset/data2'
  playlists_df = None
  for file in tqdm(os.listdir(json_path)):
    with open(os.path.join(json_path,file)) as f:
      if playlists_df is not None:
        playlists_df = pd.concat([playlists_df, pd.DataFrame(json.load(f)['playlists'])])
      else:
        playlists_df = pd.DataFrame(json.load(f)['playlists'])
  return playlists_df.reset_index().drop(['index'],axis=1)

access_token = get_access_token(clientSecrets.client_id, clientSecrets.client_secret)

playlists_df_full = load_json(True)
sample_df = playlists_df_full.sample(n=150000, random_state = 1)
playlist_track_df = make_playlist_track_df(sample_df)

artists_list = playlist_track_df['artist_name'].unique()
artist_selectbox = ''
st.title('General Sample Statistics:')
st.write(f"This sample contains {len(sample_df)} playlists")
st.pyplot(show_followers_hist(sample_df))
st.pyplot(show_playlist_length_hist(sample_df))

st.title('Statistics from Artist of Choice:')
artist_selectbox = st.selectbox('Artist',artists_list, key='selectbox_key')

if (st.button('Go',disabled=(artist_selectbox is None or len(artist_selectbox) == 0))):
  reponse = requests.get(url=f"https://api.spotify.com/v1/search?q={artist_selectbox}",headers={'Authorization': f"Bearer {access_token}"}, params={'type':'artist'})
  id_response = reponse.json()['artists']['items']
  artist_id = id_response[0]['id']
  response = requests.get(url=f"https://api.spotify.com/v1/artists/{artist_id}",headers={'Authorization': f"Bearer {access_token}"})
  artist_response = response.json()
  st.markdown(f"<div style=\"display:flex;flex-direction:row;justify-content:center;margin-bottom:20px\"><div style=\"align-self:center\"><h3 style=\"text-align:center\">{artist_selectbox} has {'{:,}'.format(artist_response['followers']['total'])} followers, and a popularity rating of {artist_response['popularity']}/100</h3><h3 style=\"text-align:center\">Genre: {', '.join(artist_response['genres'])}</h3></div><div><img width=320 src=\"{artist_response['images'][0]['url']}\"></div></div>",unsafe_allow_html=True)
  st.pyplot(show_simple_barchart(get_similar_artist_list(artist_selectbox, playlist_track_df).iloc[0:5]))
  st.write(f"This barchart ranks the most similar artists to {artist_selectbox}. The frequency is calculated by finding how many times another artist appears in the same playlist as {artist_selectbox}.")
  st.pyplot(show_tf_idf_barchart(get_tf_idf_df(artist_selectbox,playlist_track_df,sample_df).iloc[0:5]))
  st.write(f"This barchart also ranks the most similar artists to {artist_selectbox}, but uses a different method. TF IDF (term frequency-inverse document frequency) is a statistical measure that is typically used for finding word frequency in a group of documents. Words that appear very often, like 'the,' would rank lower because it is so common. In this case, artists are like words, and playlists are like the documents. An artist that appears in every playlist would be offset by how common it is. I decided to use this method because some artists, like Drake, seemed to appear in so many playlists that they would often be one of the most similar artists in the previous barchart. This barchart also takes into account how many songs the artists have in each playlist. An artist with 3 songs in a playlist with your selected artist would be ranked higher than if they only had 1 song. This is limited to 5 songs to limit outliers.")
  st.pyplot(show_violin_plot(get_duration_lists(artist_selectbox,playlist_track_df,sample_df),artist_selectbox))
  st.write(f"This violin plot shows the durations of playlists with {artist_selectbox} and the durations of playlists without {artist_selectbox}.")
  st.bokeh_chart(hv.render(show_chord_plot(get_chord_data(artist_selectbox,playlist_track_df,sample_df)), backend='bokeh'))
  st.write(f"This is a chord diagram that uses the most similar artists from the TF IDF barchart. The lines between each artist show how many playlists they appear in together.")