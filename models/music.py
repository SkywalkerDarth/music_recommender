import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from skimage import io
import requests
from PIL import Image
import subprocess

spotify_data = pd.read_csv('/home/skywalker/alx-project/models/FileStorage/SpotifyFeatures.csv')
print(spotify_data.head())

spotify_features_df = spotify_data.copy()

# OneHotEncoder to transform non-numerical data into binary
encoder_genre = OneHotEncoder(sparse=False)
encoder_key = OneHotEncoder(sparse=False)

# Fit and transform the 'genre' and 'key' columns
genre_encoded = encoder_genre.fit_transform(spotify_features_df[['genre']])
key_encoded = encoder_key.fit_transform(spotify_features_df[['key']])

#Normalization to put data in range 0-1#
scaled_features = MinMaxScaler().fit_transform([
  spotify_features_df['acousticness'].values,
  spotify_features_df['danceability'].values,
  spotify_features_df['duration_ms'].values,
  spotify_features_df['energy'].values,
  spotify_features_df['instrumentalness'].values,
  spotify_features_df['liveness'].values,
  spotify_features_df['loudness'].values,
  spotify_features_df['speechiness'].values,
  spotify_features_df['tempo'].values,
  spotify_features_df['valence'].values,
  ])

# assign values from a transposed array "scaled_features" to the specified columns in the "spotify_features_df"
spotify_features_df[['acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence']] = scaled_features.T

print(spotify_features_df.head())

#discarding the categorical and unnecessary features
spotify_features_df = spotify_features_df.drop('genre', axis = 1)
spotify_features_df = spotify_features_df.drop('track_name', axis = 1)
spotify_features_df = spotify_features_df.drop('artist_name', axis = 1)
spotify_features_df = spotify_features_df.drop('key', axis = 1)
spotify_features_df = spotify_features_df.drop('mode', axis = 1)
spotify_features_df = spotify_features_df.drop('time_signature', axis = 1)
spotify_features_df = spotify_features_df.drop('popularity', axis = 1)

print(spotify_features_df.head())

genre_encoded_df = pd.DataFrame(genre_encoded, columns=encoder_genre.categories_[0])
key_encoded_df = pd.DataFrame(key_encoded, columns=encoder_key.categories_[0])

spotify_features_df = pd.concat([spotify_features_df, genre_encoded_df, key_encoded_df], axis=1)

print(spotify_features_df.head())

#Fetching the playlist
#input your client_id and client_secret and have a playlist called 'my_playlist' on spotify
scope = 'user-library-read'
token = util.prompt_for_user_token(
    scope,
    client_id= "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", 
    client_secret="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", 
    redirect_uri='http://localhost:8881/callback'
  )
sp = spotipy.Spotify(auth=token)
playlist_dic = {}
playlist_cover_art = {}

for i in sp.current_user_playlists()['items']:
    playlist_dic[i['name']] = i['uri'].split(':')[2]
    playlist_cover_art[i['uri'].split(':')[2]] = i['images'][0]['url']

print(playlist_dic)

def generate_playlist_df(playlist_name, playlist_dic, spotify_data):

    playlist = pd.DataFrame()

    for i, j in enumerate(sp.playlist(playlist_dic[playlist_name])['tracks']['items']):
        playlist.loc[i, 'artist'] = j['track']['artists'][0]['name']
        playlist.loc[i, 'track_name'] = j['track']['name']
        playlist.loc[i, 'track_id'] = j['track']['id']
        playlist.loc[i, 'url'] = j['track']['album']['images'][1]['url']
        playlist.loc[i, 'date_added'] = j['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])

    playlist = playlist[playlist['track_id'].isin(spotify_data['track_id'].values)].sort_values('date_added',ascending = False)

    return playlist
playlist_df = generate_playlist_df('my_playlist', playlist_dic, spotify_data)

print(playlist_df.head())


def visualize_cover_art(playlist_df):
    temp = playlist_df['url'].values

    for i, url in enumerate(temp):
        command = ['img2txt', '-W', '100', '-H', '40', url]
        ascii_art = subprocess.run(command, capture_output=True, text=True).stdout
        print(ascii_art)
        print('Track:', playlist_df['track_name'].values[i])
        print('-' * 50)


def convert_to_ascii(image, width=100):
    image = image.convert('L')  # Convert image to grayscale
    aspect_ratio = image.width / image.height
    height = int(width / aspect_ratio / 2)
    resized_image = image.resize((width, height))
    pixels = resized_image.getdata()
    ascii_chars = ''.join(['@' if pixel < 128 else ' ' for pixel in pixels])
    ascii_art = '\n'.join(ascii_chars[i:i+width] for i in range(0, len(ascii_chars), width))
    return ascii_art

visualize_cover_art(playlist_df)

X = playlist_df
Y = spotify_features_df

print(X)
print(Y)

def generate_playlist_vector(spotify_features, playlist_df, weight_factor):
    
    spotify_features_playlist = spotify_features[spotify_features['track_id'].isin(playlist_df['track_id'].values)]
    spotify_features_playlist = spotify_features_playlist.merge(playlist_df[['track_id','date_added']], on = 'track_id', how = 'inner')
    
    spotify_features_nonplaylist = spotify_features[~spotify_features['track_id'].isin(playlist_df['track_id'].values)]
    
    playlist_feature_set = spotify_features_playlist.sort_values('date_added',ascending=False)

    print(playlist_feature_set)

    most_recent_date = playlist_feature_set.iloc[0,-1]
    
    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix,'days_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days)
        
    
    playlist_feature_set['weight'] = playlist_feature_set['days_from_recent'].apply(lambda x: weight_factor ** (-x))
    
    playlist_feature_set_weighted = playlist_feature_set.copy()
    
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-3].mul(playlist_feature_set_weighted.weight.astype(int),0))   
    
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-3]

    return playlist_feature_set_weighted_final.sum(axis = 0), spotify_features_nonplaylist

playlist_vector, nonplaylist_df = generate_playlist_vector(spotify_features_df, playlist_df, 1.2)
print(playlist_vector.shape)
print(nonplaylist_df.head())


def generate_recommendation(spotify_data, playlist_vector, nonplaylist_df):

    non_playlist = spotify_data[spotify_data['track_id'].isin(nonplaylist_df['track_id'].values)]
    non_playlist['sim'] = cosine_similarity(nonplaylist_df.drop(['track_id'], axis = 1).values, playlist_vector.drop(labels = 'track_id').values.reshape(1, -1))[:,0]
    non_playlist_top10 = non_playlist.sort_values('sim',ascending = False).head(10)
    non_playlist_top10['url'] = non_playlist_top10['track_id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
    
    return  non_playlist_top10

top10 = generate_recommendation(spotify_data, playlist_vector, nonplaylist_df)
print(top10.head())

visualize_cover_art(top10)
