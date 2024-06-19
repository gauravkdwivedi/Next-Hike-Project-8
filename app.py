import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex

@st.cache_data
def load_data():
    df = pd.read_csv('all_upwork_jobs.csv')
    
    # Fill NaN values in numerical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # Fill NaN values in object columns
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].fillna('')
    
    df['combined_features'] = df['title']
    return df

@st.cache_resource
def create_annoy_index(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    num_features = tfidf_matrix.shape[1]
    annoy_index = AnnoyIndex(num_features, 'angular')
    
    for i in range(tfidf_matrix.shape[0]):
        annoy_index.add_item(i, tfidf_matrix[i].toarray()[0])
    
    annoy_index.build(10)
    return tfidf_matrix, annoy_index

def get_recommendations(df, tfidf_matrix, annoy_index, job_title, n_recommendations=10):
    try:
        idx = df[df['title'] == job_title].index[0]
    except IndexError:
        return "Job title not found in dataset."

    job_vector = tfidf_matrix[idx].toarray()[0]
    nearest_neighbors = annoy_index.get_nns_by_vector(job_vector, n_recommendations + 1)
    nearest_neighbors = [n for n in nearest_neighbors if n != idx]
    return df[['title', 'budget', 'hourly_low', 'hourly_high']].iloc[nearest_neighbors]

# Load data and create index
df = load_data()
tfidf_matrix, annoy_index = create_annoy_index(df)

# Streamlit UI
st.title('Job Recommendation System')

job_title = st.text_input('Enter a job title:')
num_recommendations = st.slider('Number of recommendations:', 1, 20, 10)

if st.button('Get Recommendations'):
    with st.spinner('Finding recommendations...'):
        recommendations = get_recommendations(df, tfidf_matrix, annoy_index, job_title, num_recommendations)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write(recommendations)