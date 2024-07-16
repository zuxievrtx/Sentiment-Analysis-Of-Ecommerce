import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import nltk
from nltk.corpus import stopwords
import chardet

# Download NLTK stopwords
nltk.download('stopwords')

def color_sentiment(val):
    color = 'green' if val == 'Positif' else 'red' if val == 'Negatif' else 'orange'
    return f'background-color: {color}; color: white'


# Load the data
@st.cache_data
def load_data():
    # Detect file encoding
    with open('Final_data_labeled.csv', 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    # Read CSV file with detected encoding
    data = pd.read_csv('Final_data_labeled.csv', encoding=encoding)
    return data

# Preprocess text for word cloud
def preprocess_text(text):
    # Convert to lowercase and split into words
    words = text.lower().split()
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Create the Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title('Sentiment Analysis Dashboard for Product Reviews')

    # Load the data
    df = load_data()

    # Sidebar
    st.sidebar.header('Dashboard Controls')
    sentiment_filter = st.sidebar.multiselect('Filter by Sentiment', options=df['Sentimen'].unique(), default=df['Sentimen'].unique())

    # Filter data based on selection
    filtered_df = df[df['Sentimen'].isin(sentiment_filter)]

    # Main layout
    col1, col2 = st.columns(2)

    with col1:
        # Sentiment Distribution (Bar Chart)
        st.subheader('Sentiment Distribution')
        sentiment_counts = filtered_df['Sentimen'].value_counts()
        fig = px.bar(x=sentiment_counts.index, y=sentiment_counts.values, 
                     labels={'x': 'Sentiment', 'y': 'Count'},
                     color=sentiment_counts.index)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sentiment Proportion (Pie Chart)
        st.subheader('Sentiment Proportion')
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                     title='Sentiment Distribution')
        st.plotly_chart(fig, use_container_width=True)

    # Word Cloud
    st.subheader('Review Word Cloud')
    text = ' '.join(filtered_df['Tweets'].apply(preprocess_text))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Top Positive and Negative Reviews
    col3, col4 = st.columns(2)

    with col3:
        st.subheader('Top 5 Positive Reviews')
        positive_reviews = df[df['Sentimen'] == 'Positif'].sort_values('Tweets', key=lambda x: x.str.len(), ascending=False).head()
        for idx, row in positive_reviews.iterrows():
            st.write(f"• {row['Tweets']}")

    with col4:
        st.subheader('Top 5 Negative Reviews')
        negative_reviews = df[df['Sentimen'] == 'Negatif'].sort_values('Tweets', key=lambda x: x.str.len(), ascending=False).head()
        for idx, row in negative_reviews.iterrows():
            st.write(f"• {row['Tweets']}")

    # Review Length Analysis
    st.subheader('Review Length Analysis')
    df['review_length'] = df['Tweets'].str.len()
    fig = px.box(df, x='Sentimen', y='review_length', color='Sentimen',
                 labels={'review_length': 'Review Length', 'Sentimen': 'Sentiment'},
                 title='Distribution of Review Lengths by Sentiment')
    st.plotly_chart(fig, use_container_width=True)

     # Data Table
    st.subheader('Raw Data')
    
    # Apply styling to the dataframe
    styled_df = filtered_df.style.applymap(color_sentiment, subset=['Sentimen'])
    
    # Display the styled dataframe
    st.dataframe(styled_df)

if __name__ == '__main__':
    main()