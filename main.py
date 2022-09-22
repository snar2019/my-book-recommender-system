import streamlit as st
import pandas as pd 
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.metrics.pairwise import cosine_similarity
import pickle

header = st.container()
dataset = st.container()
Popular_books = st.container()
Trained_model = st.container()



with header:
    st.title('Book Recommemdation system app')
    st.text('''     A book recommendation system is a type of recommendation system where we have to recommend 
    similar books to the reader based on his interest. The books recommendation system is used by online 
    websites which provide ebooks like google play books, open library, good Read's, etc.''')







with dataset:

    st.header('Dataset used for the project')
    st.text('''    Data source is from kaggle. We have three different tables named Books, Users, Ratings
    We have performed Data engineering on the tables and combined the data for EDA. An overvoew of the combined
    data is shown in the data table.''')

    st.text("Books data")
    books_data = pd.read_csv('Books.csv')
    st.write(books_data.head())
    

    st.text("Popular Authors")
    book_author = books_data['Book-Author'].value_counts().head(15)
    st.bar_chart(book_author)
    
    st.text("Ratings data")
    ratings_data = pd.read_csv('Ratings.csv')
    st.write(ratings_data.head())

    st.text("Users data")
    users_data = pd.read_csv('Users.csv')
    st.write(users_data.head())

with Popular_books:
    st.header('Top 50 books based on Popularity') 

    top_50_popular_books = pickle.load(open('top_50_popular_books.pkl','rb'))
    st.write(top_50_popular_books.head())


    

with Trained_model:
    st.header('Recommendation system Model') 
    st.subheader('Overview')
    st.text('''     User will select the name of the book from Books selection drop down menu,
    and the Recommender system will display top 5 books of the same genre.''')

    similarity_score = pickle.load(open('similarity_score.pkl','rb'))
    top_50_popular_books = pickle.load(open('top_50_popular_books.pkl','rb'))
    pivot_df = pickle.load(open('pivot_df.pkl','rb'))
    books_data_new = pickle.load(open('books_data_new','rb'))

    st.subheader("Books Selection")
    final_rating_data = pickle.load(open('final_rating_data.pkl','rb'))
    books_name = set(books_data_new['title'].values)
    Book_selected = st.selectbox('Select the book name',books_name)
    
    st.write(f'you entered the book: {Book_selected}') 
    
    
    


    def recommend(Book_selected):
        index = np.where(pivot_df.index == Book_selected)[0][0]
        similar_items = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:4]
        data = []
        for i in similar_items:
            item = []
            temp_df = books_data_new[books_data_new['title'] == pivot_df.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('title')['title'].values))
            #item.extend(list(temp_df.drop_duplicates('title')['author'].values))
            item.extend(list(temp_df.drop_duplicates('title')['Image-URL-M'].values))
        
            data.append(item)
    
        return(data)
        
##### recommender section
    
    
   
    
    
    
    st.subheader("Recommender system")


   
    # images = books_data_new[['title', 'Image-URL-M']]
    # images = images.groupby('title')['Image-URL-M'].sum()
    
    if st.button('Recommend'):
        recommended_book_names = recommend(Book_selected)
        col1,col2,col3 = st.columns(3)
        with col1:
            st.text(recommended_book_names[0][0])
            st.image(recommended_book_names[0][1])
        with col2:
            st.text(recommended_book_names[1][0])
            st.image(recommended_book_names[1][1])
        with col3:
            st.text(recommended_book_names[2][0])
            st.image(recommended_book_names[2][1])
        
        
        
        
     
