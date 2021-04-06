def main():

# Importing required libraries and reading the dataset. 

    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer
    from pickle import load,dump
    from os import system
    df = pd.read_csv('Datasets/movies_data.csv')
    __clean__ = system('cls')
    # Function to create a string of important features for each movie.

    def get_imp_features(data) :
        important_features = []
        for i in range(0,data.shape[0]): 
            important_features.append(data['Title'][i]+' '+data['Certificate'][i]+' '+data['Genre'][i]+' '+data['Director'][i]+' '+data['Star1'][i]+' '+data['Star2'][i])

        return important_features

    # Getting the title from the user.

    title = input('Enter the most recent movie you watched: ')

    # Function to find a similar title to the one user has entered.

    def alt_title(df,title):

        cm1 = CountVectorizer().fit_transform([title]+list(df['Title']))
        cs1 = cosine_similarity(cm1)
        scores = list(enumerate(cs1[0]))
        sorted_scores = sorted(scores,key = lambda x : x[1], reverse = True)
        sorted_scores = sorted_scores[1:]
        t = df['Title'][sorted_scores[0][0]-1]
        return t

    # Vectorizing all the important features and creating a cosine-similarity matrix.

    df['imp_features'] = get_imp_features(df)
    cm = CountVectorizer().fit_transform(list(df['imp_features']))
    cs = cosine_similarity(cm)

    # Findig the movie id of the title the user entered.

    f= 0
    try:
        mov_id = df[df.Title == title]['movie_id'].values[0]
    except:
        f=1
        title = alt_title(df,title)
        mov_id = df[df.Title == title]['movie_id'].values[0]

    # Creating a list, where each element is a tuple having the index and similarity score as its elements, and sorting it.

    scores = list(enumerate(cs[mov_id]))
    sorted_scores = sorted(scores,key = lambda x : x[1], reverse = True)
    if f == 0:
        sorted_scores = sorted_scores[1:]


    # Printing the the most recommended movies.

    m = 0
    print('The 7 most recommended movies to the user are: \n')
    for item in sorted_scores:
        movie_title = df[df.movie_id == item[0]]['Title'].values[0]
        print(m+1,movie_title)
        m += 1
        if m > 6:
            break
    if f == 1:
        print('\n Note: Due to insufficient data, these results might not be 100% accurate.')


# Execution

main()