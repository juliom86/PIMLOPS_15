from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()


@app.get('/PlayTimeGenre/')
def PlayTimeGenre(genre: str):
    end1 = pd.read_csv('endpoint1.csv')
    # Verifica si el genero ingresado existe en el DataFrame
    if genre not in end1.columns:
        return "Invalid genre"

    # Filtra el DataFrame por el genero ingresado
    genre_df = end1[end1[genre] == 1]

    # Ordera el DataFrame por playtime_forever
    sorted_genre_df = genre_df.sort_values(by='playtime_forever',
                                           ascending=False)

    # Extrae los primeros 5 usuarios
    top_5 = sorted_genre_df.head(5)

    # Crea un diccionario con los usuarios y sus datos
    year_genre_dict = {}
    for _, row in top_5.iterrows():
        year_genre_dict[genre] = {
            'year': row['year'],
            'playtime_forever': row['playtime_forever']
        }
    return year_genre_dict

# 2


@app.get('/UserForGenre/')
def UserForGenre(genre):
    end2 = pd.read_csv('endpoint2.csv')
    # Verifica si el genero ingresado existe en el DataFrame
    if genre not in end2.columns:
        return "Invalid genre"

    # Filtra el DataFrame por el genero ingresado
    genre_df2 = end2[end2[genre] == 1]

    # Ordera el DataFrame por playtime_forever
    sorted_genre_df2 = genre_df2.sort_values(by='playtime_forever',
                                             ascending=False)

    # Extrae los primeros 5 usuarios
    top_5 = sorted_genre_df2.head(5)

    # Crea un diccionario con los usuarios y sus datos
    top_users_dict = {}
    for _, row in top_5.iterrows():
        top_users_dict[row['user_id']] = {
            'year': row['year'],
            'playtime_forever': row['playtime_forever']
        }
    return top_users_dict


# 3


@app.get('/UsersRecommend/')
def UsersRecommend(year):
    end3 = pd.read_csv('endpoint3.csv')
    e3 = end3[(end3.recommend == 1) & (end3.positivo == 1) &
              (end3.neutral == 1)]
    e3y = e3[e3['year'] == year]
    e3yg = e3y.groupby('item_name')['recommend'].sum().sort_values(
        ascending=False).head()
    return pd.DataFrame(e3yg).to_dict()


# 4


@app.get('/UsersNotRecommend/')
def UsersNotRecommend(year):
    end4 = pd.read_csv('endpoint4.csv')
    e4 = end4[(end4.recommend == 0) & (end4.negativo == 1)]
    e4y = e4[e4['year'] == year]
    e4yg = e4y.groupby('item_name')['recommend'].count().sort_values(
        ascending=False).head(3)
    return pd.DataFrame(e4yg).to_dict()


#5

@app.get('/sentiment_analysis/')
def sentiment_analysis(year: str):
    df = pd.read_csv('endpoint5.csv')
    fil = df[df.year == year]
    return {
        'Año': year,
        'Positivo': fil.positivo.to_list()[0],
        'Negativo': fil.negativo.to_list()[0],
        'Neutral': fil.neutral.to_list()[0]
    }


# ML
@app.get('/recomendacion/')
def recomendacion(title: str) -> list:
    df = pd.read_csv('reco.csv')

    # Combinamos reviews por titulos

    df["review"] = df["review"].fillna("")
    grouped = df.groupby('item_name').agg(lambda x: ' '.join(x)).reset_index()

    # 2. Calcula matriz TF-IDF usando stop words en inglés
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(grouped['review'])

    # Calcula matriz de similaridad del coseno
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    idx = grouped.index[grouped['item_name'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    item_indices = [i[0] for i in sim_scores]
    return grouped['item_name'].iloc[item_indices].tolist()
