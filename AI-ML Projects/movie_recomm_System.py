import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the dataset
df = pd.read_csv("movie_dataset.csv")

# Step 2: Combine relevant features into a single string
def combine_features(row):
    return f"{row['genre']} {row['director']}"

df["combined"] = df.apply(combine_features, axis=1)

# Step 3: Convert text to vectors using CountVectorizer
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(df["combined"])

# Step 4: Compute cosine similarity
cosine_sim = cosine_similarity(count_matrix)

# Step 5: Recommendation function
def recommend(movie_title):
    if movie_title not in df['title'].values:
        print("Movie not found in the dataset.")
        return

    idx = df[df['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

    print(f"\nRecommended movies similar to '{movie_title}':")
    for rank, (movie_idx, _) in enumerate(sorted_scores, start=1):
        recommended_title = df.iloc[movie_idx]['title']
        print(f"{rank}. {recommended_title}")

# Step 6: Take movie name from user
user_input = input("Enter a movie name: ")
recommend(user_input)
