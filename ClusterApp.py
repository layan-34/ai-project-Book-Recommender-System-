import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.cluster import KMeans
import random
from sklearn.metrics import confusion_matrix

# Constants
THRESHOLD_USER_RATINGS = 200  # Minimum number of ratings a user should have
THRESHOLD_BOOK_RATINGS = 50   # Minimum number of ratings a book should have
DEFAULT_RATING_FILL = 2       # Default value for filling missing ratings
NUM_CLUSTERS = 10             # Number of clusters for KMeans

# Load datasets
try:
    ratings = pd.read_csv(
        r"BX-Book-Ratings.csv",
        delimiter=';', encoding='latin-1', on_bad_lines='skip'
    )
    books = pd.read_csv(
        r"BX-Books.csv",
        delimiter=';', encoding='latin-1', on_bad_lines='skip', 
        dtype={
            'ISBN': str,
            'Book-Title': str,
            'Book-Author': str,
            'Year-Of-Publication': str,  # Treat as string to avoid numeric parsing issues
            'Publisher': str
        },
        low_memory=False  # Prevents DtypeWarning by processing data in chunks
    )
    users = pd.read_csv(
        r"BX-Users.csv",
        delimiter=';', encoding='latin-1', on_bad_lines='skip'
    )
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Preprocess datasets
ratings = ratings[['User-ID', 'ISBN', 'Book-Rating']]
books = books[['ISBN', 'Book-Title']]
users = users[['User-ID']]

# Filter users who rated more than the threshold
user_rating_counts = ratings['User-ID'].value_counts()
users_to_keep = user_rating_counts[user_rating_counts > THRESHOLD_USER_RATINGS].index
ratings = ratings[ratings['User-ID'].isin(users_to_keep)]

# Merge datasets and filter books with sufficient ratings
ratings_books = ratings.merge(books, on='ISBN', how='left')
number_rating = ratings_books.groupby('Book-Title')['Book-Rating'].count().reset_index()
number_rating.rename(columns={'Book-Rating': 'num_of_rating'}, inplace=True)
final_rating = ratings_books.merge(number_rating, on='Book-Title')
final_rating = final_rating[final_rating['num_of_rating'] >= THRESHOLD_BOOK_RATINGS]
ratings_books = final_rating.drop_duplicates(subset=['User-ID', 'Book-Title'])

# Create a user-item interaction matrix
user_item_matrix = ratings_books.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')
user_item_matrix.fillna(DEFAULT_RATING_FILL, inplace=True)

# Split the data
train_data, test_data = train_test_split(user_item_matrix, test_size=0.3, random_state=22)

# KMeans clustering
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)
kmeans.fit(train_data)

train_labels = kmeans.labels_
test_data_filled = test_data.reindex(columns=train_data.columns, fill_value=DEFAULT_RATING_FILL)
test_labels = kmeans.predict(test_data_filled)

# Evaluate predictions
def predict_ratings(train_data, train_labels, test_data, test_labels):
    predictions = np.zeros_like(test_data.values)
    unique_train_labels = np.unique(train_labels)
    for i, cluster_label in enumerate(test_labels):
        if cluster_label not in unique_train_labels:
            continue
        cluster_users = np.where(train_labels == cluster_label)[0]
        for idx in cluster_users:
            if idx < len(train_data):  # Ensure index is within bounds
                train_user_ratings = train_data.iloc[idx]
                predictions[i, :] += (train_user_ratings.values > 0).astype(int)
        predictions[i, :] = (predictions[i, :] > 0).astype(int)
    return predictions


predictions = predict_ratings(train_data, train_labels, test_data, test_labels)

# Convert actual ratings to binary format
actual = (test_data.values > 0).astype(int)

# Compute evaluation metrics
precision = precision_score(actual.flatten(), predictions.flatten(), zero_division=1)
recall = recall_score(actual.flatten(), predictions.flatten(), zero_division=1)
f1 = f1_score(actual.flatten(), predictions.flatten(), zero_division=1)
accuracy = accuracy_score(actual.flatten(), predictions.flatten())
cm = confusion_matrix(actual.flatten(), predictions.flatten())
print(cm)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")


from sklearn.metrics.pairwise import cosine_similarity

def recommend_books_with_similarity(user_id, k=5):
    if user_id not in user_item_matrix.index:
        return "User ID not found."

    user_idx = train_data.index.get_loc(user_id) if user_id in train_data.index else test_data.index.get_loc(user_id)
    cluster_label = train_labels[user_idx] if user_id in train_data.index else test_labels[user_idx]
    same_cluster_users = train_data.index[np.where(train_labels == cluster_label)[0]]

    # Compute similarity scores
    user_vector = train_data.loc[user_id].values.reshape(1, -1)
    cluster_vectors = train_data.loc[same_cluster_users].values
    similarities = cosine_similarity(user_vector, cluster_vectors).flatten()

    # Collect recommendations, weighted by similarity
    recommended_books = {}
    for i, neighbor_id in enumerate(same_cluster_users):
        if neighbor_id != user_id:
            neighbor_books = train_data.loc[neighbor_id][train_data.loc[neighbor_id] > 0].index
            for book in neighbor_books:
                if book not in recommended_books:
                    recommended_books[book] = 0
                recommended_books[book] += similarities[i]  # Weight by similarity

    # Sort books by weighted score and return top k
    ranked_books = sorted(recommended_books.keys(), key=lambda x: -recommended_books[x])
    return ranked_books[:k]


def recommend_books_diverse(user_id, k=5):
    if user_id not in user_item_matrix.index:
        return "User ID not found."

    user_idx = train_data.index.get_loc(user_id) if user_id in train_data.index else test_data.index.get_loc(user_id)
    cluster_label = train_labels[user_idx] if user_id in train_data.index else test_labels[user_idx]
    same_cluster_users = train_data.index[np.where(train_labels == cluster_label)[0]]

    # Collect all potential recommendations
    all_books = set()
    for neighbor_id in same_cluster_users:
        if neighbor_id != user_id:
            neighbor_books = train_data.loc[neighbor_id][train_data.loc[neighbor_id] > 0].index
            all_books.update(neighbor_books)

    # Exclude books the user has already interacted with
    user_books = set(train_data.loc[user_id][train_data.loc[user_id] > 0].index)
    recommendations = list(all_books - user_books)

    # Shuffle and select top k
    random.seed(user_id)
    random.shuffle(recommendations)
    return recommendations[:k]


# # Improved Book Recommendation Function
# def recommend_books(user_id, k=5):
#     if user_id not in user_item_matrix.index:
#         return "User ID not found."

#     user_idx = train_data.index.get_loc(user_id) if user_id in train_data.index else test_data.index.get_loc(user_id)
#     cluster_label = train_labels[user_idx] if user_id in train_data.index else test_labels[user_idx]
#     same_cluster_users = train_data.index[np.where(train_labels == cluster_label)[0]]

#     recommended_books = set()
#     for neighbor_id in same_cluster_users:
#         if neighbor_id != user_id:
#             print(neighbor_id)
#             neighbor_books = train_data.loc[neighbor_id][train_data.loc[neighbor_id] > 0].index
#             recommended_books.update(neighbor_books)
#         if len(recommended_books) >= k:
#             break
#     return list(recommended_books)[:k]

# Interactive recommendation loop
while True:
    user_input = input("Enter User ID (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    try:
        user_id = int(user_input)
        # recommendations = recommend_books(user_id, k=5)
        recommendations = recommend_books_diverse(user_id)
        print(f"Recommended Books: {recommendations}")
    except ValueError:
        print("Invalid input. Please enter a numeric User ID.")
