import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.utils import resample




# Load datasets
books = pd.read_csv('BX-Books.csv', sep=';', on_bad_lines='skip', encoding="latin-1", low_memory=False)
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('BX-Users.csv', sep=';', on_bad_lines='skip', encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', on_bad_lines='skip', encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

# Filter users who have rated at least 200 books
counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 100].index)]

# Filter books that have been rated at least 100 times
counts = ratings['ISBN'].value_counts()
ratings = ratings[ratings['ISBN'].isin(counts[counts >= 50].index)]
print(f"Total ratings after filtering users: {len(ratings)}")

# Assign features and target variable
X = ratings.drop(columns='bookRating')  # Features without the bookRating column
y = ratings['bookRating']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
print(f"Training target size: {len(y_train)}, Testing target size: {len(y_test)}")

# Drop a specified number of entries with ratings = 0 from the training set
X_train_non_zero = X_train[y_train != 0]
y_train_non_zero = y_train[y_train != 0]
X_train_zero = X_train[y_train == 0].sample(n=5000, random_state=42)
y_train_zero = y_train[y_train == 0].sample(n=5000, random_state=42)
X_train_filtered = pd.concat([X_train_non_zero, X_train_zero])
y_train_filtered = pd.concat([y_train_non_zero, y_train_zero])

X_test_non_zero = X_test[y_test != 0]
y_test_non_zero = y_test[y_test != 0]
X_test_zero = X_test[y_test == 0].sample(n=1000, random_state=42)
y_test_zero = y_test[y_test == 0].sample(n=1000, random_state=42)
X_test_filtered = pd.concat([X_test_non_zero, X_test_zero])
y_test_filtered = pd.concat([y_test_non_zero, y_test_zero])




print(f"Filtered training set size: {len(X_train_filtered)}")
print(f"Filtered testing set size: {len(X_test_filtered)}")
# Merge X_train_filtered with y_train_filtered to create the pivot table
train_pivot = pd.concat([X_train_filtered, y_train_filtered], axis=1).pivot_table(index='userID', columns='ISBN', values='bookRating', aggfunc='mean').fillna(0)
train_matrix = csr_matrix(train_pivot.values)

# Dimensionality reduction
svd = TruncatedSVD(n_components=20, random_state=42)
matrix_svd = svd.fit_transform(train_matrix)

# Train the KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model_knn.fit(train_matrix)

def get_recommendations(user_id, train_pivot, model_knn, X_test, ratings):
    if user_id not in train_pivot.index:
        # Use popularity-based recommendations for users not in the training set
        top_books = ratings.groupby('ISBN')['bookRating'].mean().sort_values(ascending=False).head(5)
        recommendations = books[books['ISBN'].isin(top_books.index)]['bookTitle'].tolist()
        return {"error": "User not found in training data", "recommendations": recommendations}

    # Get the ratings provided by the user
    user_ratings = train_pivot.loc[user_id].values.reshape(1, -1)

    # Find the k-nearest neighbors for the user
    distances, indices = model_knn.kneighbors(user_ratings, n_neighbors=7)

    # Get the ratings of the neighboring users
    neighbor_indices = indices.flatten()[1:]  # Exclude the first element as it is the user itself
    neighbor_distances = distances.flatten()[1:]

    similar_users = train_pivot.iloc[neighbor_indices]
    similar_users = similar_users.assign(distance=neighbor_distances)

    # Aggregate the ratings of similar users
    similar_users_ratings = similar_users.drop(columns='distance').mean(axis=0).fillna(0)

    # Get the top N recommended books
    recommended_books_isbn = similar_users_ratings.sort_values(ascending=False).head(5).index.tolist()

    # Ensure recommendations overlap with X_test
    test_books = set(X_test['ISBN'].unique())
    recommended_books_isbn = [isbn for isbn in recommended_books_isbn if isbn in test_books]

    # Retrieve book titles
    recommended_books = books[books['ISBN'].isin(recommended_books_isbn)]['bookTitle'].tolist()

    # Fallback to top-rated books if no recommendations overlap
    if not recommended_books:
        top_books = ratings.groupby('ISBN')['bookRating'].mean().sort_values(ascending=False).head(5)
        recommended_books = books[books['ISBN'].isin(top_books.index)]['bookTitle'].tolist()
    print("Recommended Books:", recommended_books)
    return {"recommendations": recommended_books_isbn,"Recommended_Books_title": recommended_books}

def calculate_metrics(user_id, X_test, y_test, model_knn, train_pivot, ratings):
    # Get recommendations for the specific user
    user_recommendations = get_recommendations(user_id, train_pivot, model_knn, X_test_filtered, ratings)
    if "error" in user_recommendations:
        print(user_recommendations["error"])
        return user_recommendations

    # Create binary rating for recommendations: 1 for recommended, 0 otherwise
    recommended_books = user_recommendations['recommendations']
    Recommended_Books_title=user_recommendations['Recommended_Books_title']
    # Ground truth: Set 1 if the actual rating in y_test is >= 4, else 0


    y_true = y_test_filtered[X_test_filtered['userID'] == user_id].apply(lambda x: 1 if x > 7 else 0).tolist()

    y_pred = y_test_filtered[X_test_filtered['ISBN'].isin(recommended_books)].apply(lambda x: 1 if x >7 else 0).tolist()


    # Ensure y_pred and y_true are of the same length for the user
    if len(y_pred) < len(y_true):
        y_pred += [0] * (len(y_true) - len(y_pred))
    elif len(y_pred) > len(y_true):
        y_true += [0] * (len(y_pred) - len(y_true))

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display metrics
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Confusion Matrix:\n {cm}')

    if X_test[X_test['ISBN'].isin(recommended_books)].empty:
        print("No matching ISBNs found between recommended_books and X_test!")

    return {
        "Precision": precision,
        "recall": recall,
        "F1_Score": f1,
        "Accuracy": accuracy,
        "recommended_books": Recommended_Books_title
    }




'''
def print_top_users(test_data, ratings, top_n=10):
    # Ensure the 'bookRating' column exists in the ratings DataFrame
    merged_data = test_data.merge(ratings[['userID', 'ISBN', 'bookRating']], on=['userID', 'ISBN'], how='left')

    # Count the number of ratings per user in the test data
    user_rating_counts = merged_data['userID'].value_counts()

    # Count the number of non-zero ratings per user in the test data
    non_zero_ratings_counts = merged_data[merged_data['bookRating'] != 0]['userID'].value_counts()

    # Get the top N users with the most ratings
    top_users = user_rating_counts.head(top_n)

    print(f"Top {top_n} users with the most ratings in the test data:")
    for user_id in top_users.index:
        total_ratings = top_users[user_id]
        non_zero_ratings = non_zero_ratings_counts.get(user_id, 0)
        print(f"User {user_id}: Total Ratings = {total_ratings}, Non-Zero Ratings = {non_zero_ratings}")

    return top_users

# Example usage:
print_top_users(X_test_filtered , ratings, top_n=100)
'''