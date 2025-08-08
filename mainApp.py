import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

# Load datasets
ratings = pd.read_csv(
    r"BX-Book-Ratings.csv",
    delimiter=';',
    encoding='latin-1',
    on_bad_lines='skip'
)
books = pd.read_csv(
    r"BX-Books.csv",
    delimiter=';',
    encoding='latin-1',
    on_bad_lines='skip',
    dtype={'ISBN': str, 'Book-Title': str, 'Book-Author': str},
    low_memory=False  # Suppress dtype warning
)
users = pd.read_csv(
    r"BX-Users.csv",
    delimiter=';',
    encoding='latin-1',
    on_bad_lines='skip'
)

# Preprocess datasets
ratings = ratings[['User-ID', 'ISBN', 'Book-Rating']]
books = books[['ISBN', 'Book-Title']]
users = users[['User-ID']]

# Filter users who rated more than 300 books
user_rating_counts = ratings['User-ID'].value_counts()
users_to_keep = user_rating_counts[user_rating_counts > 300].index
ratings = ratings[ratings['User-ID'].isin(users_to_keep)]

# Merge datasets
ratings_books = ratings.merge(books, on='ISBN', how='left')
number_rating = ratings_books.groupby('Book-Title')['Book-Rating'].count().reset_index()
number_rating.rename(columns={'Book-Rating': 'num_of_rating'}, inplace=True)
final_rating = ratings_books.merge(number_rating, on='Book-Title')
final_rating = final_rating[final_rating['num_of_rating'] >= 50]
# print(final_rating)
ratings_books = final_rating.drop_duplicates(subset=['User-ID', 'Book-Title'])

# Create a user-item interaction matrix
user_item_matrix = ratings_books.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')
# print(user_item_matrix)
user_item_matrix.fillna(2, inplace=True)

# Split data into training and testing sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.3, random_state=42)

# Fit KNN model
k = 100
knn = NearestNeighbors(metric='euclidean', algorithm='auto', n_neighbors=k)
# knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k)  
# knn = NearestNeighbors(n_neighbors=k) 
knn.fit(train_data.values)



# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# # Step 2: Get the nearest neighbors for each point
# distances, indices = knn.kneighbors(train_data.values)

# # Step 3: Reduce dimensionality (e.g., using PCA) to 2D for visualization
# pca = PCA(n_components=2)
# train_data_2d = pca.fit_transform(train_data.values)

# # Step 4: Plot the reduced data and the nearest neighbors
# plt.figure(figsize=(8, 6))

# # Scatter plot of all data points
# plt.scatter(train_data_2d[:, 0], train_data_2d[:, 1], c='blue', label='Data Points', alpha=0.6)

# # Plot nearest neighbors
# for i in range(len(train_data_2d)):
#     for neighbor in indices[i, 1:]:  # Skip the first one because it's the point itself
#         plt.plot([train_data_2d[i, 0], train_data_2d[neighbor, 0]], 
#                  [train_data_2d[i, 1], train_data_2d[neighbor, 1]], 
#                  c='gray', linestyle='-', alpha=0.5)

# plt.title('KNN Clustering')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend()
# plt.show()





# Print dataset info
print(f"Number of users in train_data: {len(train_data.index)}")
print(f"Number of users in test_data: {len(test_data.index)}")
print(f"Number of overlapping users: {len(set(train_data.index) & set(test_data.index))}")

# Get predictions for test data
distances, indices = knn.kneighbors(test_data.values)

# Ensure predictions array matches the test data dimensions
predictions = np.zeros_like(test_data.values)
for i in range(len(test_data)):
    for neighbor_idx in indices[i]:
        if neighbor_idx < predictions.shape[1]:  # Ensure within bounds
            predictions[i, neighbor_idx] = 1

# Convert actual ratings to binary format
actual = (test_data.values > 0).astype(int)

# Calculate metrics for each user
metrics = {
    'precision': [],
    'recall': [],
    'f1': [],
    'accuracy': []
}

for i in range(len(test_data)):
    if np.sum(actual[i]) > 0:  # Skip users with no ratings
        metrics['precision'].append(precision_score(actual[i], predictions[i], zero_division=0))
        metrics['recall'].append(recall_score(actual[i], predictions[i], zero_division=0))
        metrics['f1'].append(f1_score(actual[i], predictions[i], zero_division=0))
        metrics['accuracy'].append(accuracy_score(actual[i], predictions[i]))

# Print average metrics
print(f"Average Precision: {np.mean(metrics['precision']):.4f}")
print(f"Average Recall: {np.mean(metrics['recall']):.4f}")
print(f"Average F1-Score: {np.mean(metrics['f1']):.4f}")
print(f"Average Accuracy: {np.mean(metrics['accuracy']):.4f}")

# Metrics summary
metrics_df = pd.DataFrame(metrics)
# print("\nMetrics Summary:")
# print(metrics_df.describe())

# Book recommendation function
def recommend_books(user_id, k=5):
    """Recommends books for a given user ID based on KNN clustering."""
    if user_id not in user_item_matrix.index:
        return "User ID not found."

    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    print(user_vector)
    distances, indices = knn.kneighbors(user_vector, n_neighbors=k+1)

    recommended_books = []
    for neighbor_idx in indices.flatten()[1:]:
        if neighbor_idx < len(user_item_matrix.index):
            neighbor_id = user_item_matrix.index[neighbor_idx]
            neighbor_books = user_item_matrix.loc[neighbor_id][user_item_matrix.loc[neighbor_id] > 0].index
            recommended_books.extend(neighbor_books)

    return list(set(recommended_books))[:k]

def recommend_books1(user_id, k=5):
    # Get the user's ratings from the user-item matrix
    user_ratings = user_item_matrix.loc[user_id].values.reshape(1, -1)
    
    # Find the nearest neighbors of this user
    distances, indices = knn.kneighbors(user_ratings, n_neighbors=k)
    
    # Collect all books rated by the nearest neighbors
    recommended_books_set = set()
    
    for neighbor_idx in indices.flatten():
        neighbor_user = user_item_matrix.index[neighbor_idx]
        neighbor_ratings = user_item_matrix.loc[neighbor_user]
        
        # Add books rated by this neighbor to the recommendation list
        for book_title, rating in neighbor_ratings.items():
            if len(recommended_books_set) >= k:
                break
            if rating >= 4:
                recommended_books_set.add(book_title)
        
        # Stop if we have enough recommendations
        if len(recommended_books_set) >= k:
            break
    return recommended_books_set

def outputs(user_id):

    recommended = recommend_books1(user_id, k=5)
    return {
        "Precision": np.mean(metrics['precision']),
        "recall": np.mean(metrics['recall']),
        "F1_Score": np.mean(metrics['f1']) ,
        "Accuracy": np.mean(metrics['accuracy']),
        "recommended_books": recommended
    }