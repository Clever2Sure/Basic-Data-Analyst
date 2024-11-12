import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#Task 1
# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows
print("First few rows of the dataset:")
print(iris_df.head())

# Check the structure of the dataset
print("\nData types and info:")
print(iris_df.info())

# Check for missing values
print("\nMissing values:")
print(iris_df.isnull().sum())

# Clean the dataset (no missing values in this case)
# If there were missing values, we could use:
# iris_df.dropna(inplace=True)  # or use .fillna() if needed

print("\nDataset cleaned and ready for analysis.")

#Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic statistics of numerical columns:")
print(iris_df.describe())

# Group by 'species' and compute the mean of numerical columns
grouped_means = iris_df.groupby('species').mean()
print("\nMean of numerical columns for each species:")
print(grouped_means)

# Observations
print("\nObservations:")
print("The grouped means help identify how the different iris species compare in features such as petal length and sepal width.")

#Task 3: Data Visualization

# Line chart showing trends over time (simulated example since the Iris dataset isn't time-series data)
plt.figure(figsize=(10, 5))
sns.lineplot(data=iris_df.iloc[:, :-1])
plt.title("Line Chart of Iris Feature Trends")
plt.xlabel("Sample Index")
plt.ylabel("Feature Values")
plt.legend(iris.feature_names, loc='upper right')
plt.show()

# Bar chart showing average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=iris_df, estimator='mean')
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# Histogram of a numerical column (e.g., petal length)
plt.figure(figsize=(8, 5))
sns.histplot(iris_df['petal length (cm)'], bins=20, kde=True)
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot for sepal length vs. petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris_df)
plt.title("Sepal Length vs. Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

#Error Handling:

try:
    # Example of file handling error check
    df = pd.read_csv('non_existent_file.csv')
except FileNotFoundError:
    print("The specified file does not exist. Please check the filename and try again.")
except pd.errors.EmptyDataError:
    print("The file is empty. Please provide a valid dataset.")
except Exception as e:
    print(f"An error occurred: {e}")