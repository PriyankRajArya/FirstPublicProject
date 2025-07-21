# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
df = pd.read_csv("student_performance_expanded.csv")

# Step 3: Preview the data
print("First few rows:\n", df.head())
print("\nData Summary:\n", df.describe())

# Step 4: Calculate average score per subject
subject_avg = df.groupby("Subject")["Score"].mean().sort_values(ascending=False)

# Step 5: Bar plot
plt.figure(figsize=(8,5))
sns.barplot(x=subject_avg.values, y=subject_avg.index, palette="viridis")
plt.title("Average Scores by Subject")
plt.xlabel("Average Score")
plt.ylabel("Subject")
plt.tight_layout()
plt.show()

# Step 6: Scatter plot - Attendance vs Score
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Attendance (%)", y="Score", hue="Subject", palette="Set2", s=100)
plt.title("Attendance vs Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Correlation between Attendance and Score
correlation = df[["Attendance (%)", "Score"]].corr()
print("\nCorrelation between Attendance and Score:\n", correlation)

# Step 8: Plot grade distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Grade", order=sorted(df["Grade"].unique()), palette="pastel")
plt.title("Grade Distribution")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# Step 9: Create pivot table (Student x Subject)
pivot_table = df.pivot_table(index="Student", columns="Subject", values="Score")

# Step 10: Heatmap of subject correlations
plt.figure(figsize=(8,6))
sns.heatmap(pivot_table.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Subject Score Correlation")
plt.tight_layout()
plt.show()
