
# ============================================================
# PROJECT: STUDENT PERFORMANCE ANALYSIS
# Tools: Python, Pandas, Matplotlib
# Type: Exploratory Data Analysis (EDA)
# ============================================================

# Import Pandas for data analysis
import pandas as pd

# Import Matplotlib for charts
import matplotlib.pyplot as plt

# Load the CSV file
student = pd.read_csv("student_data.csv")

# Display the first five rows
print("\n--- First Five Rows ---")
print(student.head())

# Display the number of rows and columns
print("\n--- Dataset Shape ---")
print(student.shape)

# Display all column names
print("\n--- Column Names ---")
print(student.columns.tolist())

# Display column data types and non-null counts
print("\n--- Dataset Information ---")
student.info()

# Count missing values in every column
print("\n--- Missing Values ---")
print(student.isnull().sum())

# Count duplicate rows
print("\n--- Duplicate Rows ---")
print(student.duplicated().sum())

# Display statistics for numeric columns
print("\n--- Descriptive Statistics ---")
print(student.describe())

# Calculate minimum final grade
print("\nMinimum Final Grade:", student["G3"].min())

# Calculate maximum final grade
print("Maximum Final Grade:", student["G3"].max())

# Calculate average final grade
print("Average Final Grade:", round(student["G3"].mean(), 2))

# G1 = first-period grade
# G2 = second-period grade
# G3 = final grade

print("\n--- Average Exam Grades ---")

for grade in ["G1", "G2", "G3"]:
    print(f"Average {grade}: {student[grade].mean():.2f}")

# Calculate average marks for each examination
averages = student[["G1", "G2", "G3"]].mean()

# Create a bar chart
plt.figure(figsize=(8, 5))
plt.bar(averages.index, averages.values)

# Add titles and labels
plt.title("Average Student Grades")
plt.xlabel("Examinations")
plt.ylabel("Average Marks")

# Display the values above the bars
for index, value in enumerate(averages.values):
    plt.text(index, value, f"{value:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()

# Calculate average final grades for each study-time category
study_analysis = student.groupby("studytime")["G3"].mean()

# Study-time categories used by the dataset
study_hours = {
    1: "Less than 2 hours/week",
    2: "2 to 5 hours/week",
    3: "5 to 10 hours/week",
    4: "More than 10 hours/week"
}

# Display the results clearly
print("\n--- Average Final Grades by Study Time ---")

for group, average_marks in study_analysis.items():
    label = study_hours.get(group, f"Category {group}")
    print(f"{label}: {average_marks:.2f}")

# Convert category codes into readable labels
study_labels = [
    study_hours.get(group, f"Category {group}")
    for group in study_analysis.index
]

# Create a bar chart
plt.figure(figsize=(10, 5))
plt.bar(study_labels, study_analysis.values)

plt.title("Study Time vs Average Final Grade")
plt.xlabel("Weekly Study-Time Category")
plt.ylabel("Average Final Grade")
plt.xticks(rotation=20, ha="right")

# Add average values to the chart
for index, value in enumerate(study_analysis.values):
    plt.text(index, value, f"{value:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()

# Calculate the correlation between absences and final grades
absence_correlation = student["absences"].corr(student["G3"])

print("\n--- Absences and Final Grades ---")
print(f"Correlation: {absence_correlation:.3f}")

# Create a scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(student["absences"], student["G3"], alpha=0.6)

plt.title("Student Absences vs Final Grades")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")

plt.tight_layout()
plt.show()

# Select numerical variables relevant to student performance
selected_columns = [
    "age",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2",
    "G3"
]

# Calculate the correlation matrix
correlation_matrix = student[selected_columns].corr()

print("\n--- Correlation Matrix ---")
print(correlation_matrix.round(2))

# Create a heatmap using Matplotlib
plt.figure(figsize=(9, 7))

image = plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

# Add variable names to the axes
plt.xticks(
    range(len(selected_columns)),
    selected_columns,
    rotation=45,
    ha="right"
)

plt.yticks(range(len(selected_columns)), selected_columns)

# Display correlation values inside the cells
for row in range(len(selected_columns)):
    for column in range(len(selected_columns)):
        value = correlation_matrix.iloc[row, column]
        plt.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center"
        )

plt.colorbar(image, label="Correlation")
plt.title("Correlation Between Student Performance Variables")
plt.tight_layout()
plt.show()

# Compare average final grades by gender
print("\n--- Average Final Grade by Gender ---")
print(student.groupby("sex")["G3"].mean().round(2))

# Compare average final grades by school
print("\n--- Average Final Grade by School ---")
print(student.groupby("school")["G3"].mean().round(2))

# Compare average final grades by family support
print("\n--- Average Final Grade by Family Support ---")
print(student.groupby("famsup")["G3"].mean().round(2))

# Compare average final grades by internet access
print("\n--- Average Final Grade by Internet Access ---")
print(student.groupby("internet")["G3"].mean().round(2))

# Select the group we want to compare
group_column = "famsup"

# Calculate average final grade for each group
group_averages = student.groupby(group_column)["G3"].mean()

# Create a bar chart
plt.figure(figsize=(7, 5))
plt.bar(group_averages.index.astype(str), group_averages.values)

plt.title("Family Support vs Average Final Grade")
plt.xlabel("Family Support")
plt.ylabel("Average Final Grade")

# Display average values above the bars
for index, value in enumerate(group_averages.values):
    plt.text(index, value, f"{value:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()

# Calculate average final grade for each number of past failures
failure_analysis = student.groupby("failures")["G3"].mean()

print("\n--- Average Final Grade by Past Failures ---")
print(failure_analysis.round(2))

# Visualize the relationship
plt.figure(figsize=(8, 5))
plt.bar(
    failure_analysis.index.astype(str),
    failure_analysis.values
)

plt.title("Past Failures vs Average Final Grade")
plt.xlabel("Number of Past Class Failures")
plt.ylabel("Average Final Grade")

for index, value in enumerate(failure_analysis.values):
    plt.text(index, value, f"{value:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()

# Find the study-time category with the highest average grade
if not study_analysis.empty:
    best_study_group = study_analysis.idxmax()

    print("\n--- Key Findings ---")
    print(
        "Study-time category with the highest average final grade:",
        study_hours.get(best_study_group, best_study_group)
    )

# Find the group with the highest average grade for family support
if not group_averages.empty:
    print(
        "Family support group with the highest average final grade:",
        group_averages.idxmax()
    )

# Display the correlation between previous and final grades
print(
    "Correlation between G2 and G3:",
    round(student["G2"].corr(student["G3"]), 3)
)

print(
    "Correlation between absences and G3:",
    round(student["absences"].corr(student["G3"]), 3)
)

print("\nStudent Performance Analysis completed!")
