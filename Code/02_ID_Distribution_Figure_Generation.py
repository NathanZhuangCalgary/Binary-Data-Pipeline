import matplotlib.pyplot as plt

# Data from synthetic dataset
categories = ["RA", "SS", "RF+ SS", "RF- SS"]
counts = [91, 85, 38, 47]
colors = ["#1f77b4", "#2ca02c", "#17becf", "#98df8a"]  # blue and green shades

# Create bar graph
plt.figure(figsize=(8,6))
bars = plt.bar(categories, counts, color=colors)

# Add labels and title
plt.xlabel("Patient Groups")
plt.ylabel("Number of Patients")
plt.title("Patient Distribution in Comprehensive Autoimmune Disorder Dataset (n=176)")

# Add gridlines (major + minor)
plt.grid(which="major", linestyle="-", linewidth=0.7, alpha=0.7)
plt.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.5)
plt.minorticks_on()

# Show values on bars
for i, count in enumerate(counts):
    plt.text(i, count + 1, str(count), ha='center', va='bottom')

plt.show()
