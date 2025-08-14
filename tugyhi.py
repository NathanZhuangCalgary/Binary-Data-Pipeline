import pandas as pd



in_path = "/Users/nathanzhuang/Documents/Polygence Reasearch Project/Complete_Updated_Autoimmune_Disorder_Dataset.csv"
              # your original CSV
out_path = "your_file_deduped.csv"      # cleaned CSV

# Load dataset
df = pd.read_csv(in_path)

# Remove duplicates based on Patient_ID, keeping the first occurrence
df_clean = df.drop_duplicates(subset=["Patient_ID"], keep="first")

print(f"Starting rows: {len(df)}")
print(f"Rows after removing duplicates: {len(df_clean)}")
print(f"Duplicates removed: {len(df) - len(df_clean)}")

# Save the cleaned file
df_clean.to_csv(out_path, index=False)

print(f"Cleaned file saved as: {out_path}")

