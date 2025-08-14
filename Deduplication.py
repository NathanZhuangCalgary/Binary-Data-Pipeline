import pandas as pd


#load orignal csv
in_path = "<folder path>"
              # original CSV
out_path = "<folder path>.csv"      # cleaned CSV

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

