import pandas as pd

def clean_antibody_data_csv(input_file, output_file=None, sheet_name=0):
    """
    Clean antibody data from Excel file by:
    1. Keeping only Rheumatoid Arthritis and Sjögren's Syndrome rows
    2. Keeping only Disease column and the 5 antibody columns
    3. Removing rows with any empty cells in the antibody columns
    
    Parameters:
    input_file (str): Path to input Excel file
    output_file (str): Path to output CSV file (optional)
    sheet_name (str/int): Sheet name or index to read (default: 0 = first sheet)
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    # Read the Excel file
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    
    # Define the columns we want to keep
    target_columns = ['Disease', 'HLA-B27', 'RF_abn', 'Anti-dsDNA', 'Anti-Sm']
    
    # Keep only the columns we're interested in
    df = df[target_columns]
    
    # Filter rows to keep only Rheumatoid Arthritis and Sjögren's Syndrome
    disease_filter = df['Disease'].isin(['Rheumatoid Arthritis', 'Sjögren\'s Syndrome'])
    df = df[disease_filter]
    
    # Remove rows with any empty cells in the antibody columns
    antibody_columns = ['HLA-B27', 'RF_abn', 'Anti-dsDNA', 'Anti-Sm']
    df = df.dropna(subset=antibody_columns)
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    # Save to output file if specified - now as CSV
    if output_file:
        if output_file.endswith('.csv'):
            df.to_csv(output_file, index=False)
        else:
            df.to_csv(output_file + '.csv', index=False)
        print(f"Cleaned data saved to: {output_file}")
    
    print(f"Original data shape: {pd.read_excel(input_file, sheet_name=sheet_name).shape}")
    print(f"Cleaned data shape: {df.shape}")
    print(f"Rows kept: {len(df)}")
    
    return df

# More detailed version with progress reporting
def detailed_clean_antibody_data_csv(input_file, output_file=None, sheet_name=0):
    """
    Detailed version with more error handling and progress reporting for Excel files
    Outputs CSV instead of Excel
    """
    
    try:
        # Read the Excel file
        print(f"Reading data from: {input_file}")
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        original_shape = df.shape
        print(f"Original data: {original_shape[0]} rows, {original_shape[1]} columns")
        
        # Check if required columns exist
        required_columns = ['Disease', 'HLA-B27', 'RF_abn', 'Anti-dsDNA', 'Anti-Sm']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
        
        # Step 1: Keep only required columns
        print("\nStep 1: Keeping only required columns...")
        df = df[required_columns]
        print(f"After column filtering: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Step 2: Filter for target diseases
        print("\nStep 2: Filtering for Rheumatoid Arthritis and Sjögren's Syndrome...")
        disease_counts_before = df['Disease'].value_counts()
        print("Disease distribution before filtering:")
        print(disease_counts_before)
        
        disease_filter = df['Disease'].isin(['Rheumatoid Arthritis', 'Sjögren\'s Syndrome'])
        df = df[disease_filter]
        print(f"After disease filtering: {df.shape[0]} rows")
        
        # Step 3: Remove rows with empty antibody values
        print("\nStep 3: Removing rows with empty antibody values...")
        antibody_columns = ['HLA-B27', 'RF_abn', 'Anti-dsDNA', 'Anti-Sm', 'HLA-B27']
        rows_before_na = len(df)
        
        df = df.dropna(subset=antibody_columns)
        rows_removed = rows_before_na - len(df)
        
        print(f"Removed {rows_removed} rows with empty antibody values")
        print(f"Final data: {df.shape[0]} rows")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Final disease distribution
        print("\nFinal disease distribution:")
        print(df['Disease'].value_counts())
        
        # Save to output file if specified - now as CSV
        if output_file:
            if not output_file.endswith('.csv'):
                output_file += '.csv'
            df.to_csv(output_file, index=False)
            print(f"\nCleaned data saved to CSV: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    input_filename = r"C:\Users\natha\Desktop\ClinicalDatasetResearch\CleanedDataResults\CleanedData\cleaned_dataset.xlsx"  # Your input Excel file
    output_filename = 'cleaned_antibody_data.csv'  # Output CSV file
    
    # If you have multiple sheets, specify the sheet name or index:
    # sheet_name = 'Sheet1'  # or sheet_name = 0 for first sheet
    
    try:
        print("=== Cleaning Antibody Data from Excel File ===")
        cleaned_data = detailed_clean_antibody_data_csv(
            input_file=input_filename, 
            output_file=output_filename
            # sheet_name='Sheet1'  # Uncomment and modify if needed
        )
        
        print("\n" + "="*50)
        print("First 10 rows of cleaned data:")
        print("="*50)
        print(cleaned_data.head(10))
        
        print(f"\nData types:")
        print(cleaned_data.dtypes)
        
    except FileNotFoundError:
        print(f"Error: Excel file '{input_filename}' not found.")
        print("Please make sure the file exists and the path is correct.")
    except KeyError as e:
        print(f"Error: Required column not found in the data: {e}")
        print("Please check that your Excel file has the correct column names.")
    except ImportError:
        print("Error: Required package not installed.")
        print("Please install openpyxl or xlrd: pip install openpyxl")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")