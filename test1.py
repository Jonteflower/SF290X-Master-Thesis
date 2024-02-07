import pandas as pd

# Define the path to your CSV file
csv_file_path = 'data_up_out.csv'  # Replace with your actual file path

# Define the path for the new CSV file
new_csv_file_path = 'data_up_out_1.csv'  # Replace with your desired file path

# Load the CSV data into a DataFrame
df = pd.read_csv(csv_file_path)

# Function to extract the first element from the string representation of a tuple in 'price_iter' column
def extract_first_value(price_iter_str):
    # Convert the string to a literal tuple
    price_iter_tuple = eval(price_iter_str)
    # Return the first element of the tuple
    return price_iter_tuple[0]

# Apply the function to the 'price_iter' column
df['price_iter'] = df['price_iter'].apply(extract_first_value)

# Save the updated DataFrame to a new CSV file
df.to_csv(new_csv_file_path, index=False)
