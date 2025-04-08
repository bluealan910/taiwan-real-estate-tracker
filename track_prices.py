import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data(file_path):
    """
    Load the Taiwan real estate dataset.
    """
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please ensure the dataset is in the 'data' folder.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the dataset: clean and prepare for analysis.
    """
    # Rename columns for clarity (based on typical Taiwan real estate dataset)
    df.columns = [
        "transaction_date", "house_age", "distance_to_mrt", 
        "num_convenience_stores", "latitude", "longitude", "price_per_unit_area"
    ]
    
    # Convert transaction date to datetime (e.g., 2013.250 = March 2013)
    df["transaction_date"] = df["transaction_date"].apply(lambda x: pd.to_datetime(f"{int(x)}-{int((x % 1) * 12) + 1}-01"))
    
    # Drop any rows with missing values
    df = df.dropna()
    
    return df

def analyze_trends(df):
    """
    Analyze and visualize price trends over time.
    """
    # Group by transaction date and calculate average price per unit area
    price_trends = df.groupby("transaction_date")["price_per_unit_area"].mean().reset_index()
    
    # Plot the price trend over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=price_trends, x="transaction_date", y="price_per_unit_area", marker="o")
    plt.title("Taiwan Real Estate Price Trends Over Time")
    plt.xlabel("Transaction Date")
    plt.ylabel("Average Price per Unit Area (10,000 NTD/Ping)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(OUTPUT_DIR / "price_trends.png")
    plt.show()
    
    # Additional analysis: Price vs Distance to MRT
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="distance_to_mrt", y="price_per_unit_area", hue="num_convenience_stores", size="house_age")
    plt.title("Price vs Distance to MRT Station")
    plt.xlabel("Distance to MRT (meters)")
    plt.ylabel("Price per Unit Area (10,000 NTD/Ping)")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(OUTPUT_DIR / "price_vs_mrt.png")
    plt.show()

def main():
    # Load the dataset
    data_path = DATA_DIR / "taiwan_real_estate2.csv"
    df = load_data(data_path)
    
    if df is not None:
        # Preprocess the data
        df = preprocess_data(df)
        
        # Analyze and visualize trends
        analyze_trends(df)

if __name__ == "__main__":
    main()
