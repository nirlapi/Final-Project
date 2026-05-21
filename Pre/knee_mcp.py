import os
import pandas as pd
from mcp.server.fastmcp import FastMCP

# 1. Initialize the Server
mcp = FastMCP("KneeResearchAssistant")

# 2. Setup Absolute Pathing
# This finds exactly where this script lives on your Mac
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the path to your manifest based on your folder structure
MANIFEST_PATH = os.path.join(
    BASE_DIR, 
    "Data", 
    "exported_dataset_with_augmentation", 
    "train_manifest.csv"
)

@mcp.tool()
def get_train_class_balance() -> str:
    """
    Analyzes the train_manifest.csv to count Class 0 and Class 1.
    Use this to check for data imbalance in the knee segmentation project.
    """
    if not os.path.exists(MANIFEST_PATH):
        # We return a string so the AI can read the error
        return f"Error: Manifest not found at: {MANIFEST_PATH}"
    
    try:
        df = pd.read_csv(MANIFEST_PATH)
        
        # Checking for common label column names
        possible_cols = ['label', 'class', 'target', 'pain']
        label_col = next((c for c in possible_cols if c in df.columns), df.columns[-1])
        
        counts = df[label_col].value_counts().to_dict()
        
        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        total = c0 + c1
        
        if total == 0:
            return "The manifest is empty."
            
        ratio = (c1 / total) * 100
        
        return (
            f"Training Data Summary:\n"
            f"- Path: {MANIFEST_PATH}\n"
            f"- Class 0 (No Pain): {c0}\n"
            f"- Class 1 (Pain): {c1}\n"
            f"- Total Images: {total}\n"
            f"- Positive Class Ratio: {ratio:.2f}%"
        )
        
    except Exception as e:
        return f"An error occurred while reading the data: {str(e)}"

if __name__ == "__main__":
    # IMPORTANT: FastMCP.run uses stdio by default. 
    # Do NOT add print() statements in this file outside of tools.
    mcp.run()
