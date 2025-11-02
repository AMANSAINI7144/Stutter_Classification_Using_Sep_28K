import os
import pandas as pd

# Get absolute path to the current script directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Build absolute paths to the CSV files
episodes_path = os.path.join(base_dir, "../data/SEP28K/SEP-28k_episodes.csv")
labels_path = os.path.join(base_dir, "../data/SEP28K/SEP-28k_labels.csv")

# Read the CSV files
episodes = pd.read_csv(episodes_path)
labels = pd.read_csv(labels_path)

# Print information
print("Episodes columns:", episodes.columns.tolist())
print("Labels columns:", labels.columns.tolist())
print("Episodes shape:", episodes.shape)
print("Labels shape:", labels.shape)
