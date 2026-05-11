import pandas as pd
import numpy as np
import json
def add_mean_row_to_csv(input_file, group_file, overall_file):
    
    df = pd.read_csv(input_file, na_values='NaN')

    df['category'] = df['file_id'].apply(category_func)
    df.drop('file_id', axis=1, inplace=True)
    group_means = df.groupby('category',dropna=True).mean()

    group_means.to_csv(group_file)

    print(f"\n successfully save to  {group_file}")

    overall_mean = df.mean(numeric_only=True, skipna=True)
    overall_mean.to_csv(overall_file)
    
def category_func(x):
    x = str(x)
    if len(x) < 4:
        return '0'
    else:
        return x[0]
if __name__ == "__main__":
    input_file = 'evaluation/test_result.csv'
    group_file = 'evaluation/group_test_result.csv'
    overall_file = 'evaluation/all_test_result.csv'
    
    add_mean_row_to_csv(input_file, group_file, overall_file)