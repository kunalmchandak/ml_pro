import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_dataset(df, target_column=None):
    """
    Comprehensive dataset cleaning including:
    1. Removing duplicates
    2. Handling dirty values
    3. Detecting and converting column types
    4. Processing date columns
    5. Handling missing values
    6. Encoding categorical variables
    7. Scaling numerical features
    """
    cleaning_report = {
        'initial_shape': df.shape,
        'duplicates_removed': 0,
        'missing_values': {},
        'encoded_columns': [],
        'removed_columns': [],
        'type_conversions': [],
        'date_features_added': []
    }
    
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # -----------------------------
    # 1. Remove duplicates
    # -----------------------------
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    cleaning_report['duplicates_removed'] = initial_rows - len(cleaned_df)
    
    # -----------------------------
    # 2. Replace common dirty strings with NaN
    # -----------------------------
    dirty_values = ["NaN", "nan", " ", "", "?", "??", "error", "invalid", "none", "None", "#N/A", "NA", "null", "NULL"]
    cleaned_df = cleaned_df.replace(dirty_values, np.nan)
    
    # -----------------------------
    # 3. Detect column types
    # -----------------------------
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    
    for col in cleaned_df.columns:
        # Try to convert to numeric
        try:
            pd.to_numeric(cleaned_df[col])
            numeric_cols.append(col)
            cleaning_report['type_conversions'].append({
                'column': col,
                'type': 'numeric'
            })
        except:
            # Try to convert to datetime
            try:
                pd.to_datetime(cleaned_df[col])
                date_cols.append(col)
                cleaning_report['type_conversions'].append({
                    'column': col,
                    'type': 'date'
                })
            except:
                categorical_cols.append(col)
                cleaning_report['type_conversions'].append({
                    'column': col,
                    'type': 'categorical'
                })
    
    # -----------------------------
    # 4. Clean date columns
    # -----------------------------
    for col in date_cols:
        cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors="coerce")
        # Extract date features
        cleaned_df[f"{col}_year"] = cleaned_df[col].dt.year
        cleaned_df[f"{col}_month"] = cleaned_df[col].dt.month
        cleaned_df[f"{col}_day"] = cleaned_df[col].dt.day
        # Add new numeric columns to numeric_cols
        numeric_cols.extend([f"{col}_year", f"{col}_month", f"{col}_day"])
        cleaning_report['date_features_added'].extend([f"{col}_year", f"{col}_month", f"{col}_day"])
    
    # Drop original date columns
    if date_cols:
        cleaned_df = cleaned_df.drop(columns=date_cols)
        cleaning_report['removed_columns'].extend([{
            'column': col,
            'reason': 'date_column_converted_to_features'
        } for col in date_cols])
    
    # -----------------------------
    # 5. Convert numeric columns properly
    # -----------------------------
    for col in numeric_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
    
    # -----------------------------
    # 6. Handle missing values
    # -----------------------------
    for col in cleaned_df.columns:
        missing_count = cleaned_df[col].isnull().sum()
        if missing_count > 0:
            if col in numeric_cols:
                median_value = cleaned_df[col].median()
                cleaned_df[col].fillna(median_value, inplace=True)
                cleaning_report['missing_values'][col] = {
                    'count': int(missing_count),
                    'action': 'filled_with_median',
                    'value': float(median_value)
                }
            else:
                mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "unknown"
                cleaned_df[col].fillna(mode_value, inplace=True)
                cleaning_report['missing_values'][col] = {
                    'count': int(missing_count),
                    'action': 'filled_with_mode',
                    'value': str(mode_value)
                }
    
    # -----------------------------
    # 7. Encode categorical variables (including target)
    # -----------------------------
    # Use a fresh LabelEncoder per column to avoid cross-column mapping
    for col in categorical_cols:
        # encode all categorical columns including the target
        le = LabelEncoder()
        cleaned_df[col] = le.fit_transform(cleaned_df[col].astype(str))
        cleaning_report['encoded_columns'].append({
            'column': col,
            'unique_values': int(cleaned_df[col].nunique())
        })
    
    # -----------------------------
    # 8. Scale numerical features
    # -----------------------------
    if numeric_cols:
        scaler = StandardScaler()
        cleaned_df[numeric_cols] = scaler.fit_transform(cleaned_df[numeric_cols])
    
    # Add final shape to report
    cleaning_report['final_shape'] = cleaned_df.shape
    
    return cleaned_df, cleaning_report