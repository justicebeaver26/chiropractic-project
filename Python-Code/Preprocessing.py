#!/usr/bin/env python
# coding: utf-8

# # Subluxation Data Preprocessing

# ## Importing and Loading Datasets

# In[74]:


import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
file_path_active = "Patient Management Active 03102022 Only ID.xlsx"
df_active = pd.read_excel(file_path_active)

# Add a new column 'Active/Passive' to df_active with all values set to 'Active'
df_active.insert(3, 'Active/Passive', 'Active')  # Insert column

file_path_passive = "Patient Management Passive 03102022 Only ID.xlsx"
df_passive = pd.read_excel(file_path_passive)

# Add a new column 'Active/Passive' to df_active with all values set to 'Passive'
df_passive.insert(3, 'Active/Passive', 'Passive')  # Insert column


# ### View the Active and Passive datasets

# In[75]:


# Print the updated DataFrame
df_active.head()


# In[76]:


# Print the updated DataFrame
df_passive.head()


# ## Creating a dataframe for Vertebral Subluxation columns

# In[77]:


# Combine df_active and df_passive
active_passive_combined = pd.concat([df_active, df_passive], ignore_index=True)


# In[78]:


vs_columns = [col for col in active_passive_combined.columns if "Vertebral Subluxation Before Care" in col]
if vs_columns:
    vs_df = active_passive_combined[vs_columns].copy()
    vs_df.columns = [f"Vertebral Subluxation Before Care {idx+1}" for idx in range(len(vs_columns))]
    


# ### Processing the Vertebral Subluxation Data

# In[79]:


def process_vs_data(df):
    """Extracts labels from 'Vertebral Subluxation Before Care' and assigns severity based on '+' count."""
    if "Vertebral Subluxation Before Care" in vs_df.columns:
        vs_data = vs_df["Vertebral Subluxation Before Care"].fillna("")
        extracted = []
        unique_labels = set()

        for entry in vs_data:
            matches = re.findall(r"([A-Za-z0-9]+(?: [A-Za-z]+)?)\s*(\+{1,3})?", entry)
            
            row_dict = {}
            for label, severity in matches:
                label = label.upper().strip()  # Convert to uppercase
                severity_value = len(severity) if severity else 1  # Number of '+' determines severity (default 1)
                unique_labels.add(label)
                row_dict[label] = severity_value  # Store severity value

            extracted.append(row_dict)

        label_list = sorted(unique_labels)  # Sort labels alphabetically
        transformed_data = [{label: row.get(label, 0) for label in label_list} for row in extracted]

        return pd.DataFrame(transformed_data)
    else:
        print("Column 'Vertebral Subluxation Before Care' not found.")
        return pd.DataFrame()  # Return empty DataFrame if column is missing


# ### Transforming and Merging Data

# Since Vertebral Subluxation Before Care Columns are still not found, we will process them separately.

# In[80]:


# Process active and passive datasets
df_transformed_active = process_vs_data(df_active)
df_transformed_passive = process_vs_data(df_passive)

#  Merge transformed data with metadata
df_active_final = pd.concat([df_active, df_transformed_active], axis=1)
df_passive_final = pd.concat([df_passive, df_transformed_passive], axis=1)

#  Combine active and passive datasets
df_combined = pd.concat([df_active_final, df_passive_final], ignore_index=True)
df_combined = pd.concat([df_combined, vs_df], axis=1)

# Save intermediate data
#df_combined.to_csv("data.csv", index=False)
#print(f" Data saved to 'data.csv' with {len(df_combined)} rows.")


# ## Processing Session Data

# In[81]:


# Checking for Duplicates 
print(df_active.columns[df_active.columns.duplicated()])
print(df_passive.columns[df_passive.columns.duplicated()])


# ## Processing Chiro Adjustments Data

# In[82]:


###  Step 3: Process Chiro Adjustments
chiro_adjustment_columns = [col for col in df_active.columns if "Chiro Adjustment" in col]
if chiro_adjustment_columns:
    chiro_adjustment_df = df_active[chiro_adjustment_columns].copy()
    chiro_adjustment_df.columns = [f"Chiro Adjustment {idx+1}" for idx in range(len(chiro_adjustment_columns))]
    df_combined = pd.concat([df_combined, chiro_adjustment_df], axis=1)

#  Save the final combined dataset
#final_output_path = "data_final.csv"
#df_combined.to_csv(final_output_path, index=False)
#print(f" Final data saved to '{final_output_path}'.")

###  Step 4: Reshape & Visualize Session Trends
#  Reload the cleaned dataset
#df = pd.read_csv("data_final.csv")

df = df_combined

#  Detect all session columns
session_columns = [col for col in df.columns if re.match(r"Time/Session \d+", col)]
if not session_columns:
    print(" No session columns found, skipping trend analysis.")
else:
    #  Reshape data: Convert multiple session columns into 'Session' and 'Time'
    df_sessions = df.melt(
        id_vars=[col for col in df.columns if col not in session_columns], 
        value_vars=session_columns,
        var_name="Session",
        value_name="Time"
    )

    #  Extract session numbers (e.g., 'Time/Session 1' → 1)
    df_sessions["Session"] = df_sessions["Session"].str.extract(r'(\d+)').astype(int)


print(" All processing and visualization steps completed.")


# ### Extracting Labels from the data 

# We will extract the labels from the dataset, and create separate columns for each. For example, if the label is PL SX +++, then it would be processed as "Session 1 PLSX". Additionally, the "+" indicates the severity. Since there are three "+", the severity is 3. 

# In[83]:


#data = df_combined
vs_df = df_combined.filter(regex=r"^Vertebral Subluxation Before Care(\s\d+)?$")  # Select relevant columns

# List of labels to exclude
exclude_labels = []

# Function to extract label and severity, while standardizing label case to uppercase and removing spaces
def extract_label_severity(text):
    if isinstance(text, str):
        match = re.match(r"(.+?)(\++$)", text)  # Extract label and plus signs
        if match:
            label = match.group(1).strip().replace(" ", "").upper()  # Convert to uppercase, remove spaces
            severity = len(match.group(2))  # Count '+' signs
            return label, severity
    return None, 0  # Default case for empty/missing values

# Process each column to extract labels and severities
processed_data = []
for index, row in vs_df.iterrows():
    row_data = {}
    for col in vs_df.columns:
        session_num = re.search(r"(\d+)?$", col).group(1)  # Extract session number (if present)
        session_num = session_num if session_num else "1"  # Default to 1 if no number is found
        entries = str(row[col]).split(",")  # Split multiple labels by comma
        for entry in entries:
            label, severity = extract_label_severity(entry.strip())  # Clean label by removing extra spaces
            if label and label not in exclude_labels:  # Exclude specific labels
                formatted_label = label  # Already uppercase and no spaces
                column_name = f"Session {session_num} {formatted_label}"
                row_data[column_name] = severity
    processed_data.append(row_data)

# Create DataFrame with extracted labels and renamed columns
result_df = pd.DataFrame(processed_data).fillna(0).astype(int)  # Fill missing values with 0

# Remove columns that have "+" in their names
result_df = result_df.loc[:, ~result_df.columns.str.contains(r'\+')]

# Sort columns by session number while maintaining label order within each session
sorted_columns = sorted(result_df.columns, key=lambda x: (int(re.search(r"Session (\d+)", x).group(1)), x))
result_df = result_df[sorted_columns]  # Reorder columns

# Print result
result_df.head()


# ## Extract Time/Session columns from df_combined

# We will extract Time/Session columns from df_combined. We will also remove any duplicates. Additionally, we will rename the Time/Session (number) columns as Session (number) columns.

# In[84]:


# Trim spaces from column names
df_combined.columns = df_combined.columns.str.strip()

# Check for duplicate columns
duplicate_cols = df_combined.columns[df_combined.columns.duplicated()].tolist()
if duplicate_cols:
    print(f"Duplicate columns found: {duplicate_cols}")

# If duplicate columns exist, keep only the first occurrence
df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]

# Extract all "Time/Session" columns
time_df = df_combined.filter(regex=r"^Time/Session(\s*\d+)?$")

# Rename columns from "Time/Session X" to "Session X"
time_df.columns = [re.sub(r"^Time/Session\s*", "Session ", col) for col in time_df.columns]

# Sort columns numerically based on session number
sorted_columns = sorted(time_df.columns, key=lambda x: int(re.search(r"\d+", x).group(0)) if re.search(r"\d+", x) else 1)

# Reorder DataFrame
time_df = time_df[sorted_columns]

# Display updated DataFrame
time_df.head()


# ### Standardise Session columns

# We will standardise the Session columns.

# In[85]:


def standardize_time_format(time_str):
    # Match the time and date with or without AM/PM
    match = re.match(r"(\d{1,2}:\d{2})([APap][Mm])?\s*(\d{1,2}/\d{1,2})", time_str.strip())
    
    if match:
        time_part = match.group(1)
        am_pm = match.group(2)
        date_part = match.group(3)

        # If AM/PM is missing, infer it
        if not am_pm:
            hour = int(time_part.split(":")[0])
            if 8 <= hour < 12:
                am_pm = "AM"
            elif 12 <= hour <= 6 or hour == 12:
                am_pm = "PM"
            else:
                # Default to AM if outside expected bounds (defensive programming)
                am_pm = "AM"
        else:
            am_pm = am_pm.upper()

        # Return standardized time
        standardized_time = f"{time_part}{am_pm} {date_part}"
        return standardized_time
    
    return time_str  # Return original if no match

# Apply to all Time/Session columns
for col in time_df.columns:
    time_df[col] = time_df[col].apply(lambda x: standardize_time_format(str(x)))

# Preview
time_df.head()


# ### Combine Label and Session Columns

# In[86]:


# Combine time_df and result_df along the columns
time_label = pd.concat([time_df, result_df], axis=1)

# Print the combined DataFrame
time_label.head()


# ### Rearranging Session and Session Label columns

# In[87]:


# Extract unique session numbers
session_numbers = sorted(set(int(re.search(r"\d+", col).group(0)) for col in time_label.columns if re.match(r"^Session \d+", col)))

# Arrange columns: Each "Session X" is followed by its associated labels
ordered_columns = []
for session in session_numbers:
    related_cols = [col for col in time_label.columns if re.match(rf"^Session {session}(\s|$)", col)]
    ordered_columns.extend(sorted(related_cols))  # Maintain original order for labels

# Reorder DataFrame
time_label = time_label[ordered_columns]

# Print the final column order
print(time_label.columns)


# ## Combine the above columns with the rest of the dataset

# In[88]:


# Extract relevant columns from df_combined
cols_to_extract = ["ID", "Enrolment Date", "Initial Care Plan", "Age Group", 'Active/Passive'] + [col for col in df_combined.columns if col.startswith("Chiropractic Adjustment")]
extracted_df = df_combined[cols_to_extract]

# Combine extracted columns with combined_df
final_combined_df = pd.concat([extracted_df, time_label, chiro_adjustment_df], axis=1)

# Print the final combined DataFrame
final_combined_df.head()


# ### Save the cleaned dataset 

# In[89]:


final_combined_df.to_csv("final_combined_df.csv", index=False)


# ## Weighted Subluxation Scores

# Each subluxation score needs to be multiplied by weights specified by the client. For C1 and C2, each score would be multiplied by 3. For C3 to C7, each score would be multiplied by 2. The rest will remain the same.
# 
# We will use result_df, which is the dataframe with all the subluxation scores (label wise). 

# In[90]:


result_df.head()


# In[91]:


# Multiply columns with ' C1' or ' C2' by 3
weight3_df = result_df.loc[:, result_df.columns.str.contains(r' C1| C2')]
weight3_multiplied = weight3_df * 3

# Multiply columns with ' C3' to ' C7' by 2
weight2_df = result_df.loc[:, result_df.columns.str.contains(r' C3| C4| C5| C6| C7')]
weight2_multiplied = weight2_df * 2

# All other columns unchanged
weight1_df = result_df.loc[:, ~result_df.columns.str.contains(r' C1| C2| C3| C4| C5| C6| C7')]

# Combine everything
weighted_df = pd.concat([weight1_df, weight2_multiplied, weight3_multiplied], axis=1)

# Optional: reorder to match the original DataFrame
weighted_df = weighted_df[result_df.columns]

weighted_df


# ### Combine with time and session columns

# In[92]:


# Combine time_df and result_df along the columns
weight_scores = pd.concat([time_df, weighted_df], axis=1)

# Sort columns numerically based on extracted session numbers
sorted_columns = sorted(weight_scores.columns, key=lambda x: int(''.join(filter(str.isdigit, x))))

# Rearrange DataFrame
weight_scores = weight_scores[sorted_columns]

# Print the combined DataFrame
#weight_scores.head()


# In[93]:


# Extract relevant columns from df_combined
cols_to_extract = ["ID", "Enrolment Date", "Initial Care Plan", 'Age Group', 'Active/Passive'] 
extracted_df = df_combined[cols_to_extract]

# Combine extracted columns with combined_df
weight_scores_noaverage = pd.concat([extracted_df, weight_scores], axis=1)

# Print the final combined DataFrame
weight_scores_noaverage.head()

weight_scores_noaverage.to_csv("weight_noaverage.csv", index=False)


# ### Calculate the sum of weighted products for each Session

# In[94]:


df = pd.DataFrame(weighted_df)

# Identify unique session prefixes dynamically
session_groups = {}
for col in df.columns:
    session_prefix = " ".join(col.split()[:2])  # Extract "Session X" prefix
    if session_prefix.startswith("Session"):
        session_groups.setdefault(session_prefix, []).append(col)

# Compute simple sums instead of weighted products
weighted_df = df.copy()
for session, cols in session_groups.items():
    weighted_df[session + " Weighted Sum"] = df[cols].sum(axis=1)

# Display session-wise sum
session_sums = weighted_df[[col + " Weighted Sum" for col in session_groups]]
session_sums.head()


# ### Combine Weight Session and Time columns

# In[95]:


# Combine time_df and result_df along the columns
weight_time = pd.concat([time_df, session_sums], axis=1)

# Sort columns numerically based on extracted session numbers
sorted_columns = sorted(weight_time.columns, key=lambda x: int(''.join(filter(str.isdigit, x))))

# Rearrange DataFrame
weight_time = weight_time[sorted_columns]

# Print the combined DataFrame
weight_time.head()


# ### Combine with rest of the data

# In[96]:


# Extract relevant columns from df_combined
cols_to_extract = ["ID", "Enrolment Date", "Initial Care Plan", 'Age Group', 'Active/Passive'] 
extracted_df = df_combined[cols_to_extract]

# Combine extracted columns with combined_df
weight_subscores = pd.concat([extracted_df, weight_time], axis=1)

# Print the final combined DataFrame
weight_subscores.head()


# ### Save the above dataset

# In[97]:


weight_subscores.to_csv("weight_subscores.csv", index=False)


# ## Creating a dataframe for Visits per Session

# In[98]:


# Merge age data into vs_df
vs_df["Age Group"] = final_combined_df["Age Group"]

# Define session columns
vs_columns = [col for col in vs_df.columns if col.startswith("Vertebral Subluxation Before Care")]

# Split into Active and Passive groups
active_df = vs_df.iloc[:109]  # First 109 rows are Active
passive_df = vs_df.iloc[109:]  # Remaining rows are Passive

# Further split into Children and Adults
active_children = active_df[active_df["Age Group"] == "Children"]
active_adults = active_df[active_df["Age Group"] == "Adults"]
passive_children = passive_df[passive_df["Age Group"] == "Children"]
passive_adults = passive_df[passive_df["Age Group"] == "Adults"]

# Count visits for each session
active_children_counts = [active_children[col].notna().sum() for col in vs_columns]
active_adults_counts = [active_adults[col].notna().sum() for col in vs_columns]
passive_children_counts = [passive_children[col].notna().sum() for col in vs_columns]
passive_adults_counts = [passive_adults[col].notna().sum() for col in vs_columns]

# Create a new DataFrame in the required format
session_visits_df = pd.DataFrame(
    [active_children_counts, active_adults_counts, passive_children_counts, passive_adults_counts], 
    index=["Children (Active)", "Adults (Active)", "Children (Passive)", "Adults (Passive)"], 
    columns=[f"Session {i+1} Visits" for i in range(len(vs_columns))]
)

# Display the result
session_visits_df


# In[99]:


# Rename the index to "Age Group"
session_visits_df = session_visits_df.rename_axis("Age Group")

# Display the updated DataFrame
session_visits_df


# In[100]:


session_visits_df.to_csv("session_visits.csv", index=True)


# # Categorise Subluxation Scores 

# In[101]:


weighted_df


# ## Cervical

# In[102]:


C_df = weighted_df.loc[:, weighted_df.columns.str.contains(r'^Session \d+ C', regex=True)]
C_df


# In[103]:


df = pd.DataFrame(C_df)

# Identify unique session prefixes dynamically
session_groups = {}
for col in df.columns:
    session_prefix = " ".join(col.split()[:2])  # Extract "Session X" prefix
    if session_prefix.startswith("Session"):
        session_groups.setdefault(session_prefix, []).append(col)

# Compute simple sums instead of weighted products
#weighted_df = df.copy()
for session, cols in session_groups.items():
    df[session + " Cervical"] = df[cols].sum(axis=1)

# Display session-wise sum
cervical_sums = df[[col + " Cervical" for col in session_groups]]
cervical_sums.head()


# ## Lumbar

# In[104]:


L_df = weighted_df.loc[:, weighted_df.columns.str.contains(r'^Session \d+ L', regex=True)]
L_df


# In[105]:


df = pd.DataFrame(L_df)

# Identify unique session prefixes dynamically
session_groups = {}
for col in df.columns:
    session_prefix = " ".join(col.split()[:2])  # Extract "Session X" prefix
    if session_prefix.startswith("Session"):
        session_groups.setdefault(session_prefix, []).append(col)

# Compute simple sums instead of weighted products
#weighted_df = df.copy()
for session, cols in session_groups.items():
    df[session + " Lumbar"] = df[cols].sum(axis=1)

# Display session-wise sum
lumbar_sums = df[[col + " Lumbar" for col in session_groups]]
lumbar_sums.head()


# ## Thoracic

# In[106]:


T_df = weighted_df.loc[:, weighted_df.columns.str.contains(r'^Session \d+ T', regex=True)]
T_df


# In[107]:


df = pd.DataFrame(T_df)

# Identify unique session prefixes dynamically
session_groups = {}
for col in df.columns:
    session_prefix = " ".join(col.split()[:2])  # Extract "Session X" prefix
    if session_prefix.startswith("Session"):
        session_groups.setdefault(session_prefix, []).append(col)

# Compute simple sums instead of weighted products
#weighted_df = df.copy()
for session, cols in session_groups.items():
    df[session + " Thoracic"] = df[cols].sum(axis=1)

# Display session-wise sum
thoracic_sums = df[[col + " Thoracic" for col in session_groups]]
thoracic_sums.head()


# ## Pelvic

# In[108]:


P_df = weighted_df.loc[:, weighted_df.columns.str.contains(r'^Session \d+ P', regex=True)]
P_df


# In[109]:


df = pd.DataFrame(P_df)

# Identify unique session prefixes dynamically
session_groups = {}
for col in df.columns:
    session_prefix = " ".join(col.split()[:2])  # Extract "Session X" prefix
    if session_prefix.startswith("Session"):
        session_groups.setdefault(session_prefix, []).append(col)

# Compute simple sums instead of weighted products
#weighted_df = df.copy()
for session, cols in session_groups.items():
    df[session + " Pelvic"] = df[cols].sum(axis=1)

# Display session-wise sum
pelvic_sums = df[[col + " Pelvic" for col in session_groups]]
pelvic_sums.head()


# ## Merge datasets

# In[110]:


# Combine time_df and result_df along the columns
weight_time = pd.concat([time_df, session_sums, cervical_sums, lumbar_sums, thoracic_sums, pelvic_sums], axis=1)

# Sort columns numerically based on extracted session numbers
sorted_columns = sorted(weight_time.columns, key=lambda x: int(''.join(filter(str.isdigit, x))))

# Rearrange DataFrame
weight_time = weight_time[sorted_columns]

# Print the combined DataFrame
weight_time.head()


# In[111]:


# Extract relevant columns from df_combined
cols_to_extract = ["ID", "Enrolment Date", "Initial Care Plan", 'Age Group', 'Active/Passive'] 
extracted_df = df_combined[cols_to_extract]

# Combine extracted columns with combined_df
weight_subcat = pd.concat([extracted_df, weight_time], axis=1)

# Print the final combined DataFrame
weight_subcat.head()


# In[112]:


weight_subcat.to_csv("weight_subcat.csv", index=True)

