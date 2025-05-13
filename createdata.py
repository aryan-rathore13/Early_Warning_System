import pandas as pd
import os

# --- Configuration ---
BASE_PATH = r"D:\Early Warning System" # Use raw string for paths
INPUT_FILE_PATH = os.path.join(BASE_PATH, "input.xlsx")
SONITPUR_DATA_PATH = os.path.join(BASE_PATH, "SONITPUR_GABHARUstudent_data.csv")
CACHAR_DATA_PATH = os.path.join(BASE_PATH, "CACHAR_SONAI_student_data.csv")

OUTPUT_FILE_PATH = os.path.join(BASE_PATH, "main_input_processed.xlsx") # Output file
SONITPUR_LOG_PATH = os.path.join(BASE_PATH, "sonitpur_missing_records_log.txt")
CACHAR_LOG_PATH = os.path.join(BASE_PATH, "cachar_missing_records_log.txt")
DROPPED_INPUT_ROWS_LOG_PATH = os.path.join(BASE_PATH, "dropped_input_rows_log.txt")


# --- Column Names Configuration ---

# Column Names in Input File (input.xlsx)
UNIQUE_STUDENT_ID_COL_INPUT = 'unique_student_identifier'
UUID_COL_INPUT = 'uuid'
DISTRICT_NAME_COL_INPUT = 'district_name'
CLASS_COL_INPUT = 'Class 2023-24' # Column for class information
IS_DROPOUT_COL_INPUT = 'is_drop_out' # Column for dropout status
NEW_ATTENDANCE_COL_NAME = 'Attendance % in 2024-25'

# Column Names in District CSV Files (Sonitpur & Cachar CSVs)
UNIQUE_STUDENT_ID_COL_DISTRICT = 'unique_student_identifier'
UUID_COL_DISTRICT = 'uuid'
ATTENDANCE_COL_DISTRICT = 'attendance_percentage' # << UPDATED as per your request

# District name values (will be upper-cased for matching)
SONITPUR_DISTRICT_NAME_VAL = 'SONITPUR'
CACHAR_DISTRICT_NAME_VAL = 'CACHAR'

# Specific values for conditional logic (after standardization)
CLASS_XII_VALUE = 'Class-XII' # Exact match after stripping whitespace
PROMOTED_VALUE_LOWER = 'promoted' # Compared after converting input column to lowercase
DROPOUT_VALUE_LOWER = 'dropout'   # Compared after converting input column to lowercase


def process_student_data():
    """
    Processes student data to add new attendance column, handling specific dropout/promoted cases
    and dropping rows with missing essential identifiers from the input.
    """
    try:
        # --- 1. Load Data ---
        print(f"Loading input file: {INPUT_FILE_PATH}...")
        df_input = pd.read_excel(INPUT_FILE_PATH)
        print(f"Loaded input data with {len(df_input):,} rows and {len(df_input.columns)} columns.")

        print(f"Loading Sonitpur data: {SONITPUR_DATA_PATH}...")
        df_sonitpur = pd.read_csv(SONITPUR_DATA_PATH, low_memory=False)
        print(f"Loaded Sonitpur data with {len(df_sonitpur):,} rows and {len(df_sonitpur.columns)} columns.")

        print(f"Loading Cachar data: {CACHAR_DATA_PATH}...")
        df_cachar = pd.read_csv(CACHAR_DATA_PATH, low_memory=False)
        print(f"Loaded Cachar data with {len(df_cachar):,} rows and {len(df_cachar.columns)} columns.")

        # --- 2. Validate Essential Columns ---
        required_input_cols = [
            UNIQUE_STUDENT_ID_COL_INPUT, UUID_COL_INPUT, DISTRICT_NAME_COL_INPUT,
            CLASS_COL_INPUT, IS_DROPOUT_COL_INPUT
        ]
        for col in required_input_cols:
            if col not in df_input.columns:
                print(f"ERROR: Required column '{col}' not found in input file: {INPUT_FILE_PATH}")
                return

        required_district_cols = [UNIQUE_STUDENT_ID_COL_DISTRICT, UUID_COL_DISTRICT, ATTENDANCE_COL_DISTRICT]
        for col in required_district_cols:
            if col not in df_sonitpur.columns:
                print(f"ERROR: Required column '{col}' not found in Sonitpur data: {SONITPUR_DATA_PATH}")
                return
            if col not in df_cachar.columns:
                print(f"ERROR: Required column '{col}' not found in Cachar data: {CACHAR_DATA_PATH}")
                return

        # --- 3. Pre-processing and Standardization of Input Data ---
        # Add original index column BEFORE any filtering for accurate logging
        df_input['original_input_excel_row_index'] = df_input.index

        print("Standardizing relevant columns in input data (IDs, district, class, dropout status)...")
        # Standardize ID columns
        df_input[UNIQUE_STUDENT_ID_COL_INPUT] = df_input[UNIQUE_STUDENT_ID_COL_INPUT].fillna('').astype(str).str.strip()
        df_input[UUID_COL_INPUT] = df_input[UUID_COL_INPUT].fillna('').astype(str).str.strip()
        
        # Standardize other relevant columns
        df_input[DISTRICT_NAME_COL_INPUT] = df_input[DISTRICT_NAME_COL_INPUT].astype(str).str.strip().str.upper()
        df_input[CLASS_COL_INPUT] = df_input[CLASS_COL_INPUT].astype(str).str.strip() # Keep case for 'Class-XII'
        df_input[IS_DROPOUT_COL_INPUT] = df_input[IS_DROPOUT_COL_INPUT].astype(str).str.strip().str.lower() # To lowercase for 'promoted'/'dropout'

        # Ensure district name values used for comparison are also uppercase
        global SONITPUR_DISTRICT_NAME_VAL, CACHAR_DISTRICT_NAME_VAL
        SONITPUR_DISTRICT_NAME_VAL = SONITPUR_DISTRICT_NAME_VAL.upper()
        CACHAR_DISTRICT_NAME_VAL = CACHAR_DISTRICT_NAME_VAL.upper()

        # --- 4. Drop Rows from INPUT where both main identifiers are missing ---
        original_row_count_input = len(df_input)
        # Condition for dropping: USI is empty AND UUID is empty (columns are already standardized)
        rows_to_drop_mask = (df_input[UNIQUE_STUDENT_ID_COL_INPUT] == '') & \
                            (df_input[UUID_COL_INPUT] == '')
        
        num_rows_to_drop = rows_to_drop_mask.sum()
        dropped_rows_info = []

        if num_rows_to_drop > 0:
            print(f"Found {num_rows_to_drop:,} rows in the input file where both '{UNIQUE_STUDENT_ID_COL_INPUT}' and '{UUID_COL_INPUT}' are missing/empty.")
            
            # Collect info about dropped rows for logging
            if num_rows_to_drop > 0: # Check again due to potential mask re-evaluation in some pandas versions if df is very small
                df_to_drop_info = df_input[rows_to_drop_mask][[
                    'original_input_excel_row_index',
                    UNIQUE_STUDENT_ID_COL_INPUT,
                    UUID_COL_INPUT
                ]]
                for _,_row_info in df_to_drop_info.iterrows():
                    dropped_rows_info.append(
                        f"Original Index: {_row_info['original_input_excel_row_index']}, "
                        f"{UNIQUE_STUDENT_ID_COL_INPUT}: '{_row_info[UNIQUE_STUDENT_ID_COL_INPUT]}', "
                        f"{UUID_COL_INPUT}: '{_row_info[UUID_COL_INPUT]}'\n"
                    )

            df_input = df_input[~rows_to_drop_mask].copy() # Keep rows where at least one ID is present
            print(f"Dropped these rows. Remaining rows in input data for processing: {len(df_input):,}")
            with open(DROPPED_INPUT_ROWS_LOG_PATH, 'w') as f_log_dropped:
                f_log_dropped.write(f"Log of rows dropped from '{INPUT_FILE_PATH}' because both "
                                    f"'{UNIQUE_STUDENT_ID_COL_INPUT}' and '{UUID_COL_INPUT}' were missing/empty:\n\n")
                f_log_dropped.writelines(dropped_rows_info)
                print(f"  Details of dropped rows logged to: {DROPPED_INPUT_ROWS_LOG_PATH}")

        else:
            print(f"No rows found in the input file where both '{UNIQUE_STUDENT_ID_COL_INPUT}' and '{UUID_COL_INPUT}' are simultaneously missing/empty.")


        # --- 5. Standardize ID columns in District Data ---
        print("Standardizing ID column data types in district CSVs...")
        for df, id_cols_list in [
            (df_sonitpur, [UNIQUE_STUDENT_ID_COL_DISTRICT, UUID_COL_DISTRICT]),
            (df_cachar, [UNIQUE_STUDENT_ID_COL_DISTRICT, UUID_COL_DISTRICT])
        ]:
            for col in id_cols_list:
                df[col] = df[col].fillna('').astype(str).str.strip()
        
        # --- 6. Prepare Lookup Dictionaries for District Data ---
        print("Creating lookup dictionaries for district data...")
        sonitpur_usi_map = df_sonitpur.drop_duplicates(
            subset=[UNIQUE_STUDENT_ID_COL_DISTRICT], keep='first'
        ).set_index(UNIQUE_STUDENT_ID_COL_DISTRICT)[ATTENDANCE_COL_DISTRICT].to_dict()
        sonitpur_uuid_map = df_sonitpur.drop_duplicates(
            subset=[UUID_COL_DISTRICT], keep='first'
        ).set_index(UUID_COL_DISTRICT)[ATTENDANCE_COL_DISTRICT].to_dict()

        cachar_usi_map = df_cachar.drop_duplicates(
            subset=[UNIQUE_STUDENT_ID_COL_DISTRICT], keep='first'
        ).set_index(UNIQUE_STUDENT_ID_COL_DISTRICT)[ATTENDANCE_COL_DISTRICT].to_dict()
        cachar_uuid_map = df_cachar.drop_duplicates(
            subset=[UUID_COL_DISTRICT], keep='first'
        ).set_index(UUID_COL_DISTRICT)[ATTENDANCE_COL_DISTRICT].to_dict()
        print("Lookup dictionaries created.")

        # --- 7. Initialize New Column and Log Lists ---
        df_input[NEW_ATTENDANCE_COL_NAME] = pd.NA
        sonitpur_missing_log_entries = []
        cachar_missing_log_entries = []
        
        processed_count = 0
        found_count = 0
        skipped_logging_count = 0

        print(f"Processing {len(df_input):,} rows from the (potentially filtered) input file to find attendance data...")
        # --- 8. Iterate, Match, and Log (with new conditions) ---
        for current_df_idx, row in df_input.iterrows(): # current_df_idx is the index in the current (filtered) df_input
            original_input_excel_idx = row['original_input_excel_row_index'] # For consistent logging

            current_unique_id = row[UNIQUE_STUDENT_ID_COL_INPUT]
            current_uuid = row[UUID_COL_INPUT]
            current_district = row[DISTRICT_NAME_COL_INPUT]

            attendance_value = pd.NA

            if current_district == SONITPUR_DISTRICT_NAME_VAL:
                attendance_value = sonitpur_usi_map.get(current_unique_id)
                if pd.isna(attendance_value) and current_uuid: # Only try UUID if it's not empty
                    attendance_value = sonitpur_uuid_map.get(current_uuid)
            
            elif current_district == CACHAR_DISTRICT_NAME_VAL:
                attendance_value = cachar_usi_map.get(current_unique_id)
                if pd.isna(attendance_value) and current_uuid: # Only try UUID if it's not empty
                    attendance_value = cachar_uuid_map.get(current_uuid)

            if pd.isna(attendance_value): # Attendance not found in district files
                student_class = row[CLASS_COL_INPUT]
                student_dropout_status = row[IS_DROPOUT_COL_INPUT] # This is already lowercased

                is_exempt_from_missing_log = False
                # Condition 1: Class XII and Promoted
                if student_class == CLASS_XII_VALUE and student_dropout_status == PROMOTED_VALUE_LOWER:
                    is_exempt_from_missing_log = True
                # Condition 2: Dropout
                elif student_dropout_status == DROPOUT_VALUE_LOWER:
                    is_exempt_from_missing_log = True
                
                if is_exempt_from_missing_log:
                    skipped_logging_count +=1
                    # Optionally, you could set a specific value for NEW_ATTENDANCE_COL_NAME here if needed
                    # e.g., df_input.loc[current_df_idx, NEW_ATTENDANCE_COL_NAME] = "Graduated/Dropout"
                    # For now, it remains pd.NA as attendance is genuinely not applicable for 2024-25
                else:
                    # Not exempt, so log as missing
                    log_message = (
                        f"Original Input Excel Row Index: {original_input_excel_idx}, "
                        f"{UNIQUE_STUDENT_ID_COL_INPUT}: '{current_unique_id}', "
                        f"{UUID_COL_INPUT}: '{current_uuid}'. "
                        f"Details - Input Class: '{student_class}', Input Dropout Status: '{student_dropout_status}'.\n"
                    )
                    if current_district == SONITPUR_DISTRICT_NAME_VAL:
                        sonitpur_missing_log_entries.append(log_message)
                    elif current_district == CACHAR_DISTRICT_NAME_VAL:
                        cachar_missing_log_entries.append(log_message)
            else: # Attendance was found
                df_input.loc[current_df_idx, NEW_ATTENDANCE_COL_NAME] = attendance_value
                found_count += 1
            
            processed_count += 1
            if processed_count % 5000 == 0:
                print(f"  Processed {processed_count:,}/{len(df_input):,} rows. "
                      f"Found {found_count:,} attendance values. "
                      f"Skipped logging for {skipped_logging_count:,} (Graduated/Dropout).")
        
        print(f"Finished processing rows. Total processed: {processed_count:,}. "
              f"Total attendance values populated: {found_count:,}. "
              f"Records not logged due to Graduated/Dropout status: {skipped_logging_count:,}.")

        # --- 9. Write Log Files ---
        # (Sonitpur and Cachar logs will now only contain genuinely missing, non-exempted records)
        print(f"Writing Sonitpur missing records log to: {SONITPUR_LOG_PATH}")
        with open(SONITPUR_LOG_PATH, 'w') as f:
            if sonitpur_missing_log_entries:
                f.write(
                    f"Records from input file (district: {SONITPUR_DISTRICT_NAME_VAL}) "
                    f"not found in Sonitpur data file ({SONITPUR_DATA_PATH}) "
                    f"using either '{UNIQUE_STUDENT_ID_COL_DISTRICT}' or '{UUID_COL_DISTRICT}', "
                    f"and not exempted by Class-XII/Promoted or Dropout status:\n\n"
                )
                f.writelines(sonitpur_missing_log_entries)
                print(f"  Logged {len(sonitpur_missing_log_entries):,} missing records for Sonitpur.")
            else:
                f.write(f"No missing records (requiring logging) found for {SONITPUR_DISTRICT_NAME_VAL} district.\n")
                print(f"  No missing records (requiring logging) for Sonitpur.")
        
        print(f"Writing Cachar missing records log to: {CACHAR_LOG_PATH}")
        with open(CACHAR_LOG_PATH, 'w') as f:
            if cachar_missing_log_entries:
                f.write(
                    f"Records from input file (district: {CACHAR_DISTRICT_NAME_VAL}) "
                    f"not found in Cachar data file ({CACHAR_DATA_PATH}) "
                    f"using either '{UNIQUE_STUDENT_ID_COL_DISTRICT}' or '{UUID_COL_DISTRICT}', "
                    f"and not exempted by Class-XII/Promoted or Dropout status:\n\n"
                )
                f.writelines(cachar_missing_log_entries)
                print(f"  Logged {len(cachar_missing_log_entries):,} missing records for Cachar.")
            else:
                f.write(f"No missing records (requiring logging) found for {CACHAR_DISTRICT_NAME_VAL} district.\n")
                print(f"  No missing records (requiring logging) for Cachar.")

        # --- 10. Save Output ---
        # Drop the temporary original index column before saving
        if 'original_input_excel_row_index' in df_input.columns:
            df_input_to_save = df_input.drop(columns=['original_input_excel_row_index'])
        else:
            df_input_to_save = df_input

        print(f"Saving updated data to: {OUTPUT_FILE_PATH}...")
        df_input_to_save.to_excel(OUTPUT_FILE_PATH, index=False)
        print(f"Processing complete! Output file saved to '{OUTPUT_FILE_PATH}'.")
        print(f"Log files saved to '{SONITPUR_LOG_PATH}', '{CACHAR_LOG_PATH}', and '{DROPPED_INPUT_ROWS_LOG_PATH}'.")

    except FileNotFoundError as e:
        print(f"ERROR: File not found. Please check file paths. Details: {e}")
    except KeyError as e:
        print(f"ERROR: A column was not found in one of the DataFrames. "
              f"This usually means a mismatch in configured column names vs actual column names in files. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    process_student_data()