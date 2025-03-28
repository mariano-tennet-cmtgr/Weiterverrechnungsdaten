import pandas as pd
from datetime import datetime
import glob
import os
from joblib import Parallel, delayed
import streamlit as st
import logging
import warnings
from collections import Counter  #falls der Concat nicht möglich ist
import dask.dataframe as dd

# Set maximum number of cores to 12
os.environ['NUMEXPR_MAX_THREADS'] = '12'

# Ignore warnings
warnings.filterwarnings("ignore")

# Logging configurations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def prepare_data_for_variable(csv_path: str) -> dd:
    logging.info(f"Data is being prepared for this file: {os.path.basename(csv_path)}")
    # Import masterdata as csv
    tm_df = pd.read_csv(csv_path, sep=";", header=None, encoding='ANSI', decimal=",", low_memory=False)

    # Correct first row
    tm_df.iloc[0] = tm_df.iloc[0].map(
        lambda x: (x.replace('Arbeitsabhängige Sonstige Kosten [€/Viertelstunden]',
                             'Arbeitsabhängige Sonstige Kosten [€/Viertelstunde]')
                   .replace('variable Erzeugungsauslagen [€/Viertelstunde]',
                            'Variable Erzeugungsauslagen [€/Viertelstunde]')
                   .replace('Leistung (MW)', 'Leistung [MW]'))
        if isinstance(x, str) else x
    )

    # Correct second row
    tm_df.iloc[1] = tm_df.iloc[1].map(
        lambda x: (x.replace('Zuständiger RAS-ÜNB ', 'Zuständiger RAS-ÜNB')
                   .replace('RD 2.0 Ersparte Aufwendungen [€]', 'RD 2.0 Ersparte Aufwendungen [€/Viertelstunde]')
                   .replace('RD 2.0 Zusätzliche Aufwendungen [€]', 'RD 2.0 Zusätzliche Aufwendungen [€/Viertelstunde]')
                   .replace('Arbeitsabhängige Sonstige Kosten [€]', 'Summe arbeitsabhängige Sonstige Kosten [€]')
                   .replace('Anteiliger Werteverbrauch [€]', 'Summe anteiliger Werteverbrauch [€]')
                   .replace('Entgangene Erlöse Intraday [€]', 'Summe entgangene Erlöse Intraday [€]'))
        if isinstance(x, str) else x
    )

    # Convert the first row to string if not already a string
    header_row = tm_df.iloc[0].apply(lambda x: str(x) if not pd.api.types.is_integer_dtype(x) else x)

    # Remove 'nan'  and values for KTS und VWS
    headers = header_row[
        (header_row != 'nan') &
        (~header_row.str.startswith('Weiterverrechnungsschlüssel')) &
        (~header_row.str.startswith('Kostenteilungsschlüssel'))
        ]

    # Create a list of indices
    header_indices = headers.index

    # Create ranges of 100 columns for each header index
    headers_ranges = [range(idx, idx + 100) for idx in header_indices]

    # Get all column indices
    all_indices = set(range(tm_df.shape[1]))

    # Filter columns that are not within any of the ranges
    filtered_cols = [
        idx for idx in all_indices
        if not any(idx in r for r in headers_ranges) and not (pd.isna(tm_df.iloc[1, idx]))
    ]

    # Subset the dataframe with the filtered columns
    df_attributes = tm_df.iloc[:, filtered_cols]
    df_attributes.columns = df_attributes.iloc[1, :]
    df_attributes = df_attributes.drop([0, 1], axis=0).reset_index(drop=True)

    # Create new df for timeseries columns
    tm_df_variable_final = pd.DataFrame()

    # Concatenate variable ranges
    for col in header_indices:
        tm_df_variable = tm_df.iloc[:, col:col + 100]
        tm_df_variable = (tm_df_variable.drop([0, 1])
                          .reset_index(drop=True))
        tm_df_variable.columns = [f"{header_row[col]}_{x}" for x in range(1, 101)]
        tm_df_variable_final = pd.concat([tm_df_variable_final, tm_df_variable], axis=1)

    # Identify columns in tm_df_variable_final that do not start with "Art der Anweisung"
    cols_to_convert = [col for col in tm_df_variable_final.columns if not col.startswith("Art der Anweisung")]

    # Identify columns in df_attributes that contain "€" or "MWh"
    cols_to_convert.extend([
        col for col in df_attributes.columns
        if isinstance(col, str) and (("€" in col) or ("MWh" in col))
    ])

    # Concatenate df_attributes and tm_df_variable_final
    df = pd.concat([df_attributes, tm_df_variable_final], axis=1)

    # Convert those columns to numeric, using errors='coerce' to handle non-convertible values
    df[cols_to_convert] = (df[cols_to_convert].
                           replace(",", ".", regex=True).
                           apply(pd.to_numeric, errors='coerce'))

    logging.info(f"Data was prepared: {os.path.basename(csv_path)}")

    # Calculate memory per partition
    memory_per_partition = 100 * 10 ** 6  # 100 MB
    num_partitions = max(1, int(df.memory_usage().sum() / memory_per_partition))

    # Convert to Dask DataFrame with the calculated number of partitions
    ddf = dd.from_pandas(df, npartitions=num_partitions)

    # Check current partition sizes
    logging.info(f"Datafile was prepared with {ddf.partitions} partitions")

    return ddf


def process_single_file(file:str, UNB_dict:dict) -> dd:
    """
    Processes a single WV file: prepares the data and filters it based on the 'Zuständiger RAS-ÜNB'.

    Parameters:
    file (str): The file path.
    ÜNB_dict (dict): Dictionary mapping file keys to 'Zuständiger RAS-ÜNB' values.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    # Prepare the data from the file
    df_new = prepare_data_for_variable(file)

    logging.info(f"process_single_file() was started for file: {file}")

    # Determine the appropriate ÜNB value based on the filename
    for key, value in UNB_dict.items():
        if key in file:
            # Filter the DataFrame based on the 'Zuständiger RAS-ÜNB' column
            df_new = df_new.loc[df_new['Zuständiger RAS-ÜNB'] == value]
            break

    logging.info(f"WVD filtered with Zuständiger RAS-ÜNB={value}")

    return df_new


def join_weiterverrechnungen(ordner: str, n_jobs: int = -1) -> dd:
    """
    Filters Weiterverrechnungsdateien in ordner by 'Zuständiger RAS-ÜNB' and joins them into a dataframe.

    Parameters:
    ordner (str): Directory containing the WV files.
    variable (str): Name of the timeseries variable to generate a consolidated file. None for all variables.
    n_jobs (int): Number of jobs to run in parallel. Default is -1 (use all available cores).

    Returns:
    pd.DataFrame: DataFrame with joined Weiterverrechnungsdateien.
    """
    logging.info("join_weiterverrechnungen() was started")
    WV_files = glob.glob(os.path.join(ordner, f'*WV_*.csv'))

    UNB_dict = {
        "AMP": "Amprion",
        "TTG": "TenneT DE",
        "TNG": "TransnetBW",
        "50H": "50Hertz"
    }

    # Process each file in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(file, UNB_dict) for file in WV_files
    )
    logging.info("WVD files were processed")

    try:
        # Concatenate all DataFrames
        ddf = dd.concat(results, axis=0, join='outer', interleave_partitions=True)

        logging.info("WVD files were correctly concatenated")

        # Convert TM von to datetime
        ddf['TM von'] = dd.to_datetime(ddf['TM von'], format="%d.%m.%Y", errors='coerce')

        # Calculate min and max dates
        date_df = ddf["TM von"].compute()
        min_date = date_df.min().strftime("%d%m%Y")
        max_date = date_df.max().strftime('%d%m%Y')

        # Save to Excel
        file_name = f"WVD_consolidated_{min_date}_{max_date}_full.csv"

        try:
            dd.to_csv(df=ddf,
                      filename=file_name,
                      single_file=True,
                      encoding='ANSI',
                      index=False,
                      sep=";",
                      decimal=",")

            logging.info(f"{file_name} was created in the directory")

            return ddf

        except UnicodeEncodeError as e:
            logging.error(f"{file_name} could not be created due to UnicodeEncodeError:{e}. Please check!")
            st.error(f"{file_name} could not be created due to UnicodeEncodeError:{e}. Please check!")

    except pd.errors.InvalidIndexError:
        logging.info("InvalidIndexError was catched.")
        # If Error beim Concat
        all_columns = [col for df in results for col in df.columns]

        # Step 2: Count occurrences of each column name
        column_counts = Counter(all_columns)

        # Step 3: Create dataframe
        check_df = pd.DataFrame({"Variables": list(column_counts.keys()),
                                 "Counts": list(column_counts.values())
                                 })

        # Step 4: Filter problem variables:
        filtered_check_df = check_df[check_df.loc[:, "Counts"] < len(WV_files)]

        # Raise error
        logging.error(f"Error in the input data files. Some columns do not appear in every WVD file:\n "
                      f"Column(s): {filtered_check_df['Variables'].tolist()} \n"
                      f"Frequency: {filtered_check_df['Counts'].tolist()}")

        st.error(f"Error in the input data files. Some columns do not appear in every WVD file:\n "
                 f"Column(s): {filtered_check_df['Variables'].tolist()} \n"
                 f"Frequency: {filtered_check_df['Counts'].tolist()}")


def create_aggregated_timeseries(tm_df_kw_data: pd.DataFrame, variable: str, date_df: pd.DataFrame) -> dd:
    logging.info(f"create_aggregated_timeseries() was started for {variable}")
    # Ensure 'TM von' column is in datetime format
    tm_df_kw_data['TM von'] = pd.to_datetime(tm_df_kw_data['TM von'], format='%d.%m.%Y')

    # Reshape the original data to long format, extracting the quarter_hour number
    tm_df_kw_data_long = tm_df_kw_data.melt(
        id_vars=['TM von'],
        value_vars=[f"{variable}_{x}" for x in range(1, 101)],
        var_name="quarter_hour",
        value_name=variable
    )
    tm_df_kw_data_long['quarter_hour'] = tm_df_kw_data_long['quarter_hour'].str.extract('(\d+)').astype(int)

    # Sort the reshaped data by date and quarter_hour for accurate merging
    tm_df_kw_data_long = tm_df_kw_data_long.sort_values(by=['TM von', 'quarter_hour'])

    # Merge the generated timestamps with the reshaped data
    tm_df_kw_data_merged = pd.merge(
        date_df, tm_df_kw_data_long,
        left_on=['date', 'quarter_hour'],
        right_on=['TM von', 'quarter_hour'],
        how='left'
    )

    # Group by timestamp and sum the values (if multiple entries exist)
    tm_df_kw_data_merged = tm_df_kw_data_merged.groupby(['date', 'quarter_hour'], as_index=False)[variable].sum()

    # Convert MW to MWh by dividing by 4, if required
    if variable == "Leistung [MW]":
        # Convert the column to numeric and divide by 4
        tm_df_kw_data_merged[variable] = tm_df_kw_data_merged[variable] / 4

    tm_df_kw_data_merged = (date_df.set_index(['date', 'quarter_hour'])
                            .join(tm_df_kw_data_merged.set_index(['date', 'quarter_hour']), how='left'))

    # Drop columns 'date' and 'quarter_hour'
    tm_df_kw_data_merged = (tm_df_kw_data_merged.reset_index()
                            .drop(['date', 'quarter_hour'], axis=1)
                            .set_index('timestamp', drop=True))

    # Calculate memory per partition
    memory_per_partition = 100 * 10 ** 6  # 100 MB
    num_partitions = max(1, int(tm_df_kw_data_merged.memory_usage().sum() / memory_per_partition))

    # Convert to Dask DataFrame with the calculated number of partitions
    tm_ddf_kw_data_merged = dd.from_pandas(tm_df_kw_data_merged, npartitions=num_partitions)

    # Check current partition sizes
    logging.info(f"Datafile for {variable} was finished with {tm_ddf_kw_data_merged.partitions} partitions")

    return tm_ddf_kw_data_merged


def apply_aggregation(group_df: pd.DataFrame, variable: str, group_keys: tuple, date_df: pd.DataFrame) -> dd:
    """
    Apply the create_aggregated_timeseries function to a single group.

    Parameters:
    group_df (pd.DataFrame): The subset DataFrame for the current group.
    variable (str): The variable of interest for aggregation.
    group_keys (tuple): The tuple containing the current group keys.

    Returns:
    pd.DataFrame: Aggregated DataFrame for the given group.
    """
    logging.info("apply_aggregation() was started")

    # Get the keys, which will then be used in the column name
    ras_unb, ao, ao_code, ao_art, tm_richtung = group_keys

    # Apply the create_aggregated_timeseries with group_df
    result = create_aggregated_timeseries(group_df, variable, date_df)

    # If the resulting df is not empty, then the column name is written
    if len(result.index) > 0:
        result = result.rename(columns={variable: f'{ras_unb}|{ao}|{ao_code}|{ao_art}|{tm_richtung}'})

    # Otherwise the following will be printed
    else:
        logging.info(f"No data found for combination: {ras_unb}, {ao}, {ao_code}, {ao_art}, {tm_richtung}")

    return result


def aggregate_all_combinations(kw_leistung_df: dd, variable: str, n_jobs: int = -1) -> pd.DataFrame:
    """
    Apply the aggregation function to every unique combination of 'Zuständiger RAS-ÜNB', 'Aktivierungsobjekt', 'AO-Code'
    and 'AO-Art'.

    Parameters:
    kw_leistung_df (dd): Original DataFrame containing all relevant columns.

    Returns:
    dict: Dictionary of DataFrames with keys as tuples of the form (ras_unb, ao, ao_code, ao_art).
    """
    # Define the custom order for 'Zuständiger RAS-ÜNB'
    logging.info("aggregate_all_combinations() was started")

    # Get the ÜNB for custom order
    custom_order = ['TenneT DE', 'Amprion', '50Hertz', 'TransnetBW']
    kw_leistung_df['Zuständiger RAS-ÜNB'] = pd.Categorical(
        kw_leistung_df['Zuständiger RAS-ÜNB'],
        categories=custom_order,
        ordered=True
    )

    # Sort the DataFrame by 'Zuständiger RAS-ÜNB' and 'Aktivierungsobjekt'
    kw_leistung_df = kw_leistung_df.sort_values(by=['Zuständiger RAS-ÜNB', 'Aktivierungsobjekt'])

    # Ensure 'TM von' column is in datetime format
    kw_leistung_df['TM von'] = pd.to_datetime(kw_leistung_df['TM von'], format="%Y-%m-%d")

    # Define the complete date range based on the data
    start_date = kw_leistung_df['TM von'].min().normalize()
    end_date = kw_leistung_df['TM von'].max().normalize() + pd.Timedelta(hours=23, minutes=45)

    # Reindex to the complete date range
    complete_date_range = pd.date_range(start=start_date, end=end_date,
                                        freq='15min', tz='Europe/Berlin')

    # Create date_df
    date_df = pd.DataFrame({'timestamp': complete_date_range.tz_localize(None)})
    date_df['date'] = pd.to_datetime(date_df['timestamp'].dt.date, format='%Y-%m-%d')
    date_df['quarter_hour'] = date_df.groupby('date').cumcount() + 1

    # Group by the relevant columns
    grouped = kw_leistung_df.groupby(['Zuständiger RAS-ÜNB', 'Aktivierungsobjekt', 'AO-Code', 'AO-Art', 'TM-Richtung'], observed=False)

    # Replace variable name if Arbeit [MWh] is chosen
    # Apply parallel processing using joblib
    logging.info("Aggregations are being constructed")

    if variable == "Arbeit [MWh]":
        results = Parallel(n_jobs=n_jobs)(
            delayed(apply_aggregation)(group_df, "Leistung [MW]", group_keys, date_df)
            for group_keys, group_df in grouped
        )

    else:
        results = Parallel(n_jobs=n_jobs)(
                delayed(apply_aggregation)(group_df, variable, group_keys, date_df)
                for group_keys, group_df in grouped
            )

    logging.info("Aggregations were carried out correctly")

    # Filter out empty dask.Dataframes and concatenate the results
    combined_results = [dfs for dfs in results if not len(dfs.index) == 0]

    if combined_results:
        # Concatenate date_df with the combined_results DataFrame
        final_ddf = dd.concat(combined_results, axis=1, join='outer', interleave_partitions=True)

        logging.info("Data files were appended correctly")

        # Calculate min and max dates
        min_date = start_date.strftime("%d%m%Y")
        max_date = end_date.strftime('%d%m%Y')

        # Create csv file_name
        variable_name = variable.split(" [")[0]
        time_series_filename = f'aggregated_timeseries_combined_{variable_name}_{min_date}_{max_date}.csv'

        # Create csv file
        dd.to_csv(df=final_ddf, filename=time_series_filename,
                  single_file=True,
                  encoding='ANSI',
                  index=True,
                  sep=";",
                  decimal=",")

        logging.info(f"Timeseries csv file has been created for {variable_name} \n Check current working directory {os.getcwd()}")

        return final_ddf

    else:
        logging.info("combined_results was empty")


def main():
    st.title("Weiterverrechnungsdateien")

    # Sidebar header
    st.sidebar.header("Select parameters")

    # Radio button to ask if a consolidated file already exists
    file_exists = st.sidebar.radio("Does a consolidated file already exist?", ("Nein", "Ja"))

    if file_exists == "Nein":
        ordner = st.sidebar.text_input('Enter the folder path containing the WVD of the 4ÜNB:',
                                       r"C:\Users\{KID}\Downloads\{Beispiel_Ordner}")

    elif file_exists == "Ja":
        existing_file_path = st.sidebar.file_uploader("Enter the path to the existing consolidated file:", "csv")

    variable_list = st.sidebar.multiselect(
        'Select the variables of interest for creating timeseries files',
        ["Arbeit [MWh]", "Variable Erzeugungsauslagen [€/Viertelstunde]", "Arbeitsabhängige Sonstige Kosten [€/Viertelstunde]",
         "Anteiliger Werteverbrauch [€/Viertelstunde]", "Entgangene Erlöse Intraday [€/Viertelstunde]",
         "RD 2.0 Ausfallleistung [MW]", "RD 2.0 Entschädigung Entgangene Einnahmen EEG und KWK [€/Viertelstunde]",
         "RD 2.0 Zusätzliche Aufwendungen [€/Viertelstunde]", "RD 2.0 Ersparte Aufwendungen [€/Viertelstunde]",
         "RD 2.0 Abweichung zwischen dem bilanziellen Ausgleich und der Ausfallarbeit [€/Viertelstunde]",
         "Abweichungen zu dem vom anfNB als BK-Fahrplan gelieferten energetischen Ausgleich im Clusterfall [€/Viertelstunde]",
        "RD 2.0 Entschädigung Energetischer Ausgleich [€/Viertelstunde]"]
    )


    if st.sidebar.button('Process Data'):
        if file_exists == "Ja" and existing_file_path:
            logging.info("Loading existing consolidated file")
            st.write(
                f"{datetime.now()} ----- Loading existing consolidated file (see console for log information):")

            consolidated_WVD = pd.read_csv(existing_file_path,
                                           sep=";",
                                           encoding='ANSI',
                                           decimal=",",
                                           header=0,
                                           low_memory=False,
                                           thousands=".")

            logging.info("Existing consolidated file has been loaded")
            st.dataframe(consolidated_WVD.head(15))

        elif file_exists == "Nein" and ordner:
            logging.info("Creating consolidated file")
            st.write(f"{datetime.now()} ----- Creating consolidated file (see console for log information):")
            consolidated_WVD = join_weiterverrechnungen(ordner)

            logging.info("Consolidated file has been created")
            st.write(f"{datetime.now()} ----- Consolidated file has been created. Ablagepfad: \n {os.getcwd()}")
            #st.dataframe(consolidated_WVD.compute().head(16))

        # Process time series if a variable is selected
        if variable_list:
            logging.info(f"Creating timeseries file for the variable(s) chosen")
            st.write(
                f"{datetime.now()} ----- Creating timeseries file for the variable(s) chosen (see console for log information)")

            if type(consolidated_WVD) != pd.DataFrame:
                consolidated_WVD = consolidated_WVD.compute()
                logging.info(
                    f"Consolidated file has been transformed from dask.dataframe to {type(consolidated_WVD)}")
            try:
                Parallel(n_jobs=-1)(
                    delayed(aggregate_all_combinations)(consolidated_WVD, variable) for variable in variable_list
                )
            except ValueError as ve:
                st.write(f"{datetime.now()} ----- Please check log information. Exception has been raised: {ve}")

            logging.info("Timeseries files were successfully created!")
            st.write(f"{datetime.now()} ----- Timeseries file has been created. Ablagepfad: \n {os.getcwd()}")


if __name__ == "__main__":
    main()
