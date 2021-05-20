# here will contain all functions to read every file from data folder
import numpy as np
import glob
import os
import pandas as pd
def read_indv_data(PROJ_ROOT,data_type="raw"):
    """read_data Read data from respective folder specified by data_type

    Just input project root, data_type. Data will be cleaned, read, returned in list of series format.

    Args:
        PROJ_ROOT (str): The root of the project
        data_type (str, optional): Type of data 'raw', 'interim', 'processed', .... Defaults to "raw".

    Returns:
        list: dataframe of data in folder
    """
    data_path = os.path.join(PROJ_ROOT,
                            "data",
                            "data_type")
    site_data_files = glob.glob(data_path+"/*.csv")
    site_data_series = []
    for file in site_data_files:
        tmp_series = pd.read_csv(file, index_col=['site_id','time'], parse_dates=True)
        if tmp_series.index.get_level_values(0)[0] not in [48,49,16,15]:
            tmp_series = tmp_series[['PM25']]
    #         Replace 0 with median
            tmp_series['PM25'] = tmp_series['PM25'].replace(0, np.nanmedian(tmp_series['PM25'].values))
    #         Replace null with median
            tmp_series['PM25'] = tmp_series['PM25'].fillna(value=np.nanmedian(tmp_series['PM25'].values))
            site_data_series.append(tmp_series)
    return site_data_series