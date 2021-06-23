import pandas as pd
import numpy as np

# Gia tri so sanh de tinh muc do AQI
def categorize_AQI(AQI_data):
    """
    Input: Series of AQI_values
    Output: Series of AQI category
    7 categories [Good, Moderate, Unhealthy for Sensitive, Unhealthy, Very Unhealthy, Hazardous, Out of AQI]
    range of categories [0-50, 51-100, 101-150, 151-200, 201-300, 301-500, >500]
    """
    bins = [-1, 50, 100, 150, 200, 300, 500, np.inf]
    labels = ["Good", "Moderate", "Unhealthy for Sensitive",
              "Unhealthy", "Very Unhealthy", "Hazardous", "Beyond AQI"]
    return pd.cut(AQI_data, bins=bins, labels=labels)


def calculate_AQI_h(data=None):
    def create_BP_df():
        import pandas as pd
        BP_Ii = [0, 50, 100, 150, 200, 300, 400, 500]
        BP_O3_1h = [0, 160, 200, 300, 400, 800, 1000, 1200]
        BP_O3_8h = [0, 120, 170, 210, 400]
        BP_CO = [0, 10000, 30000, 45000, 60000, 90000, 120000, 150000]
        BP_SO2 = [0, 125, 350, 550, 800, 1600, 2100, 2630]
        BP_NO2 = [0, 100, 200, 700, 1200, 2350, 3100, 3850]
        BP_PM10 = [0, 50, 150, 250, 350, 420, 500, 600]
        BP_PM25 = [0, 25, 50, 80, 150, 250, 350, 500]
        BP = pd.DataFrame(data=[range(1, 9), BP_Ii, BP_O3_1h, BP_O3_8h, BP_CO, BP_SO2, BP_NO2, BP_PM10, BP_PM25],
                          index=['I', 'Ii', 'O3', 'O3_8h', 'CO', 'SO2', 'NO2', 'PM10', 'PM25'], dtype=np.int64)
        BP = BP.transpose().set_index('I')
        # Add a upper to calculate I = 8
        BP.loc[9] = [999999, 999999, 999999,
                     999999, 999999, 999999, 999999, 999999]
        BP.index = BP.index.astype(np.int64)
        return BP

    def calculate_BP_I(polutant_data, BP=create_BP_df()):
        import pandas as pd
        """
        For: Calculating I for all polutant values. I then used to calculate AQI
        Input:
            + polutant_data: dataframe values of all polutant
            + BP: BP table
        Ouput:
            + DataFrame with I of all pollutants [CO, NO2, O3, PM25, SO2]
        """
        result_I = []
        BP_polutant_column = BP.loc[:, polutant_data.name]
        result_I = polutant_data.apply(
            lambda x: BP_polutant_column.ge(x).idxmax() - 1)
        return result_I

    # Cong thuc 1 = ((Ii[i+1] - Ii[i])/ (BP[i+1] - BP[i]))* (C[h] - BP[i]) +Ii[i]
    # Cong thuc PM25 = ((Ii[i+1] - Ii[i])/ (BP[i+1] - BP[i]))* (nowcast[h] - BP[i]) +Ii[i]
    def calculate_AQI_formular(polutant_data, I, nowcast, BP=create_BP_df()):
        result = []
        for data in zip(polutant_data.index, polutant_data.values, I.values):
            polutant_name, polutant_value, I = data
            I = int(I)
            try:
                if polutant_name == 'PM25':
                    if nowcast == 0:
                        AQI = 0
                    else:
                        AQI = ((BP.loc[I + 1, 'Ii'] - BP.loc[I, 'Ii']) /
                               (BP.loc[I + 1, polutant_name] - BP.loc[I, polutant_name]) *
                               (nowcast - BP.loc[I, polutant_name])) + BP.loc[I, 'Ii']
                        AQI = AQI.round(0)
                else:
                    AQI = ((BP.loc[I + 1, 'Ii'] - BP.loc[I, 'Ii']) /
                           (BP.loc[I + 1, polutant_name] - BP.loc[I, polutant_name]) *
                           (polutant_value - BP.loc[I, polutant_name])) + BP.loc[I, 'Ii']
                    AQI = AQI.round(0)
            except KeyError:
                if I == 0:
                    AQI = 0
                else:
                    AQI = 500
            result.append(AQI)
        return result

    # Cong thuc tinh nowcast danh rieng cho PM25
    def nowcast(polutant_data, calculating_hour_value):
        time = calculating_hour_value.name
        # Loc 12h gan nhat
        tmp_12h_data_storage = polutant_data.loc[
            idx[time - pd.Timedelta(hours=12): time]]
        # Dao nguoc index lai
        tmp_12h_data_storage = tmp_12h_data_storage.iloc[::-1]
        # Kiem tra xem trong 3 gio gan nhat thi it nhat 2 gio phai co so lieu
        # Neu khong qua duoc check o tren thi so lieu tinh duoc dat la 0
        if tmp_12h_data_storage.iloc[0:3].isna().sum().values[0] > 1:
            calculating_hour_value.loc[:] = 0
        elif len(tmp_12h_data_storage) < 2:
            calculating_hour_value.loc[:] = 0
        else:
            min_value = tmp_12h_data_storage.min().values[0]
            max_value = tmp_12h_data_storage.max().values[0]
            try:
                w = min_value / max_value
            except ZeroDivisionError:
                print(time)
            if w < 1/2:
                w = 1/2
            # Tinh gia tri nowcast dua theo cong thuc
            if w == 1/2:
                nowcast = sum(
                    w**i * tmp_12h_data_storage.iloc[i-1] for i in range(1, len(tmp_12h_data_storage)+1))
            else:
                nowcast = sum([(w**(i-1)) * tmp_12h_data_storage.iloc[i-1] for i in range(1, len(
                    tmp_12h_data_storage)+1)]) / sum([(w**(i-1)) for i in range(1, len(tmp_12h_data_storage) + 1)])
            calculating_hour_value = nowcast.values[0]
        return calculating_hour_value

# ==========================================================================================================
    idx = pd.IndexSlice
    _global_BP = create_BP_df()
    sorted_data = data.sort_index(level=['site_id', 'time'], ascending=[1, 1])
    site_ids = list(data.index.get_level_values(0).unique())
#     set_trace()
    for site_id in site_ids:
        try:
            tmp = sorted_data.loc[site_id, ['PM25']].apply(
                (lambda x: nowcast(sorted_data.loc[site_id, ['PM25']], x)), axis=1,
                result_type='broadcast'
            )
        except Exception as e:
            print("Error in site {}".format(site_id))
            print(e)
#         set_trace()
        sorted_data.loc[site_id, ['NowCast']] = tmp.values

    # Air index number of all polutant data
    I_number = sorted_data.drop('NowCast', axis=1).apply(
        (lambda x: calculate_BP_I(x, _global_BP)), axis=0, result_type='broadcast')
    calculated_AQI = I_number.apply((lambda I: calculate_AQI_formular(sorted_data.loc[I.name, :], I, sorted_data.loc[I.name, 'NowCast'], _global_BP)),
                                    axis=1, result_type='broadcast')
    calculated_AQI['AQI_h'] = calculated_AQI.max(axis=1)
    calculated_AQI['AQI_h_Polutant'] = calculated_AQI.idxmax(axis=1)
    calculated_AQI['AQI_h_label'] = categorize_AQI(calculated_AQI['AQI_h'])
    calculated_AQI['AQI_h_I'] = calculated_AQI['AQI_h_label'].cat.codes + 1
    # Rearranging columns
    calculated_AQI = calculated_AQI[[
        'CO', 'NO2', 'PM25', 'AQI_h', 'AQI_h_Polutant', 'AQI_h_I', 'AQI_h_label']]
    return calculated_AQI
