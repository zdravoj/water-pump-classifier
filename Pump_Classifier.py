import pandas as pd
from joblib import load
from json import dumps
from numpy import log10

def predict_pump_status(json_input):
    # load encoder and model
    encoder = load('pump_classifier_transformer.joblib')
    wp_classifier = load('pump_classifier_model.joblib')
    # convert input to DataFrame (for cleaning function)
    input_df = pd.read_json(json_input, orient='index', convert_dates=False)
    # clean data (for model format)
    cleaned_input_df = clean_pump_data(input_df)
    # encode data
    encoded_input = encoder.transform(cleaned_input_df)
    # generate prediction
    input_prediction = wp_classifier.predict(encoded_input)[0]
    # prepare prediction for JSON conversion
    wp_predict = {"output": input_prediction}
    # convert prediction to JSON
    well_output = dumps(wp_predict, indent=2)
    # return prediction
    return well_output


def custom_pump_impute(dataframe):
    # define values to impute in nulls
    impute_pairs = {
        'amount_tsh': 1.0,
        'gps_height': 0,
        'basin': 'Lake Victoria',
        'population': 0,
        'public_meeting': 'unknown',
        'extraction_type_class': 'gravity',
        'management_group': 'user-group',
        'payment_type': 'never pay',
        'water_quality': 'soft',
        'quantity_group': 'enough',
        'source_type': 'spring',
        'waterpoint_type': 'communal standpipe',
        'date_recorded': '1850-01-01',
        'construction_year': 0
    }
    # impute null values
    for k, v in impute_pairs.items():
        dataframe[k] = dataframe[k].fillna(v)
    # include 'None' in imputed values for scheme_management
    dataframe.scheme_management = dataframe.scheme_management.apply(
        lambda x: 'unknown' if (pd.isna(x) or x == 'None') else x
    )
    return dataframe


def clean_pump_data(pump_dataframe):
    # drop columns
    pump_dataframe = pump_dataframe.drop(columns=[
    'num_private',
    'subvillage',
    'lga',
    'ward',
    'region',
    'region_code',
    'district_code',
    'scheme_name',
    'extraction_type',
    'extraction_type_group',
    'management',
    'payment',
    'quality_group',
    'quantity',
    'source',
    'source_class',
    'waterpoint_type_group',
    'funder',
    'installer',
    'wpt_name',
    'recorded_by',
    'permit'
    ])
    # apply custom impute function
    pump_dataframe = custom_pump_impute(pump_dataframe)
    # extract year from 'date_recorded'
    year_recorded = pump_dataframe.date_recorded.apply(lambda x: int(x[:4]))
    # calculate pump age
    pump_age = year_recorded - pump_dataframe.construction_year
    # impute inaccurate values
    pump_age = pump_age.apply(lambda x: -100 if (x < 0 or x > 100) else x)
    # assign pump_age to feature matrix
    pump_dataframe['pump_age'] = pump_age
    # drop date_recorded and construction_year features
    pump_dataframe = pump_dataframe.drop(columns=['date_recorded', 'construction_year'])
    # apply log transformations
    pump_dataframe.amount_tsh = pump_dataframe.amount_tsh.apply(lambda x: log10(x+1))
    pump_dataframe.population = pump_dataframe.population.apply(lambda x: log10(x+1))
    # convert public_meeting boolean values to string objects
    pump_dataframe.public_meeting = pump_dataframe.public_meeting.apply(lambda x: str(x))
    return pump_dataframe
