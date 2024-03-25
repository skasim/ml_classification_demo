import numpy as np 
import pandas as pd
import os
from dotenv import load_dotenv

import constants
import utils

load_dotenv()  # take environment variables from .env.

train_file=os.getenv("TRAIN_FILE")
google_vecs_file=os.getenv("GOOGLE_VECS_FILE")
label_enc_file=os.getenv("LABEL_ENC_FILE")
minmax_scaler_file=os.getenv("MINMAX_SCALER_FILE")
out_data_file = os.getenv("OUT_DATA_FILE")

def file_generator(data_file, out_data_file, google_vecs_file, label_enc_file, minmax_scaler_file, is_train=False):    

    # load file
    with open(data_file) as f:
        df = pd.read_csv(f, header=0)
        print(f"file loaded...")
    
    df = df.rename(columns=constants.column_renames)
    # drop rows with null values for target label
    df = df[df.country_of_origin.notnull()]
    # drop columns with high correlation between them
    cols2drop = ["_id", "shipper_address", "consignee_address", "notify_address", "bill_of_lading",
                        "weight_kg", "marks_and_numbers", "carrier_name", "carrier_address", "carrier_city",
                        "carrier_state", "carrier_zip", "distribution_port", "quantity", "quantity_unit", "measurement",
                        "measurement_unit", "consignee_zipcode"]
    df.drop(columns=cols2drop, inplace=True)
    # move y to end of the dataframe
    column_to_move = df.pop("country_of_origin")
    df.insert(29 - len(cols2drop), "country_of_origin", column_to_move)
    print("data munging complete.")

    # divide features for processing
    df_categorical = df.copy(deep=True)
    df_product_details = df.copy(deep=True)
    df_numerical = df.copy(deep=True)
    df_categorical = df_categorical[["shipper", "consignee", "us_port", "notify", "vessel_name", "ship_registered_in", "carrier_code", "country_of_origin"]]
    df_product_details = df_product_details[["product_details"]]
    df_numerical = df_numerical[["arrival_date", "weight_lb", "container_count"]]

    # categorical data tranformations
    values_to_keep = utils.ident_top_values_to_keep(df_categorical, df, is_train)
    df_categorical = utils.replace_low_ranking_values(df_categorical, values_to_keep)
    df_target = df_categorical.copy(deep=True)
    df_target = df_target[["country_of_origin"]]
    df_categorical.drop(columns=["country_of_origin"], inplace=True)
    df_categorical = utils.impute_missing_values_categorical(df_categorical)
    one_hot_enc_categorical = pd.get_dummies(df_categorical, columns=['shipper', 'consignee', 'us_port', 'notify', 'vessel_name',
       'ship_registered_in', 'carrier_code',], drop_first=True, dtype=int) 
    print("categorical transformations complete. starting text transformations. this will take ~8 minutes. get some coffee.")
    
    # text data transformations
    df_product_details = utils.impute_missing_text_data(df_product_details)
    vec_product_details = utils.vectorize_product_details(google_vecs_file, df_product_details)
    print("text transformations complete.")

    # numeric data
    df_numerical['arrival_date'] = pd.to_datetime(df_numerical['arrival_date'], format="mixed")
    df_numerical["arrival_date"] = df_numerical['arrival_date'].dt.to_period('M')
    df_date = df_numerical.copy(deep=True)
    df_date.drop(columns=["weight_lb", "container_count"], inplace=True)    
    df_numerical.drop(columns=["arrival_date"], inplace=True)
    print("numeric transformations complete.")

    # combine one-hot encoded categorical, text vectorized product details, and binned date
    one_hot_enc_categorical.reset_index(drop=True, inplace=True) # categorical column    
    vec_product_details.reset_index(drop=True, inplace=True) # text column
    df_date.reset_index(drop=True, inplace=True) # date column
    combi_df = pd.concat([df_date, one_hot_enc_categorical, vec_product_details], axis=1) 
    # label encode target
    y = utils.label_encode_target(df_target, label_enc_file, is_train=is_train)
    print("target label data encoding complete.")

    labeled_target = pd.DataFrame(y, columns=["country_of_origin_labels"])
    # apply MinMaxScaler to weight_lb and container_count and save the scaler for application to test set
    scaled = utils.minmax_scale_numeric_data(df_numerical, y, minmax_scaler_file, is_train=is_train)
    print("numeric data scaling complete.")

    # convert np array to df
    scaled_numeric = pd.DataFrame(scaled, columns=["weight_lb_scaled", "container_count_scaled"])
    # add the above df to the combi df to get final dataframe
    scaled_numeric.reset_index(drop=True, inplace=True) # numerical data
    labeled_target.reset_index(drop=True, inplace=True) # labeled target
    combi_df.reset_index(drop=True, inplace=True)
    X_df = pd.concat([scaled_numeric, combi_df, labeled_target], axis=1) 
    print(f"outdatafile: {out_data_file}")
    X_df.to_pickle(out_data_file)
    print(X_df.head())
    print(f"file {out_data_file} saved to disk.")




if __name__ == "__main__":
    google_vecs_file = os.getenv("GOOGLE_VECS_FILE")
    label_enc_file = os.getenv("LABEL_ENC_FILE")
    minmax_scaler_file = os.getenv("MINMAX_SCALER_FILE")
    out_data_file = os.getenv("OUT_DATA_FILE")
    is_train = (os.getenv("IS_TRAIN", "False") == "True")
    out_data_file = os.getenv("OUT_DATA_FILE_TRAIN") if is_train else os.getenv("OUT_DATA_FILE_TEST")
    in_data_file = os.getenv("TRAIN_FILE") if is_train else os.getenv("TEST_FILE")
    print(f"input file: {in_data_file}")
    print(f"output file: {out_data_file}")
    if is_train:    
        print(f"Proceeding with training data...")
    else:
        print(f"Proceeding with test data...")
    file_generator(in_data_file, out_data_file, google_vecs_file, label_enc_file, minmax_scaler_file, is_train=is_train)
