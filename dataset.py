from converters.json_files_to_csv import convert_json_dir_to_csv

intent_mapping = {
    10: "open_account_free",
    11: "open_account_current",
    12: "open_account_deposit",
    20: "loan_free",
    21: "loan_interest",
    30: "card2card",
    31: "paya",
    32: "convert_cheque",
    40: "receipt_payment",
    41: "installment_payment",
    50: "turnover_bill",
    51: "balance_bill",
    60: "submit_cheque",
    61: "recieve_cheque",
    70: "change_password",
    71: "duplicate_card",
    72: "close_card",
    80: "delegate_account",
    81: "currency_request",
    90: "software_problem",
    91: "signin_problem"
}


train_json_dir="train"
train_csv="train.csv"
validation_json_dir="validation"
validation_csv="validation.csv"

convert_json_dir_to_csv(
    input_dir=train_json_dir,
    output_csv_path=train_csv,
    text_col_name="input_text",
    intent_col_name="intent_id",
    slots_col_name="slots",
    intent_mapping=intent_mapping
)
convert_json_dir_to_csv(
    input_dir=validation_json_dir,
    output_csv_path=validation_csv,
    text_col_name="input_text",
    intent_col_name="intent_id",
    slots_col_name="slots",
    intent_mapping=intent_mapping
)