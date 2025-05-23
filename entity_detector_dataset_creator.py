import pandas as pd
from ast import literal_eval

def convert_slots_to_entity_detection_format(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.drop(columns=['label'])
    
    df['slots'] = df['slots'].apply(literal_eval)
    
    def convert_labels(slots):
        return ['O' if tag == 'O' or tag == 'o' else 'E' for tag in slots]
    
    df['labels'] = df['slots'].apply(convert_labels)
    df['tokens'] = df['text'].apply(lambda x: x.split())
    
    count = 0
    for idx, row in df.iterrows():
        if len(row['tokens']) != len(row['labels']):
            count += 1
            continue
    
    print(count)
    df[['tokens', 'labels']].to_csv(output_path, index=False)

convert_slots_to_entity_detection_format("train.csv", "entity_detection_train.csv")
convert_slots_to_entity_detection_format("validation.csv", "entity_detection_val.csv")