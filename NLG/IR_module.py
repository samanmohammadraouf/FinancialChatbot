import numpy as np
from rank_bm25 import BM25Okapi
import pandas as pd

class IRModule:
    def __init__(self, dataframe, text_col_name, label_col_name, slot_col_name):
        self.df = dataframe
        self.text_col = text_col_name
        self.label_col = label_col_name
        self.slot_col_name = slot_col_name

    def get_examples(self, input_text, input_label, slot_name):
        filtered_df = self.df[self.df[self.label_col] == input_label]
        filtered_df = filtered_df[filtered_df[self.slot_col_name].str.contains(slot_name)]
        
        if len(filtered_df) == 0:
            return ""
            
        elif len(filtered_df) == 1:
            example = filtered_df.iloc[0][self.text_col]
            return f"مثال:\n{example}"
            
        else: 
            tokenized_docs = [doc.split() for doc in filtered_df[self.text_col]]
            bm25 = BM25Okapi(tokenized_docs)
            tokenized_query = input_text.split()
            
            doc_scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(doc_scores)[-2:][::-1]
            
            examples = []
            for idx in top_indices:
                if idx < len(filtered_df):
                    examples.append(filtered_df.iloc[idx][self.text_col])
            
            if len(examples) == 0:
                return ""
            elif len(examples) == 1:
                return f"مثال:\n{examples[0]}"
            else:
                return f"مثال اول:\n{examples[0]}\n\nمثال دوم:\n{examples[1]}"

    def get_guided_examples(self, input_text, input_label, slot_name):
        try:
            # First filter by label and slot 
            filtered_df = self.df[self.df[self.label_col] == input_label]
            filtered_df = filtered_df[filtered_df[self.slot_col_name].str.contains(slot_name)]
            
            if len(filtered_df) == 0:
                return ""
                
            # BM25 ranking
            tokenized_docs = [doc.split() for doc in filtered_df[self.text_col]]
            bm25 = BM25Okapi(tokenized_docs)
            tokenized_query = input_text.split()
            
            doc_scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(doc_scores)[-2:][::-1]

            guided_examples = []
            for idx in top_indices:
                if idx >= len(filtered_df):
                    continue
                    
                row = filtered_df.iloc[idx]
                text = row[self.text_col]
                slot_str = row[self.slot_col_name]
                
                try:
                    slot_list = eval(slot_str)
                except:
                    raise ValueError("Invalid slot format")
                
                words = text.split()
                if len(words) != len(slot_list):
                    raise ValueError("Text and slot length mismatch")
                
                guided_examples.append(list(zip(words, slot_list)))
            
            return guided_examples
            
        except Exception as e:
            print(f"Error in guided examples: {str(e)}")
            return self.get_examples(input_text, input_label, slot_name)