import pandas as pd
import re
import json
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

class CSVtoJSONConverter:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            captions += "Findings: "
            captions += row["findings"]
            captions += " "
            captions += "Impression: "
            captions += row["impression"]

            # use space instead of newline
            captions = captions.replace("\n", " ") # \n替换

            # split sentences
            splitter = re.compile("[0-9]+\.") # 按照1. 2.进行分割
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions] # .分割
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions: # 遍历每个子句
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ") # 无法显示或者无法识别的字符
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+") # nltk tokenizer
                tokens = tokenizer.tokenize(cap.lower()) # 句子划分成单词 ['findings', 'there', 'is', 'no', 'focal', 'consolidation', 'pleural', 'effusion', 'or', 'pneumothorax']
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii") # 忽略非 ASCII 字符（如 é）
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens).capitalize())

                cnt += len(included_tokens)

            if cnt >= 3: # 如果超过3句话
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row["Path"]] = ". ".join(study_sent) + "."  # Combine sentences into a single string with periods

        return path2sent

    def convert_to_json(self, output_train_file, output_valid_file):
        path2sent = self.create_path_2_sent_mapping()
        
        train_data = []
        valid_data = []
        
        for _, row in self.df.iterrows():
            path = row["Path"]
            split = row["split"]
            if path in path2sent:
                entry = {
                    "id": row["Path"].split("/")[-1].split(".")[0],
                    "image": row["Path"].split("/")[-1],
                    "conversations": [
                        {
                            "from": "human",
                            "value": "Illustrate the image through a detailed medical report\n<image>"
                        },
                        {
                            "from": "report",
                            "value": path2sent[path]
                        }
                    ]
                }
                if split == "train":
                    train_data.append(entry)
                elif split == "valid":
                    valid_data.append(entry)
        
        with open(output_train_file, 'w') as f:
            json.dump(train_data, f, indent=4)
        
        with open(output_valid_file, 'w') as f:
            json.dump(valid_data, f, indent=4)

if __name__ == "__main__":
    converter = CSVtoJSONConverter("data/MIMIC-CXR/images_processed/master.csv")
    converter.convert_to_json("train_IF_mention.json", "valid_IF_mention.json")