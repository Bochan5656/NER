import torch
from transformers import BertJapaneseTokenizer, BertForTokenClassification

def main():
    # 東北大学乾研究室の事前学習済みモデルを指定
    model_name = "cl-tohoku/bert-base-japanese-v3"
    
    # トークナイザーとモデルの初期化
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=0) # num_labelsはデータセットに合わせて変更します

    print("Model and Tokenizer loaded successfully.")

if __name__ == "__main__":
    main()
