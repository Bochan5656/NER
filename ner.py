from transformers import pipeline
from transformers import BertJapaneseTokenizer, BertForTokenClassification

MODEL_DIR = "./model"

model = BertForTokenClassification.from_pretrained(MODEL_DIR)
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR)

ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)

text = "{'column0':'石丸伸二','column1':'174cm','column2':'100kg','column3':'東京都','column4':'世田谷区'}"
#text = "株式会社はJurabi、東京都台東区に本社を置くIT企業である。"
results = ner_pipeline(text)

print(f"\n入力文: {text}\n")
print("--- NER 結果（生データ）---")
for r in results:
    print(r)

# B/I タグをまとめて固有表現として整形表示
print("\n--- 抽出された固有表現 ---")
entities = []
current_entity = None
for r in results:
    label = r['entity']
    word = r['word']
    if label.startswith('B-'):
        if current_entity:
            entities.append(current_entity)
        current_entity = {'word': word, 'label': label[2:], 'score': r['score']}
    elif label.startswith('I-') and current_entity:
        # サブワード（## で始まるトークン）の場合は結合
        if word.startswith('##'):
            current_entity['word'] += word[2:]
        else:
            current_entity['word'] += word
        current_entity['score'] = (current_entity['score'] + r['score']) / 2
    else:
        if current_entity:
            entities.append(current_entity)
            current_entity = None
if current_entity:
    entities.append(current_entity)

if entities:
    for e in entities:
        print(f"  [{e['label']}] {e['word']}  (スコア: {e['score']:.4f})")
else:
    print("  固有表現は検出されませんでした。")
