# ner.py 詳細解説

このドキュメントでは、学習済みモデルを使って日本語テキストから固有表現を抽出する `ner.py` のプログラム詳細と、出力されるスコアの意味・算出方法について解説します。

---

## 1. プログラム全体の流れ

```
モデル読み込み → パイプライン作成 → テキスト推論 → 生データ表示 → 固有表現の整形・表示
```

---

## 2. 各ブロックの詳細

### ブロック①：インポートとモデル読み込み

```python
from transformers import pipeline
from transformers import BertJapaneseTokenizer, BertForTokenClassification

MODEL_DIR = "./model"

model = BertForTokenClassification.from_pretrained(MODEL_DIR)
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_DIR)
```

| 要素 | 説明 |
|---|---|
| `BertForTokenClassification` | トークンごとにラベルを予測する BERT モデル。`train.py` で学習・保存されたものを読み込む |
| `BertJapaneseTokenizer` | 日本語テキストをトークン（単語の断片）に分割するクラス |
| `MODEL_DIR = "./model"` | `train.py` が学習結果を保存したフォルダ |

---

### ブロック②：NER パイプラインの作成

```python
ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)
```

Hugging Face の `pipeline` は、テキストの前処理 → モデル推論 → 後処理をまとめて行う便利なクラスです。  
`'ner'` を指定することで、固有表現抽出タスク専用のパイプラインが生成されます。

---

### ブロック③：推論の実行と生データ表示

```python
text = "株式会社はJurabi、東京都台東区に本社を置くIT企業である。"
results = ner_pipeline(text)

print(f"\n入力文: {text}\n")
print("--- NER 結果（生データ）---")
for r in results:
    print(r)
```

`results` はトークンごとの予測結果のリストで、以下のような辞書形式になっています：

```python
{
    'entity': 'B-法人名',   # BIO タグ付きのラベル名
    'score': 0.9821,        # モデルの確信度（0〜1）
    'index': 3,             # 入力トークン列における位置インデックス
    'word': 'Jurabi',       # 対応するトークン文字列
    'start': 5,             # 入力文字列中の開始位置（文字インデックス）
    'end': 11               # 入力文字列中の終了位置（文字インデックス）
}
```

> **注意**：「生データ」ではトークン単位で1行ずつ出力されるため、1つの固有表現が複数行に分かれることがあります（例：`B-地名` の後に `I-地名` が続く）。

---

### ブロック④：固有表現の整形・表示（BIO タグのマージ）

```python
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
```

この処理では、BIO タグで分割されたトークンを1つの固有表現にまとめています。

#### BIO タグとは

| タグ | 意味 | 例 |
|---|---|---|
| `B-XXX` | 固有表現の **先頭** トークン | `B-地名` → 「東京」 |
| `I-XXX` | 固有表現の **継続** トークン（2文字目以降） | `I-地名` → 「都台東区」 |
| `O` | 固有表現 **以外** のトークン | 「に」「本社」など |

#### サブワードの処理

BERT のトークナイザは、辞書にない単語を `##` プレフィックス付きのサブワードに分割することがあります。

```
例：「Jurabi」→ ['Ju', '##ra', '##bi']
```

`##` で始まるトークンは前のトークンと結合（`##` 部分は除去）し、元の単語を復元します。

---

## 3. スコアの意味と算出方法

### スコアとは

各トークンに付与されるスコアは、モデルが**そのラベルを予測することへの確信度**を表します。  
値の範囲は `0.0`〜`1.0` で、**1.0 に近いほど自信がある**ことを意味します。

### 内部での算出方法（Softmax）

BERT モデルは最終層でラベル数分の数値（ロジット）を出力します。  
これを Softmax 関数で確率分布に変換したものがスコアです。

```
ロジット: [−2.1,  0.3,  8.4,  1.2, ...]   ← ラベル数分の生スコア
    ↓ Softmax
確率:      [0.001, 0.01, 0.982, 0.005, ...] ← 合計が 1.0 になる
```

最も確率が高いラベルが予測ラベルとして選ばれ、その確率値がスコアとして出力されます。

### 複数トークンにまたがる固有表現のスコア

`B-XXX` と `I-XXX` の複数トークンにまたがる固有表現のスコアは、`ner.py` では**逐次的な平均**で算出しています：

```python
current_entity['score'] = (current_entity['score'] + r['score']) / 2
```

例えば「東京都台東区」が3トークンに分かれた場合：

| ステップ | 処理 | スコア |
|---|---|---|
| `B-地名`「東京」 | 新規エンティティ作成 | 0.98 |
| `I-地名`「都台東区」 | 平均計算 | (0.98 + 0.96) / 2 = 0.97 |

> **補足**：より厳密にしたい場合は、全トークンのスコアを集めて最後に一括平均する方法もあります。

### スコアの読み方の目安

| スコア範囲 | 解釈 |
|---|---|
| 0.95 以上 | 高確信度。ほぼ確実に正しい予測 |
| 0.80〜0.95 | 概ね信頼できる予測 |
| 0.50〜0.80 | やや不確か。文脈によっては誤検出の可能性あり |
| 0.50 未満 | 信頼性が低い。要注意 |

---

## 4. 出力例

```
入力文: 株式会社はJurabi、東京都台東区に本社を置くIT企業である。

--- NER 結果（生データ）---
{'entity': 'B-法人名', 'score': 0.9821, 'index': 3, 'word': 'Jurabi', 'start': 5, 'end': 11}
{'entity': 'B-地名',   'score': 0.9743, 'index': 6, 'word': '東京', 'start': 12, 'end': 14}
{'entity': 'I-地名',   'score': 0.9612, 'index': 7, 'word': '都台東区', 'start': 14, 'end': 18}

--- 抽出された固有表現 ---
  [法人名] Jurabi    (スコア: 0.9821)
  [地名]   東京都台東区 (スコア: 0.9678)
```

---

## 5. まとめ

| 項目 | 内容 |
|---|---|
| モデル | `./model` に保存された `BertForTokenClassification` |
| 推論単位 | トークン（単語の断片）ごと |
| スコア算出 | Softmax による確率値（複数トークンは逐次平均） |
| BIO マージ | `B-` で開始、`I-` で継続、`O` または新しい `B-` で確定 |
| サブワード結合 | `##` プレフィックスのトークンは前のトークンに結合 |
