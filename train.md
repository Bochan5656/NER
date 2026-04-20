# train.py 詳細解説

このドキュメントでは、日本語 NER モデルをファインチューニングする `train.py` のプログラム詳細を、初学者向けにわかりやすく解説します。

---

## 全体の処理フロー

```
① 設定・データ読み込み
        ↓
② NERDataset クラスの定義（PyTorch Dataset）
        ↓
③ 学習用・検証用データへの分割
        ↓
④ モデル・Trainer の準備
        ↓
⑤ 学習の実行 → 評価 → モデル保存
```

---

## ブロック① 設定・データ読み込み（1〜16行目）

```python
import json
import torch
from transformers import BertJapaneseTokenizer, BertForTokenClassification, BertConfig
from label import label2id, id2label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LENGTH = 128
BERT_MODEL = "cl-tohoku/bert-base-japanese-v2"
TAGGED_DATASET_PATH = "./ner-wikipedia-dataset/ner_tagged.json"
MODEL_DIR = "./model"
LOG_DIR  = "./logs"

with open(TAGGED_DATASET_PATH, 'r') as f:
    encoded_tagged_sentence_list = json.load(f)
```

### 定数の意味

| 定数 | 値 | 説明 |
|---|---|---|
| `MAX_LENGTH` | 128 | 1文あたりのトークン数の上限。超えた部分は切り捨て |
| `BERT_MODEL` | `cl-tohoku/bert-base-japanese-v2` | 東北大学乾研究室の日本語 BERT 事前学習モデル |
| `TAGGED_DATASET_PATH` | `ner_tagged.json` | `ragged.py` が生成した学習用データ（BIO タグ付き） |
| `MODEL_DIR` | `./model` | 学習後のモデルを保存するフォルダ |
| `LOG_DIR` | `./logs` | 学習ログ（損失の推移など）を保存するフォルダ |

### device（実行デバイス）の自動選択

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

GPU（CUDA）が使える環境なら自動的に GPU で高速実行し、なければ CPU で実行します。

### 読み込むデータ（`ner_tagged.json`）の構造

`ragged.py` によって前処理済みのデータが入っています。各サンプルは以下の辞書形式です：

```json
{
  "input_ids":      [2, 1234, 567, ...],   // トークンを数値IDに変換したもの
  "attention_mask": [1, 1, 1, ..., 0, 0],  // 有効なトークンに1、パディングに0
  "labels":         [-100, 9, 10, ..., -100] // 正解ラベルID（-100は損失計算を無視）
}
```

| キー | 内容 |
|---|---|
| `input_ids` | 各トークンの数値ID（BERT の辞書に基づく） |
| `attention_mask` | 実際のトークンは `1`、パディング部分は `0` |
| `labels` | 各トークンの正解ラベルID。`-100` は `[CLS]`/`[SEP]` などの特殊トークンで損失計算から除外 |

---

## ブロック② NERDataset クラス（21〜31行目）

```python
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_tagged_sentence_list):
        self.encoded_tagged_sentence_list = encoded_tagged_sentence_list

    def __len__(self):
        return len(self.encoded_tagged_sentence_list)https://github.com/

    def __getitem__(self, idx):
        item = {key: torch.tensor(val).to(device) for key, val in
                self.encoded_tagged_sentence_list[idx].items()}
        return item
```

PyTorch でデータを扱うための**カスタムデータセットクラス**です。  
`torch.utils.data.Dataset` を継承することで、`Trainer` が自動的にバッチ処理できるようになります。

### 各メソッドの役割

| メソッド | 役割 |
|---|---|
| `__init__` | データリストを受け取ってインスタンス変数に保存 |
| `__len__` | データ全件数を返す（ループの終了条件に使われる） |
| `__getitem__` | インデックス `idx` のデータを辞書形式で返す。値はすべて PyTorch の Tensor に変換し、`device`（GPU/CPU）に転送する |

### Tensor への変換が必要な理由

```python
torch.tensor(val).to(device)
```

BERT モデルは Python のリストをそのまま受け取れないため、数値配列を GPU/CPU 上の **Tensor（テンソル）** に変換する必要があります。

---

## ブロック③ データの分割（33〜37行目）

```python
train_encoded_tagged_sentence_list, eval_encoded_tagged_sentence_list = \
    train_test_split(encoded_tagged_sentence_list)

train_data = NERDataset(train_encoded_tagged_sentence_list)
eval_data  = NERDataset(eval_encoded_tagged_sentence_list)
```

`sklearn` の `train_test_split` で全データを**学習用（75%）と検証用（25%）**に分割します（デフォルト比率）。

| 変数 | データ量の目安 | 用途 |
|---|---|---|
| `train_data` | 全体の約 75% | モデルのパラメータ更新に使う |
| `eval_data` | 全体の約 25% | 学習中の精度確認（過学習の検出）に使う |

> **なぜ分割するの？**  
> 同じデータで学習と評価を行うと、モデルが答えを丸暗記してしまいます（過学習）。  
> 学習に使っていない検証用データで評価することで、**未知のデータへの汎化性能**を正しく測れます。

---

## ブロック④ モデルと Trainer の準備（46〜93行目）

### 4-1. 事前学習モデルの読み込み

```python
config = BertConfig.from_pretrained(BERT_MODEL, id2label=id2label, label2id=label2id)
model  = BertForTokenClassification.from_pretrained(BERT_MODEL, config=config).to(device)
tokenizer = BertJapaneseTokenizer.from_pretrained(BERT_MODEL)
```

| コード | 説明 |
|---|---|
| `BertConfig.from_pretrained(...)` | モデルの設定情報を読み込み、`label2id`/`id2label` のラベル対応表を注入する |
| `BertForTokenClassification.from_pretrained(...)` | 東北大 BERT の事前学習済みの重みを読み込み、NER 用の分類ヘッドを追加したモデルを構築する |
| `.to(device)` | モデルを GPU または CPU に転送する |
| `BertJapaneseTokenizer.from_pretrained(...)` | 日本語テキストをトークンに分割するためのトークナイザーを読み込む |

#### ラベル対応表（`label.py` より）

`label2id` / `id2label` は `label.py` で定義されており、以下の 17 ラベルを使います：

| ID | ラベル | 意味 |
|---|---|---|
| 0 | `O` | 固有表現以外 |
| 1 | `B-人名` | 人名の先頭 |
| 2 | `I-人名` | 人名の継続 |
| 3 | `B-法人名` | 法人名の先頭 |
| 4 | `I-法人名` | 法人名の継続 |
| 5 | `B-政治的組織名` | 政治的組織名の先頭 |
| 6 | `I-政治的組織名` | 政治的組織名の継続 |
| 7 | `B-その他の組織名` | その他組織名の先頭 |
| 8 | `I-その他の組織名` | その他組織名の継続 |
| 9 | `B-地名` | 地名の先頭 |
| 10 | `I-地名` | 地名の継続 |
| 11 | `B-施設名` | 施設名の先頭 |
| 12 | `I-施設名` | 施設名の継続 |
| 13 | `B-製品名` | 製品名の先頭 |
| 14 | `I-製品名` | 製品名の継続 |
| 15 | `B-イベント名` | イベント名の先頭 |
| 16 | `I-イベント名` | イベント名の継続 |

---

### 4-2. 学習パラメーター（TrainingArguments）

```python
training_args = TrainingArguments(
    output_dir                = MODEL_DIR,  # モデルの保存先
    num_train_epochs          = 2,          # 全データを何周学習するか
    per_device_train_batch_size = 8,        # 学習時のバッチサイズ
    per_device_eval_batch_size  = 32,       # 評価時のバッチサイズ
    warmup_steps              = 500,        # 学習率ウォームアップのステップ数
    weight_decay              = 0.01,       # 重み減衰率（過学習防止）
    logging_dir               = LOG_DIR,    # ログの保存先
)
```

#### 各パラメーターの詳細

| パラメーター | 値 | 説明 |
|---|---|---|
| `num_train_epochs` | 2 | 全データを2周（エポック）学習する。多いほど精度が上がりやすいが過学習のリスクも増す |
| `per_device_train_batch_size` | 8 | 1回の重み更新に使うサンプル数（学習時）。小さいほどメモリ使用量が少ない |
| `per_device_eval_batch_size` | 32 | 評価時のバッチサイズ。評価は重みを更新しないので大きくできる |
| `warmup_steps` | 500 | 最初の 500 ステップは学習率を 0 から徐々に上げる。急激な更新によるモデル崩壊を防ぐ |
| `weight_decay` | 0.01 | L2 正則化の強さ。大きすぎると学習が進まず、小さすぎると過学習する |

#### 学習率ウォームアップのイメージ

```
学習率
  ↑
  |         ___________（最大学習率を維持）
  |        /
  |       /
  |      /
  |_____/
  +----+------------------→ ステップ数
       ↑
   500 ステップ（warmup完了）
```

---

### 4-3. 評価指標の計算（`compute_metrics`）

```python
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)  # ①

    predictions = [[id2label[p] for p in prediction] for prediction in predictions]  # ②
    labels      = [[id2label[l] for l in label] for label in labels]                 # ②

    results = metric.compute(predictions=predictions, references=labels)  # ③
    return {
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
        "accuracy":  results["overall_accuracy"],
    }
```

| ステップ | 処理内容 |
|---|---|
| ① `np.argmax(predictions, axis=2)` | 各トークンについてラベル数分の確率値（ロジット）の中から、最も確率の高いラベルのインデックスを選ぶ |
| ② `id2label[...]` | 数値の ID をラベル文字列（`"B-地名"` など）に変換する |
| ③ `metric.compute(...)` | seqeval ライブラリで NER 専用の評価指標（Precision / Recall / F1）を計算する |

#### 評価指標の意味

| 指標 | 計算式 | 意味 |
|---|---|---|
| **Precision（適合率）** | 正解 ÷ モデルが検出した総数 | 検出した固有表現のうち、正しかった割合 |
| **Recall（再現率）** | 正解 ÷ 正解データ中の総数 | 正解データの固有表現のうち、検出できた割合 |
| **F1 スコア** | 2 × P × R ÷ (P + R) | Precision と Recall の調和平均。総合的な精度指標 |
| **Accuracy（正確度）** | 正解トークン数 ÷ 総トークン数 | 全トークンのうち正しくラベルを付けた割合 |

> **NER では F1 スコアが最重要指標です。**  
> 文章の大半が `O`（固有表現以外）のため、Accuracy だけ高くても意味がない場合があります。

#### seqeval の評価が「トークン単位」ではない理由

seqeval は BIO タグを考慮した**固有表現単位**での評価を行います。  
例えば「東京都台東区」が `B-地名 → I-地名` の2トークンで構成されている場合、  
**両方のトークンが正解** のときのみ、その固有表現を正解と見なします。

---

### 4-4. Trainer の初期化

```python
trainer = Trainer(
    model            = model,           # 学習対象のモデル
    args             = training_args,   # 学習パラメーター
    compute_metrics  = compute_metrics, # エポック終了後に呼ばれる評価関数
    train_dataset    = train_data,      # 学習用データセット
    eval_dataset     = eval_data,       # 検証用データセット
    processing_class = tokenizer        # トークナイザー（データコレーターに使用）
)
```

`Trainer` は以下をすべて自動で管理してくれます：
- バッチ単位での forward/backward パス
- 勾配計算と重みの更新（Adam オプティマイザー）
- エポックごとの評価と結果のログ出力
- GPU への自動転送

---

## ブロック⑤ 学習・評価・保存（95〜99行目）

```python
trainer.train()         # 学習の実行
trainer.evaluate()      # 最終評価
trainer.save_model(MODEL_DIR)  # モデルの保存
```

| メソッド | 処理内容 |
|---|---|
| `trainer.train()` | 設定したエポック数・バッチサイズで学習を実行。各エポック後に `compute_metrics` で評価が行われる |
| `trainer.evaluate()` | 学習終了後に検証用データで最終評価を実行し、指標を表示する |
| `trainer.save_model(MODEL_DIR)` | 学習済みモデルの重みと設定を `./model` フォルダに保存する。このフォルダが `ner.py` で使われる |

### `./model` フォルダに保存されるファイル

| ファイル | 内容 |
|---|---|
| `config.json` | モデルの構造設定（ラベル対応表を含む） |
| `model.safetensors` | 学習済みの重みパラメーター |
| `tokenizer_config.json` | トークナイザーの設定 |
| `vocab.txt` | 日本語の語彙辞書 |

---

## まとめ

| ブロック | 処理 | 主なクラス/関数 |
|---|---|---|
| ① | 設定・データ読み込み | `json.load`, `torch.device` |
| ② | データセットクラス定義 | `NERDataset(Dataset)` |
| ③ | データ分割 | `train_test_split` |
| ④ | モデル・Trainer 準備 | `BertForTokenClassification`, `TrainingArguments`, `Trainer` |
| ⑤ | 学習・評価・保存 | `trainer.train()`, `trainer.evaluate()`, `trainer.save_model()` |
