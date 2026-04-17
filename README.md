# NER

Hugging Face TransformersのBERTを使用し、日本語の固有表現抽出（NER: Named Entity Recognition）タスクにおけるモデルのファインチューニングを行うためのリポジトリです。
東北大学乾研究室が公開している日本語の事前学習モデルを使用します。

## 環境構築

このリポジトリを使用するには、まず必要なライブラリをインストールしてください。
以下のコマンドを実行して、依存パッケージをインストールします。

```bash
pip install torch transformers unidic_lite fugashi scikit-learn datasets seqeval
```
