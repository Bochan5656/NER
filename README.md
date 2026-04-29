# NER

Hugging Face TransformersのBERTを使用し、日本語の固有表現抽出（NER: Named Entity Recognition）タスクにおけるモデルのファインチューニングを行うためのリポジトリです。
東北大学乾研究室が公開している日本語の事前学習モデルを使用します。

## 環境構築

このリポジトリを使用するには、まず必要なライブラリをインストールしてください。
以下のコマンドを実行して、仮想環境の作成と依存パッケージのインストールを行います。

```bash
conda create -n ner python=3.10 -y
conda activate ner
pip install torch transformers unidic_lite fugashi scikit-learn datasets seqeval evaluate accelerate
```

## レポート
詳しいBERTやNERについて：[レポート](note.md)
