# Fractal-Audio-Compression
音源波形の自己相似性を利用した音声圧縮

## 使用方法

`fractal_audio_compression.py` はステレオWAVファイルを入力として、
フラクタル圧縮および展開を行うサンプルスクリプトです。

### 圧縮

```bash
python fractal_audio_compression.py input.wav dummy.wav --mode compress --params params.npz
```

入力長がブロックサイズの倍数でない場合は自動的にゼロパディングされます。

### 復元

```bash
python fractal_audio_compression.py dummy.wav output.wav --mode decompress --params params.npz
```

復元時には圧縮時のパディングが自動的に除去されます。

`dummy.wav` は使用されませんが、引数として指定する必要があります。

## 最適化の効果

ドメイン探索をベクトル化したことで Python での二重ループを排除し、
圧縮処理は環境にもよりますが従来の数分の一程度まで短縮されます。
復元処理もインデックスをまとめて扱うことで若干高速化しましたが、
アルゴリズム自体は同じため音質への影響はほとんどありません。