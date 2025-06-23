# Fractal-Audio-Compression
音源波形の自己相似性を利用した音声圧縮

## 使用方法

`fractal_audio_compression.py` はステレオWAVファイルを入力として、
フラクタル圧縮および展開を行うサンプルスクリプトです。
処理の進捗は `tqdm` によるプログレスバーで表示されます。

必要なパッケージは以下のコマンドでインストールできます。

```bash
pip install -r requirements.txt
```


### 圧縮

```bash
python fractal_audio_compression.py input.wav dummy.wav \
    --mode compress --params params.npz \
    --block-size 1024 --search-step 512
```

`--block-size` と `--search-step` を変更することで圧縮率を調整できます。
入力長がブロックサイズの倍数でない場合は自動的にゼロパディングされます。

### 復元

```bash
python fractal_audio_compression.py dummy.wav output.wav \
    --mode decompress --params params.npz \
    --iterations 8 --scale 2
```

`--iterations` を増やすと復元精度が向上します。
`--scale` でサンプルレートを何倍にして出力するかを指定できます。
フラクタル圧縮の特性上、任意の倍率でのアップスケーリングが可能です。
復元時には圧縮時のパディングが自動的に除去されます。

### 評価

```bash
python fractal_audio_compression.py input.wav output.wav \
    --mode evaluate --params params.npz \
    --iterations 8
```

保存されたパラメータから音源を復元し、
元ファイルとの平均二乗誤差を表示します。

## パラメータの影響

| オプション | 値を大きくすると | 値を小さくすると |
|-----------|----------------|----------------|
| `--block-size` | ブロックが大きくなり、計算量は減るが細部の再現性が低下する | ブロックが細かくなり精度は上がるが処理時間は長くなる |
| `--search-step` | domain探索候補が減り高速になるが、最適なブロックが見つかりにくくなる | 候補が増えて精度は上がるが計算量が増える |
| `--iterations` | 繰り返し回数が増えるため復元品質が向上する | 回数を減らすと早く終わるが誤差が残りやすい |
| `--scale` | サンプルレートが高くなり任意倍率での出力が可能 | 1に近づくほど元の音源に近いサンプルレートになる |

`dummy.wav` は使用されませんが、引数として指定する必要があります。

## 最適化の効果

ドメイン探索をベクトル化したことで Python での二重ループを排除し、
圧縮処理は環境にもよりますが従来の数分の一程度まで短縮されます。
復元処理もインデックスをまとめて扱うことで若干高速化しましたが、
アルゴリズム自体は同じため音質への影響はほとんどありません。
さらにブロックの代入をまとめて行うよう改善し、
わずかながら復元時間を短縮しました。

## GPU版について

`fractal_audio_compression_GPU.py` は CuPy を利用した GPU 対応版です。
計算は単精度 (`float32`) で行うことでメモリ使用量を削減しています。
利用する際は CUDA 対応 GPU と CuPy のインストールが必要です。