# Fractal-Audio-Compression
音源波形の自己相似性を利用したフラクタル圧縮デモです。各ブロックの参照元と線形変換を保存することで、指定回数の繰り返しで元波形を近似します。

## 準備
必要なパッケージは次のコマンドでインストールできます。
```bash
pip install -r requirements.txt
```

## 使い方
### 圧縮
```bash
python fractal_audio_compression.py input.wav dummy.wav \
    --mode compress --params params.npz \
    --block-size 1024 --search-step 512
```
`--block-size` と `--search-step` を調整すると圧縮率が変わります。入力がブロックサイズの倍数でない場合は自動でゼロパディングされます。

### 復元
```bash
python fractal_audio_compression.py dummy.wav output.wav \
    --mode decompress --params params.npz \
    --iterations 8 --scale 2
```
`--iterations` を増やすと復元精度が向上します。`--scale` を指定すると復元時にブロックインデックスを倍率分だけ拡大し、出力サンプルレートもその倍率で保存します。ブロックをそのまま拡大配置するだけの簡易的な手法のため、2倍スケール後に0.5倍速再生すると連続性が損なわれ「ぶつぶつ」したノイズが残ることがあります。高品質なアップスケーリングを望む場合は、別途タイムストレッチやリサンプラーと組み合わせてください。

### 評価
```bash
python fractal_audio_compression.py input.wav output.wav \
    --mode evaluate --params params.npz \
    --iterations 8
```
保存されたパラメータを用いて元音源との平均二乗誤差を計算します

## パラメータの影響

| オプション | 値を大きくすると | 値を小さくすると |
|-----------|----------------|----------------|
| `--block-size` | ブロックが大きくなり計算量は減るが細部の再現性が低下 | ブロックが細かくなり精度は上がるが処理時間が長くなる |
| `--search-step` | domain探索候補が減り高速になるが最適なブロックを逃しやすい | 候補が増えて精度は上がるが計算量が増える |
| `--iterations` | 繰り返し回数が増えるため復元品質が向上 | 回数を減らすと処理は速いが誤差が残りやすい |
| `--scale` | サンプルレートが高くなり任意倍率での出力が可能。ただしブロック拡大のみで補間は行われない | 1に近づくほど元のサンプルレートに近い |

`dummy.wav` は出力先のプレースホルダーであり実際には使用しません。

## 最適化について
ドメイン探索をベクトル化することで Python 上の二重ループを除去し、従来より大幅に高速化しました。復元処理もインデックスをまとめて扱うことでわずかに速度向上しています。アルゴリズム自体は同じため音質への影響はほぼありません。

## GPU 版
`fractal_audio_compression_GPU.py` は CuPy を利用した GPU 対応版です。計算は単精度 (`float32`) で行われるためメモリを節約できます。使用するには CUDA 対応 GPU と CuPy のインストールが必要です。

## コントリビューション
このプロジェクトの一部コードとドキュメントは OpenAI の Codex を用いて生成および修正されました。