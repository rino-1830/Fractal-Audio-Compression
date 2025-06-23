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