# local-tts Web UI 使い方

## Irodori-TTS

https://github.com/Aratako/Irodori-TTS

## どのWeb UIを使うか

| 用途 | 起動ファイル | URL |
| --- | --- | --- |
| 通常の音声合成、参照音声あり/なし | `gradio_app.py` | `http://127.0.0.1:7860` |
| 長文を分割して1本のWAVにする | `gradio_app_long.py` | `http://127.0.0.1:7862` |
| VoiceDesign、声質を文章で指定 | `gradio_app_voicedesign.py` | `http://127.0.0.1:7861` |

## 通常版を起動

PowerShellで:

```powershell
# clone したリポジトリのディレクトリで実行
.\.venv\Scripts\python.exe .\gradio_app.py --server-name 127.0.0.1 --server-port 7860
```

ブラウザで開く:

```text
http://127.0.0.1:7860
```

基本操作:

1. `Checkpoint` はまず `Aratako/Irodori-TTS-500M-v3` のままでOK。
2. `Model Device` と `Codec Device` はGPUが使えるなら `cuda`、不安なら `cpu`。
3. `Model Precision` / `Codec Precision` はCPUなら `fp32`、CUDAなら `bf16` も選べます。
4. `Load Model` を押します。初回はモデルのダウンロードで時間がかかります。
5. `Text` に読み上げたい文章を入れます。
6. 声を寄せたい場合は `Reference Audio Upload` に音声ファイルを入れます。入れなければ参照なしで生成されます。
7. `Generate` を押します。

出力先:

```text
Irodori-TTS\gradio_outputs\
```

## 長文版を起動

```powershell
# clone したリポジトリのディレクトリで実行
.\.venv\Scripts\python.exe .\gradio_app_long.py --server-name 127.0.0.1 --server-port 7862
```

ブラウザ:

```text
http://127.0.0.1:7862
```

基本操作:

1. `Preset` は通常なら `v3 Base`。
2. `Load Model` を押します。
3. `Text` に長文を貼るか、`Text File` にテキストファイルを入れます。
4. 参照音声を使うなら `Reference Audio` に入れます。使わないなら `No Reference` をON。
5. まず `Preview Split` を押して分割結果を確認します。
6. 問題なければ `Generate Long Audio` を押します。

出力先:

```text
Irodori-TTS\gradio_long_outputs\日時フォルダ\
  long.wav
  chunk_001.wav
  chunk_002.wav
```

長文生成の目安:

- `Chunk Max Seconds` は30秒以下にします。既定値の25秒が無難です。
- 文章が細かく切れすぎる場合は `Chars Per Second` を上げます。
- チャンク間の間を調整したい場合は `Pause ms` を変えます。

## VoiceDesign版を起動

```powershell
# clone したリポジトリのディレクトリで実行
.\.venv\Scripts\python.exe .\gradio_app_voicedesign.py --server-name 127.0.0.1 --server-port 7861
```

ブラウザ:

```text
http://127.0.0.1:7861
```

基本操作:

1. `Model Checkpoint` は `Aratako/Irodori-TTS-500M-v2-VoiceDesign` のままでOK。
2. `Load Model` を押します。
3. `Text` に読み上げたい文章を入れます。
4. `Caption / Style Prompt` に声質や読み方を書きます。
5. `Generate` を押します。

Caption例:

```text
落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。
```

出力先:

```text
Irodori-TTS\gradio_outputs_voicedesign\
```

## 別PCやスマホから開く

同じLAN内の別端末から開きたい場合は、`--server-name 0.0.0.0` で起動します。

```powershell
.\.venv\Scripts\python.exe .\gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

別端末のブラウザでは、起動したPCのIPアドレスを使います。

```text
http://PCのIPアドレス:7860
```

## 停止方法

起動しているPowerShellで `Ctrl + C` を押します。モデルをメモリから外したいだけなら、Web UI上の `Unload Model` を押します。

## よくある困りごと

### `python` が見つからない

この環境ではPowerShellのPATHに `python` がないことがあります。次の形で起動してください。

```powershell
.\.venv\Scripts\python.exe .\gradio_app.py --server-name 127.0.0.1 --server-port 7860
```

### `Address already in use` やポート使用中

別のポートで起動します。

```powershell
.\.venv\Scripts\python.exe .\gradio_app.py --server-name 127.0.0.1 --server-port 7870
```

その場合は `http://127.0.0.1:7870` を開きます。

### 生成が遅い

CPUではかなり時間がかかります。GPUが使えるなら `Model Device` と `Codec Device` を `cuda` にします。長文ではまず短い文章で動作確認してから本番の文章を流すと安心です。

### メモリ不足になる

`Num Candidates` を1にします。長文版では `Chunk Max Seconds` を短くします。CUDA使用時は一度 `Unload Model` を押すか、PowerShellを止めて起動し直します。

### 初回だけ時間がかかる

Hugging Faceからモデルやcodecを取得します。2回目以降はキャッシュが効くので短くなります。

## 開発時の検証

```powershell
python -m unittest discover -s tests -v
```

3つのWeb UIで共通の入力変換を、モデルやGPUを読み込まずに確認します。
CIはWindowsとLinuxで実行します。実際の音声生成は、別途モデルを読み込んで確認してください。
