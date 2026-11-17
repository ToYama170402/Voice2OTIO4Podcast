# 使い方

## 基本の形

```bash
uv run podcast-preprocessor [オプション] 音声ファイル... 出力.otio
```

- 音声ファイルは話者ごとに1つずつ指定します。指定順に `source_1`、`source_2` として扱われ、それぞれ独立したトラック群になります。1ファイル内の話者分離は行いません。
- 出力OTIOと同じ場所に、字幕ファイル（拡張子を `.srt` に変えたもの）が生成されます。出力先は `--srt-output` で変更できます。
- 処理の進捗は標準エラーに表示されます（端末に接続されている場合のみ）。

## 生成物の見方（DaVinci Resolveでの編集）

OTIOをResolveにインポートすると、音声ファイルごとに「目印トラック」「字幕トラック」「音声トラック」の3トラックができます。

- カット候補の箇所には、目印トラックに `[要カット: 理由]` という名前のクリップが立ち、赤いマーカーが付きます。
- 目印・字幕・音声の3要素はリンクされているため、目印クリップを選ぶと対応する音声スライスも選択できます。不要と判断したら音声スライスごと削除してください。
- 音声は全時間が保持されています。候補でない箇所も含め、時間軸は元の音声のままです。
- `--acoustic-fillers` を付けた場合、音声クリップ上に黄色マーカー「フィラー候補」が付きます。これは候補の提示のみで、スライスは分かれていません。聞いて確認してから切ってください。

## レシピ

### CPUで軽く動かす（faster-whisper）

```bash
uv run podcast-preprocessor speaker_a.wav podcast.otio --model small --device cpu --compute-type int8
```

`small` でもメモリが不足する場合は `--model base` または `--model tiny` に下げてください。逆に精度を上げたい場合は `medium` や `turbo` を指定します（CPUでは時間がかかります）。

### SenseVoiceで速く処理する

```bash
uv sync --extra sensevoice
uv run podcast-preprocessor speaker_a.wav podcast.otio --backend sensevoice --acoustic-fillers
```

SenseVoiceはフィラーを文字起こしへ残さないため、フィラーを切りたい場合は `--acoustic-fillers` の併用がおすすめです。無音の多い音声ではVADも効きます。

```bash
uv run podcast-preprocessor speaker_a.wav podcast.otio --backend sensevoice --sensevoice-vad
```

### 長時間の音声・途中で止まっても大丈夫なようにする

何もしなくても、文字起こし結果はファイルごとにキャッシュされます（既定: `~/.cache/podcastpreprocessor/transcripts/`）。途中で失敗しても、同じ入力・同じ設定で再実行すれば、完了したファイルはキャッシュから読み込まれ、残りだけが処理されます。

SenseVoiceではさらに、チャンク（既定30秒）単位で途中経過が保存され、チャンクの途中から再開できます。faster-whisperはファイル単位のキャッシュのみです。

キャッシュを無効化する場合は `--no-cache`、保存先を変える場合は `--cache-dir` を指定します。入力ファイルや文字起こし設定が変わると、キーが変わって自動的に再文字起こしされます。

### ネットワークのない環境で使う

モデルを事前にダウンロードしておき、`--offline` を付けます。キャッシュ済みモデルだけで実行し、無ければエラーになります。

```bash
uv run podcast-preprocessor speaker_a.wav podcast.otio --backend sensevoice --offline
```

モデルを手動配置した場合は `--sensevoice-model`（SenseVoice本体）と `--sensevoice-vad-model`（VAD）で場所を指定できます。

### m4aなどの音声を使う

SenseVoiceバックエンドでは、入力音声は推論前に16kHzモノラルWAVへ自動変換されるため、m4a（AAC）などもそのまま渡せます。変換結果は `~/.cache/podcastpreprocessor/converted/` にキャッシュされ、再実行時は変換を省略します（`--sensevoice-audio-cache-dir` で変更可）。

## オプション一覧

`uv run podcast-preprocessor --help` の内容と同じです。

### 共通

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--backend` | `faster-whisper` | 文字起こしのバックエンド（`faster-whisper` / `sensevoice`） |
| `--srt-output` | 出力OTIOの拡張子を `.srt` にしたもの | 字幕SRTの出力先 |
| `--fps` | `30` | タイムラインのフレームレート |
| `--sequence-name` | `Podcast_Timeline` | OTIO内のシーケンス名 |
| `--acoustic-fillers` | 無効 | 音声信号からもフィラー候補を検出し、音声クリップに黄色マーカーで付ける |

### faster-whisper

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--model` | `small` | Whisperモデルサイズ（`tiny` / `base` / `small` / `medium` / `large-v3` / `turbo` など） |
| `--device` | `auto` | 推論デバイス（`auto` / `cpu` / `cuda`） |
| `--compute-type` | `default` | 計算型（`default` / `int8` / `float16` など。CPUでは `int8` が軽量） |

### SenseVoice

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--sensevoice-model` | 自動ダウンロード | ローカルのSenseVoiceモデルディレクトリ |
| `--sensevoice-quantization` | `int8` | ONNXモデルの量子化（`none` / `int8` / `int4`。`int4` は公式配布モデルが登録されている場合のみ） |
| `--sensevoice-cache-dir` | `~/.cache/podcastpreprocessor/` | SenseVoiceモデルのキャッシュ先 |
| `--sensevoice-audio-cache-dir` | `~/.cache/podcastpreprocessor/converted` | 16kHzモノラルWAVへの変換キャッシュ先 |
| `--sensevoice-num-threads` | `4` | CPU推論のスレッド数 |
| `--sensevoice-provider` | `cpu` | ONNX Runtimeのプロバイダ（`cpu` / `cuda` / `coreml`） |
| `--sensevoice-chunk-duration-s` | `30` | 文字起こしを進めるチャンクの長さ（秒）。途中経過はこの単位で保存される |
| `--batch-size-s` | `60` | 1回の推論にまとめる音声の長さ（秒） |
| `--language` | `ja` | SenseVoiceの言語コード |
| `--use-itn` / `--no-use-itn` | 有効 | 逆テキスト正規化（ITN: 数字・日付などの表記変換） |
| `--offline` | 無効 | モデルをダウンロードせず、キャッシュ済みモデルだけで実行する |

### SenseVoice VAD（`--sensevoice-vad` 指定時）

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--sensevoice-vad` | 無効 | Silero VADで発話区間だけを検出して推論する |
| `--sensevoice-vad-model` | 自動ダウンロード | ローカルのSilero VAD ONNXモデル |
| `--sensevoice-vad-threshold` | `0.5` | 発話確率のしきい値 |
| `--sensevoice-vad-min-silence-s` | `0.25` | 発話区間を終端する最小の無音長（秒） |
| `--sensevoice-vad-min-speech-s` | `0.1` | 保持する最小の発話長（秒） |
| `--sensevoice-vad-max-speech-s` | `120` | 1つの発話区間の最大長（秒）。長い発話だけこの長さで分割する |
| `--sensevoice-vad-padding-s` | `0.25` | 発話区間の前後に付ける余白（秒） |
| `--sensevoice-vad-threshold-db` | `-40` | エネルギーVADのしきい値（最大フレームに対するdB） |

### キャッシュ

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--cache-dir` | `~/.cache/podcastpreprocessor/transcripts` | 文字起こしキャッシュの保存先 |
| `--no-cache` | 無効 | 文字起こしキャッシュを使わない・保存しない |

## トラブルシューティング

- **モデルの読み込みに失敗する**: メモリ不足のことが多いです。faster-whisperなら `--model` を下げる、`--device cpu --compute-type int8` を指定する、SenseVoiceなら量子化 `int8` を使ってください。
- **ダウンロードで止まる**: 初回実行時はモデルを取得します。ネットワークが使えない環境では、事前にモデルを用意して `--offline` を使うか、`--sensevoice-model` でローカルのモデルを指定してください。
- **SenseVoiceでフィラーが候補に出ない**: SenseVoiceはフィラーを文字起こしへ残しません。`--acoustic-fillers` を併用してください。
- **再実行しても最初から処理される**: 入力ファイルのパス・サイズ・更新時刻や、文字起こし設定が変わるとキャッシュキーが変わります。`--no-cache` を付けていないかも確認してください。
- **Resolveでリンクやマーカーの見え方が違う**: Resolveのバージョンにより表示が異なる場合があります。字幕と音声のスライス自体は通常のクリップとして編集できます。
