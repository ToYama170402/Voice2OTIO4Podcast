# PodcastPreprocessor

日本語ポッドキャストの音声ファイル群を文字起こしし、字幕・カット候補・元音声をリンクしたOpenTimelineIO（`.otio`）を生成します。

## セットアップ

```powershell
uv sync --extra dev
```

faster-whisperのモデルとSudachi辞書は初回セットアップ時に取得されます。CPUではメモリ使用量を抑えるため、`small`モデルと`int8`を指定してください。

## 実行

```powershell
uv run podcast-preprocessor speaker_a.wav speaker_b.wav podcast.otio --model small --device cpu --compute-type int8 --fps 30
```

`turbo`など大きいモデルをCPUで読み込めない場合は、`base`または`tiny`に下げてください。モデルの読み込みに失敗した場合、CLIは利用可能な設定を案内して終了します。

処理は次の順で進みます。

1. faster-whisperで日本語音声を単語タイムスタンプ付きで認識
2. 句読点、文字数、表示時間を基準に字幕を分割
3. 0.4秒超の無音、フィラー、`no_speech_prob > 0.8`をカット候補として判定
4. DaVinci Resolve向けのOTIOを出力

入力ファイルは指定順に `source_1`、`source_2` として別々の話者/音声トラックになります。1ファイル内の話者分離は行いません。

## OTIO構成

音声ソースごとに、目印・字幕・音声の3トラックを生成します。正常区間は字幕と音声、カット候補区間は目印・字幕・音声の3要素を `metadata.Resolve_OTIO.Link Group ID` で相互リンクします。候補クリップには判定理由と赤ラベル相当のメタデータを付与します。

音声トラックは先頭から末尾まで保持します。字幕間、先頭、末尾のギャップもOTIOの `Gap` または音声スライスとして出力し、カット候補を検出しても元の時間軸を短縮しません。

Whisperモデルなしの固定データ検証は `process_segments` または `process_sources` から実行できます。単体テストではOTIOの再読込、トラック構造、リンクID、音声の連続性、SudachiPyによる分割を検証します。

## 既知の制約

- 初期互換性対象はDaVinci Resolve、固定タイムベースの既定値は30fpsです。
- `Resolve_OTIO` のリンク動作、ラベル表示、マーカー表示はResolveのバージョンによって実機確認が必要です。
- 音声は物理的に削除せず、字幕境界に合わせたOTIO上のスライスとして保持します。
- 異なるサンプルレートやチャンネル数の音声をミックス・変換する処理は行いません。
