# PodcastPreprocessor

日本語ポッドキャスト音声を文字起こしし、字幕・カット候補フラグ・音声をリンクしたFCP7 XMLを生成します。

## セットアップ

```powershell
uv sync --extra dev
```

faster-whisperのモデルは初回実行時に取得されます。GPUを使わない場合はCLIに`--device cpu`を指定してください。

## 実行

```powershell
uv run podcast-preprocessor input.wav output.xml --model small --device cpu --fps 30
```

処理は次の順で進みます。

1. faster-whisperで日本語音声を単語タイムスタンプ付きで認識
2. 句読点、文字数、表示時間を基準に字幕を分割
3. 0.4秒超の無音、フィラー、`no_speech_prob > 0.8`をカット候補として判定
4. DaVinci Resolve向けのFCP7 XMLを出力

## XML構成

話者または音声ソースごとに、字幕トラック、フラグトラック、音声トラックを生成します。正常区間は字幕と音声、カット候補区間はフラグ・字幕・音声の3要素を相互リンクします。候補フラグには赤ラベルと判定理由のマーカーを付与します。

固定データを使ったXML生成は、Whisperモデルなしでも`process_segments`から実行できます。単体テストではXMLの整形式、トラック数、リンクID、判定境界、単語タイムスタンプ保持を検証します。

## 既知の制約

- 初期互換性対象はDaVinci Resolve、固定タイムベースの既定値は30fpsです。
- XMLのリンク動作とラベル表示は、Resolveのバージョンによって実機確認が必要です。
- 音声は物理的に削除せず、字幕境界に合わせたXML上のスライスとして保持します。
