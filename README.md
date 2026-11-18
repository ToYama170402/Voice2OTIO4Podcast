# PodcastPreprocessor（Voice2OTIO4Podcast）

日本語ポッドキャストの音声を文字起こしし、**字幕と「カット候補」の目印が付いたタイムライン**を生成するツールです。

出力はOpenTimelineIO（`.otio`）形式で、DaVinci Resolveに読み込んでそのまま編集を始められます。音声は切り取らず元の時間軸をすべて保持し、無音・フィラー・言い直しなどの「切ってもよさそうな箇所」には目印を付けるだけなので、最終判断は編集者が行えます。

## 使い方の流れ

1. 話者ごとに分かれた音声ファイルを用意する（1ファイル = 1話者）
2. このツールを実行し、`.otio` と `.srt` を生成する
3. DaVinci Resolveに `.otio` をインポートする
4. 「目印トラック」にある `[要カット: 理由]` のクリップを目安に、対応する音声スライスを削除していく

## クイックスタート

必要なもの: Python 3.10以上と [uv](https://docs.astral.sh/uv/)。

```bash
# インストール（faster-whisperバックエンド）
uv sync

# 実行（話者1人）
uv run podcast-preprocessor speaker_a.wav podcast.otio
```

初回実行時に文字起こしモデルがダウンロードされます。同じ場所に字幕ファイル `podcast.srt` も生成されます。

話者が2人の場合:

```bash
uv run podcast-preprocessor speaker_a.wav speaker_b.wav podcast.otio
```

CPUで実行する場合は、軽量な設定を指定してください。

```bash
uv run podcast-preprocessor speaker_a.wav podcast.otio --model small --device cpu --compute-type int8
```

SenseVoiceバックエンドを使う場合は、追加依存をインストールして `--backend sensevoice` を指定します。

```bash
uv sync --extra sensevoice
uv run podcast-preprocessor speaker_a.wav podcast.otio --backend sensevoice
```

オプションの一覧は `uv run podcast-preprocessor --help` で確認できます。詳しい使い方は [docs/usage.md](docs/usage.md) を参照してください。

## 出力の構成

音声ファイルごとに、次の3トラックが生成されます。

| トラック | 内容 |
| --- | --- |
| 目印トラック | カット候補の箇所にだけ `[要カット: 理由]` クリップを配置 |
| 字幕トラック | 文節境界で分割された字幕 |
| 音声トラック | 元音声を字幕境界でスライスしたもの（全時間を保持） |

同じ箇所の3要素は `Resolve_OTIO.Link Group ID` で相互リンクされるため、Resolve上で連動して選択・削除できます。カット候補には判定理由（例: `判定: フィラー (えーと)`）と赤ラベル相当のメタデータが付きます。

## 何をカット候補にするか

- **無音**: 0.4秒を超える間隔
- **フィラー**: 「えーと」「あのー」など。字幕分割の段階で独立したスライスになるため、前後の発話を巻き込まずに切れます
- **低信頼度**: `no_speech_prob > 0.8` の区間
- **言い直し**: 直後の発話に言い直された発話と、発話冒頭の繰り返し。トークン列（単語タイムスタンプ）から試行の範囲を特定し、試行だけを独立スライスにします

`--acoustic-fillers` を付けると、音声信号からもフィラー候補を検出し、音声クリップに黄色マーカーで付けます。SenseVoiceのようにフィラーを文字起こしへ残さないバックエンドでも候補を拾えます（精度はテキストベースより劣るため、自動スライスにはせずマーカー提示のみです）。

## バックエンドの選び方

| | faster-whisper（既定） | SenseVoice |
| --- | --- | --- |
| 特徴 | 精度を調整しやすい。GPU対応 | 高速で軽量。日本語に強い |
| フィラー | 文字起こしに残る | 文字起こしから落ちる（`--acoustic-fillers` の併用がおすすめ） |
| 追加依存 | なし | `uv sync --extra sensevoice` |
| 向いている用途 | 迷ったらこちら | 長時間の音声を速く処理したいとき |

## 主な機能

- **字幕分割**: SudachiPyの文節境界でのみ分割し、単語や文節の途中で切らない（最大20文字 / 最大4秒が目安）
- **キャッシュと再開**: 文字起こし結果はファイルごとにキャッシュされ、途中失敗しても再実行で続きから処理できます。SenseVoiceではチャンク（既定30秒）単位の途中経過も保存されます
- **音声変換（SenseVoice時）**: 入力音声は16kHzモノラルWAVへ変換されてから推論されるため、m4aなども扱えます。変換結果はキャッシュされます
- **VAD（SenseVoice時）**: `--sensevoice-vad` でSilero VADが発話区間だけを推論し、無音の多い音声を高速化します
- **進捗表示**: 処理中は全体の進捗率を標準エラーへ表示します（端末接続時のみ）

## ドキュメント

- [docs/usage.md](docs/usage.md) — 使い方の詳細、全オプション、レシピ、トラブルシューティング
- [docs/specification.md](docs/specification.md) — システム仕様書
- [docs/detailed_design.md](docs/detailed_design.md) — 詳細設計書（モジュール構成と判定ルール）

## 既知の制約

- 対応編集ソフトはDaVinci Resolveを想定しています。`Resolve_OTIO` のリンク動作やラベル・マーカー表示は、Resolveのバージョンによって実機確認が必要です
- 1ファイル内の話者分離は行いません。話者はファイル単位で分けてください
- 音声は物理的に削除しません。削除はResolve上で行います
- 文字起こしバックエンドは単語タイムスタンプを返す必要があります。単語タイムスタンプを返せないSenseVoice推論経路は利用できません
- SenseVoiceのモデルはsherpa-onnxの公式releaseから取得し、SHA256を検証します。モデルのライセンスと取得条件は配布元を確認してください

## 開発

```bash
uv sync --extra dev --extra sensevoice
uv run pytest
```

CIと同じチェックは手元でも回せます（PRではすべて必須です）。

```bash
uv run ruff format --check .   # フォーマット
uv run ruff check .            # リント
uv run mypy src/podcastpreprocessor  # 型チェック（strict）
uv run pytest                  # テスト＋カバレッジ（下限80%）
uv run mutmut run              # ミューテーションテスト（週次CIでも実行）
```

- テストにはHypothesisによるプロパティベーステスト（`tests/test_properties.py`）を含みます
- 依存関係の脆弱性チェック（pip-audit）と、ビルドしたwheelのクリーンインストール確認もCIで回しています
