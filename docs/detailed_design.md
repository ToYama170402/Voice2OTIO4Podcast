# システム詳細設計書

## 1. 全体処理フロー

```text
[ 音声ファイル群（話者ごとに1ファイル） ]
             │
             ▼
┌───────────────────────────────────────┐
│  1. Transcriber                       │ faster-whisper / SenseVoiceSmall
│     文字起こし・単語タイムスタンプ取得  │ 結果はファイル単位でキャッシュ
└────────────┬──────────────────────────┘
             │ RawSegment + WordTimestamp
             ▼
┌───────────────────────────────────────┐
│  2. Segmentator                       │ SudachiPyの文節境界でのみ分割
│     字幕単位への再分割                 │ フィラーは独立セグメント化
└────────────┬──────────────────────────┘
             │ RefinedSegment
             ▼
┌───────────────────────────────────────┐
│  3. Evaluator / RestartDetector       │ 無音・フィラー・低信頼度・言い直し
│     カット候補判定                     │ 音響フィラーはマーカーとして別系統
└────────────┬──────────────────────────┘
             │ ProcessedSegment
             ▼
┌───────────────────────────────────────┐
│  4. OTIO Generator                    │ 話者ごとに3トラック構築
│     OTIO / SRT 出力                    │ Resolve_OTIO Link Group ID 付与
└────────────┬──────────────────────────┘
             ▼
     [ .otio / .srt ]
```

処理全体の進捗は `ProgressReporter` が標準エラーへ表示する（端末接続時のみ、フェーズごとの重み付き全体率）。

## 2. モジュール構成

すべて `src/podcastpreprocessor/` 配下。

| モジュール | 役割 |
| --- | --- |
| `cli.py` | コマンドライン引数の定義と全体の実行制御 |
| `factory.py` | バックエンド名と設定からTranscriberを構築 |
| `transcriber.py` | faster-whisperアダプター |
| `sensevoice.py` | SenseVoice（sherpa-onnx）アダプター。変換・VAD・チャンク推論・ジャーナル再開 |
| `vad.py` | Silero VADの発話区間をサンプル範囲へ変換するユーティリティとVAD設定の構築 |
| `model_manager.py` | SenseVoice / Silero VADモデルの取得（SHA256検証）とキャッシュ解決、`--offline` 対応 |
| `audio_io.py` | 入力音声を16kHzモノラルWAVへ変換（PyAVストリーミング、キャッシュ付き） |
| `transcript_cache.py` | 文字起こし結果のファイル単位キャッシュ（キー = 入力ファイル情報 + 文字起こし設定） |
| `progress.py` | フェーズ重み付きの進捗表示 |
| `segmentator.py` | 単語列全体を文節境界で字幕単位へ再分割（`segment_source` / `segment_raw`） |
| `evaluator.py` | セグメント単位のカット候補判定（無音・フィラー・低信頼度） |
| `restart_detector.py` | トークン列の走査による言い直し検出（試行区間を返す） |
| `filler_detector.py` | 音声信号からのフィラー候補検出（numpyのみ） |
| `pipeline.py` | ソースごとの処理統合（分割 → 判定 → `ProcessedSegment` 化）と入出力の実行 |
| `otio.py` | OTIOタイムラインの構築と書き出し（トラック、リンク、マーカー） |
| `timeline.py` | 音声スライス列の構築（全時間保持のタイル化） |
| `srt.py` | SRT字幕の書き出し |
| `models.py` | データ構造（次節）とフレーム換算 |

## 3. モジュール詳細

### 3.1 Transcriber（音声認識）

- 選択したバックエンドで音声ファイルを読み込み、共通の `RawSegment` と単語レベルの `WordTimestamp` を抽出する。
- 共通契約: バックエンドは `transcribe` と `transcribe_sources` を提供し、単語タイムスタンプを秒単位で返す。単語タイムスタンプを取得できない推論結果は処理を継続せずエラーにする。
- **SenseVoiceモデル**: 初回実行時に固定releaseのモデルアーカイブをダウンロードし、SHA256検証後にキャッシュ（`~/.cache/podcastpreprocessor/sensevoice/`）へ保存する。`--offline` ではキャッシュを必須とする。
- **音声変換**: SenseVoiceでは推論前に `audio_io` が入力を16kHzモノラルWAVへ変換する（PyAVストリーミング）。変換結果はキャッシュされ（既定 `~/.cache/podcastpreprocessor/converted/`）、再実行時は省略される。これによりsoundfileが直接読めないm4a等も扱える。
- **チャンクと再開**: SenseVoiceはチャンク（既定30秒）単位で推論し、途中経過をジャーナルへ保存する。再実行時は完了チャンクを再利用する。faster-whisperは生成器ベースのため、再開はファイル単位キャッシュのみ。
- **VAD**: `--sensevoice-vad` 指定時はSilero VADで発話区間を抽出し、区間を基本単位として推論する。検出区間には既定0.25秒のpaddingを付け、近接区間を結合する。長い発話だけ `--sensevoice-vad-max-speech-s`（既定120秒）で分割する。各区間の単語時刻へ元音声の開始時刻を加算し、元の時間軸を維持する。感度等は `--sensevoice-vad-threshold`、`--sensevoice-vad-min-silence-s`、`--sensevoice-vad-min-speech-s`、`--sensevoice-vad-padding-s` で調整する。VADモデルはSenseVoiceモデルと別アーティファクトとしてキャッシュ・SHA256検証・offline解決を行う。
- なお、エネルギー方式の発話区間検出（`_detect_speech_regions`、`--sensevoice-vad-threshold-db`）は現在の推論経路では使用していない（単体テストのみが参照）。

### 3.2 Segmentator（字幕分割）

- 抽出された単語列をSudachiPy（SplitMode A）で文節ユニット化し、字幕表示単位へ再分割する。
- 分割ルール:
  - **制限**: 1セグメント最大20文字 / 最大表示時間4.0秒（推奨3.5秒）。
  - **境界の保証**: 分割境界は文節の先頭位置にのみ置く。上限を超える場合は文節の途中で切らず、手前の文節境界へ境界を移す。単一の文節だけで上限を超える場合は、その文節をまるごと1セグメントにする。複合名詞（名詞の連なり）と連体詞+名詞は同一文節として扱う。
  - **時間算出**: 分割後の最初の単語の `start` と最後の単語の `end` をセグメントの開始・終了時間に設定する。
  - **フィラーの独立**: 感動詞（フィラー）とそれに続く助詞・句読点からなるフィラー句は独立したセグメントとして切り出す。OTIO上も独立スライスになるため、前後の発話を巻き込まずにカットできる。単語タイムスタンプがフィラーと通常の発話を1単語に含む場合は、その単語は独立化できない。
  - **ソース単位の分割**: 音声認識のセグメント境界は発話の切れ目とは限らず、単語や文節の途中で切れていることがある。そのため分割は生セグメントごとではなく、1ソース分の単語列全体に対して行う（`segment_source`）。各セグメントの `no_speech_prob` は先頭の単語が属していた生セグメントの値を引き継ぐ。

### 3.3 Evaluator（カット候補判定）

タイムライン軸の短縮は行わず、ムダな区間にカット候補フラグを付与する。

- **無音・ポーズ**: 直前セグメントとの間隔が0.4秒超。
- **フィラー**: セグメント文字列がフィラー（`えーと`、`あのー`、`んー`、`まー`、`えっ`）を含む、またはセグメント全体がフィラー語である。
- **雑音 / 低信頼度**: `no_speech_prob > 0.8` のセグメント。

### 3.4 言い直し検出（RestartDetector）

トークン列（単語タイムスタンプ）を直接走査し、破棄された試行の区間（`RestartRegion`）を返す。

- トークン列を発話単位に区切ってから判定する。区切りは生セグメント境界、文末記号の直後、0.4秒超のギャップ、0.8秒以上の長保持トークン（SenseVoiceがポーズを直前トークンの持続時間に吸収するため）の直後。
- パターンは2つ: (1) 直後の発話が対象の冒頭を言い直す（全文、または未完の発話で共通接頭辞4文字以上かつカバー率60%以上。間隔4秒以内、試行6秒以内）、(2) 発話冒頭の句（4文字以上）が即座に繰り返され、その後も発話が続く。
- 除外: 完結文の部分的な言い直し、相槌・同意の定型句とその繰り返し、繰り返し自体が全体をなす強調表現、フィラー単独セグメント。
- 試行の終端は、最終トークンがポーズ吸収で異常に長い場合に中央値で補正する。
- pipelineでの統合: 領域がセグメントの半分以上を占めればセグメントごと候補化し、内側に収まる場合はセグメントを分割して試行だけを候補スライスにする。既に別理由で候補のセグメントには言い直しの理由を追加する。

### 3.5 音響フィラー検出（FillerDetector）

- バックエンドがフィラーを文字起こしに残さない場合に備え、音声信号だけから候補区間を検出する。方式はエネルギー島（ポーズに挟まれた0.12〜1.0秒の有声音）+ 内部の音高・スペクトルの安定性判定で、numpyのみで動作する。
- `--acoustic-fillers` 有効時は、トークン列の走査（Sudachi単語境界でフィラー単語を抽出しトークン時刻へ写像）による候補も統合する。重なりは音響候補を優先する。
- 候補は精度がテキストベースに劣るためスライスにはせず、`ProcessedSegment.clip_markers` を経てOTIO音声クリップの黄色マーカー「フィラー候補」として提示する。

### 3.6 OTIO Generator

- `opentimelineio` でタイムラインを構築し、`.otio` と `.srt` を書き出す。
- 音声トラックは `timeline.build_audio_slices` が先頭から末尾までをタイル状に保持する。字幕間・先頭・末尾のギャップも欠落させない。

## 4. トラック構成とリンク仕様

入力は1ファイル = 1話者 / 1音声ソースとし、話者ごとに次の3トラックを生成する。話者分離モデルは使用しない。トラック名にはソース名（`source_1` など）が前置される。

| トラック | 種別 | 役割 |
| --- | --- | --- |
| 目印トラック | Video | カット候補時のみ `[要カット: 理由]` クリップを配置し、赤マーカー「カット候補」を包含する |
| 字幕トラック | Video | 字幕テキストのタイトルクリップを配置 |
| 音声トラック | Audio | 元音声をカットせず保持し、字幕境界に合わせてスライス配置。音響フィラー候補のクリップマーカーもここに付く |

### 自動リンク（Link Group ID）

同一セグメントに属する目印・字幕・音声の各クリップには、共通の連番IDをメタデータへ埋め込む。

```python
clip.metadata["Resolve_OTIO"] = {"Link Group ID": link_group_id}
```

正常区間は字幕・音声の2要素、カット候補区間は目印・字幕・音声の3要素が同じIDを共有する。候補クリップのメタデータには `cut_candidate: true`、`marker_reason`（判定理由）、`color_label`（`Red`）も付与する。

## 5. データ構造

```python
@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start: float  # 秒
    end: float  # 秒
    probability: float


@dataclass
class RawSegment:  # バックエンドの生出力
    id: int
    start: float
    end: float
    text: str
    no_speech_prob: float
    words: list[WordTimestamp]
    source_id: str = "source_1"


@dataclass
class RefinedSegment:  # 字幕分割後
    segment_id: str
    track_index: int
    start: float
    end: float
    text: str
    no_speech_prob: float
    words: list[WordTimestamp]
    source_id: str = "source_1"


@dataclass(frozen=True)
class ClipMarker:  # 音声クリップ内のマーカー（クリップ先頭からのフレーム）
    name: str
    start_frame: int
    end_frame: int


@dataclass
class ProcessedSegment:  # 判定後、OTIO生成の入力
    segment_id: str
    track_index: int
    start_frame: int
    end_frame: int
    in_frame: int
    out_frame: int
    subtitle_text: str
    is_cut_candidate: bool = False
    flag_label: str = ""  # 例: "[要カット: フィラー (えーと)]"
    marker_reason: str = ""  # 例: "判定: フィラー (えーと)"
    color_label: str = "None"  # 候補は "Red"
    source_id: str = "source_1"
    audio_path: str = ""
    link_group_id: int = 0
    clip_markers: list[ClipMarker] = []
```

## 6. 出力OTIOの例（抜粋）

カット候補1件分のみを抜粋。実際は話者ごとに3トラックが並び、全セグメントが時系列に配置される。

```json
{
  "OTIO_SCHEMA": "Timeline.1",
  "name": "Podcast_Timeline",
  "tracks": {
    "OTIO_SCHEMA": "Stack.1",
    "children": [
      {
        "OTIO_SCHEMA": "Track.1",
        "name": "source_1 目印トラック",
        "kind": "Video",
        "children": [
          {
            "OTIO_SCHEMA": "Clip.2",
            "name": "[要カット: 言い直し (その日の夜)]",
            "metadata": {
              "Resolve_OTIO": { "Link Group ID": 130 },
              "cut_candidate": true,
              "marker_reason": "判定: 言い直し (その日の夜)",
              "color_label": "Red"
            },
            "markers": [
              { "OTIO_SCHEMA": "Marker.2", "name": "カット候補", "color": "RED" }
            ]
          }
        ]
      },
      {
        "OTIO_SCHEMA": "Track.1",
        "name": "source_1 字幕トラック",
        "kind": "Video",
        "children": [
          {
            "OTIO_SCHEMA": "Clip.2",
            "name": "その日の夜は",
            "metadata": { "Resolve_OTIO": { "Link Group ID": 130 } }
          }
        ]
      },
      {
        "OTIO_SCHEMA": "Track.1",
        "name": "source_1 音声トラック",
        "kind": "Audio",
        "children": [
          {
            "OTIO_SCHEMA": "Clip.2",
            "name": "output_0_g5b5.m4a",
            "metadata": { "Resolve_OTIO": { "Link Group ID": 130 } },
            "media_references": {
              "DEFAULT_MEDIA": {
                "OTIO_SCHEMA": "ExternalReference.1",
                "target_url": "output_0_g5b5.m4a"
              }
            }
          }
        ]
      }
    ]
  }
}
```
