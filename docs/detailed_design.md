# 2. システム詳細設計書

## 2.1 全体処理フロー

```text
[ 音声ファイル群 (.wav / .mp3) ]
             │
             ▼
┌───────────────────────────────────────┐
│  1. Transcriber                       │ Faster-Whisper / OpenAI-Whisper
│     (文字起こし・単語タイムスタンプ取得)│ (`word_timestamps=True`)
└────────────┬──────────────────────────┘
             │ Raw Segments + Word Timestamps
             ▼
┌───────────────────────────────────────┐
│  2. Segmentator (NLP分割)             │ SudachiPy / MeCab等による形態素解析
│     (文字数・表示時間・文節単位の再分割)│ 単語タイムスタンプに基づく時間の再割り当て
└────────────┬──────────────────────────┘
             │ Refined Segments
             ▼
┌───────────────────────────────────────┐
│  3. Evaluator                         │ ポーズ/無音・フィラー・雑音等の
│     (カット候補判定 & マーカー付与)   │ ルールベースフラグ判定
└────────────┬──────────────────────────┘
             │ Processed Segments
             ▼
┌───────────────────────────────────────┐
│  4. OTIO Generator                    │ 話者ごとに3層トラック構築 &
│     (OTIOノード構築 & Link Group ID) │ `Resolve_OTIO` メタデータ埋め込み
└────────────┬──────────────────────────┘
             │
             ▼
     [ OTIO 出力 (.otio) ]

```

---

## 2.2 モジュール詳細設計

### ① Transcriber (音声認識)

* **処理内容:** 音声ファイルを読み込み、単語レベルのタイムスタンプを含むテキストデータを抽出。
* **主要設定:** `word_timestamps=True`, `language="ja"`, `hallucination_silence_threshold=0.6`

### ② Segmentator (NLP字幕分割)

* **処理内容:** 抽出された単語列を形態素解析（SudachiPy等）し、適切な字幕表示単位に再分割。
* **分割ルール:**
* **文字数・時間制限:** 1セグメント最大 $20$ 文字前後 / 最大表示時間 $3.0\text{秒} \sim 4.0\text{秒}$。
* **境界判定:** 句読点（`、` `。`）や接続助詞で分割。複合名詞や「名詞＋助詞」の途中での分割は禁止。
* **時間算出:** 分割後の最初の単語の `start` と最後の単語の `end` をセグメントの開始・終了時間に設定。
* **初期実装:** SudachiPyの形態素境界を使い、句読点を優先しながら名詞の直後での分割を避ける。

### ③ Evaluator (ルール判定)

* **処理内容:** タイムライン軸の短縮は行わず、ムダな区間にカット候補フラグを付与。
* **判定条件:**
* **無音・ポーズ:** 間隔が $0.4\text{秒}$ 超のギャップ。
* **フィラー:** 辞書（`["えーと", "あのー", "んー", "まー", "えっ"]`）または正規表現一致。
* **雑音/低信頼度:** `no_speech_prob > 0.8` のセグメント。

### ④ OTIO Generator (OTIO構築)

* **処理内容:** `opentimelineio` Pythonライブラリを使用してOTIOスキーマを構築。

---

## 2.3 トラック構成および Resolve_OTIO リンク仕様

入力は1ファイル=1話者/1音声ソースとし、話者ごとに目印・字幕・音声の3トラックを生成する。話者分離モデルは使用しない。

### トラック構成（話者 $N$ あたり 3 レイヤー）

| トラック種別 | トラック種別 (OTIO) | トラック名 | 役割・構造仕様 |
| --- | --- | --- | --- |
| **Video Track $2N$** | `TrackKind.Video` | **目印用トラック** | カット候補時のみ `[要カット:理由]` クリップを配置。マーカーを包含。 |
| **Video Track $2N-1$** | `TrackKind.Video` | **本番字幕トラック** | 画面表示用のクリーンな字幕テキスト。 |
| **Audio Track $N$** | `TrackKind.Audio` | **音声トラック** | 元音声をカットせず保持。字幕境界（`start` / `end`）に合わせてスライス配置。

 |

### 自動リンク (`Link Group ID`) 付与ロジック

同一セグメント $i$ に属する **「目印クリップ」「本番字幕クリップ」「スライス音声クリップ」** の3つの `otio.schema.Clip` に対し、共通の連動ID（整数値）を以下の形式でメタデータに埋め込む。

```python
link_group_id = segment_index  # 1, 2, 3...

# 3要素すべての metadata に同一の Link Group ID を付与
clip.metadata["Resolve_OTIO"] = {
    "Link Group ID": link_group_id
}

```

---

## 2.4 データ構造設計 (Data Models)

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float

@dataclass
class RawSegment:
    id: int
    start: float
    end: float
    text: str
    no_speech_prob: float
    words: List[WordTimestamp] = field(default_factory=list)

@dataclass
class RefinedSegment:
    segment_id: str
    track_index: int
    start: float
    end: float
    text: str
    no_speech_prob: float
    words: List[WordTimestamp] = field(default_factory=list)

@dataclass
class ProcessedSegment:
    segment_id: str
    track_index: int
    start_time: float           # 開始時間（秒）
    end_time: float             # 終了時間（秒）
    subtitle_text: str          # 本番字幕テキスト
    is_cut_candidate: bool = False
    flag_label: str = ""        # 例: "[要カット: フィラー]"
    marker_reason: str = ""     # 例: "判定: フィラー (えーと)"
    link_group_id: int = 0      # DaVinci連携用リンクグループID

```

---

## 2.5 出力 OTIO 構造サンプル（データイメージ）

```json
{
  "OTIO_SCHEMA": "Timeline.1",
  "metadata": {
    "Resolve_OTIO": { "Resolve OTIO Meta Version": "1.0" }
  },
  "name": "Podcast_Timeline",
  "global_start_time": { "OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 0.0 },
  "tracks": {
    "OTIO_SCHEMA": "Stack.1",
    "children": [
      {
        "OTIO_SCHEMA": "Track.1",
        "name": "目印トラック",
        "kind": "Video",
        "children": [
          {
            "OTIO_SCHEMA": "Clip.2",
            "name": "[要カット: フィラー]",
            "metadata": {
              "Resolve_OTIO": { "Link Group ID": 1 }
            },
            "source_range": {
              "OTIO_SCHEMA": "TimeRange.1",
              "start_time": { "OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 150.0 },
              "duration": { "OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 60.0 }
            }
          }
        ]
      },
      {
        "OTIO_SCHEMA": "Track.1",
        "name": "字幕トラック",
        "kind": "Video",
        "children": [
          {
            "OTIO_SCHEMA": "Clip.2",
            "name": "えーと、本題に入ります",
            "metadata": {
              "Resolve_OTIO": { "Link Group ID": 1 }
            },
            "source_range": {
              "OTIO_SCHEMA": "TimeRange.1",
              "start_time": { "OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 150.0 },
              "duration": { "OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 60.0 }
            }
          }
        ]
      },
      {
        "OTIO_SCHEMA": "Track.1",
        "name": "音声トラック",
        "kind": "Audio",
        "children": [
          {
            "OTIO_SCHEMA": "Clip.2",
            "name": "output.wav",
            "metadata": {
              "Resolve_OTIO": {
                "Link Group ID": 1,
                "Channels": [
                  { "Source Channel ID": 0, "Source Track ID": 0 },
                  { "Source Channel ID": 1, "Source Track ID": 0 }
                ]
              }
            },
            "source_range": {
              "OTIO_SCHEMA": "TimeRange.1",
              "start_time": { "OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 150.0 },
              "duration": { "OTIO_SCHEMA": "RationalTime.1", "rate": 30.0, "value": 60.0 }
            },
            "media_references": {
              "DEFAULT_MEDIA": {
                "OTIO_SCHEMA": "ExternalReference.1",
                "target_url": "C:\\Users\\yamat\\Downloads\\output.wav"
              }
            }
          }
        ]
      }
    ]
  }
}

```
