# システム詳細設計書：Whisper連携マルチトラックXML自動生成システム (WhisperToFcpXml Pipeline)

## 1. システム概要および処理フロー

本システムは、入力されたマルチトラック音声ファイル群に対して音声認識、NLPに基づく字幕単位の最適分割、カット候補判定、およびFCP7 XML（xmeml v5）形式のタイムライン出力を行うエンドツーエンドの自動化パイプラインである。

ムダな区間（フィラー・無音・雑音等）を物理的に除去・短縮（リップルカット）せず時間軸全体を維持したまま、字幕セグメントの区切りに合わせて音声ファイルをスライス（分割）配置し、編集ソフト上で一括操作可能なリンク構造を構築する。

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
│  4. FCP7 XML Generator                │ 話者ごとに3層トラック構築 &
│     (音声スライス配置 & 3要素リンク)  │ 相互<link>タグ埋め込み
└────────────┬──────────────────────────┘
             │
             ▼
     [ FCP7 XML 出力 (.xml) ]

```

## 2. モジュール仕様

### 2.1 Transcriber (音声認識モジュール)

* **役割:** 音声ファイルを読み込み、単語レベルのタイムスタンプを含むテキストデータを抽出。
* **主要設定:** `word_timestamps=True`, `language="ja"`, `hallucination_silence_threshold=0.6`

### 2.2 Segmentator (NLP字幕分割モジュール)

* **役割:** Whisperの出力を形態素解析し、人間が読みやすい適正な字幕単位へ再分割。
* **分割ルール:**
* **制限上限:** 1セグメント最大 $20$ 文字前後 / 最大表示時間 $3.0\text{秒} \sim 4.0\text{秒}$。
* **境界判定:** 句読点（`、` `。`）、接続助詞で優先分割。複合名詞・接頭辞/接尾辞の途中での分割は禁止。
* **時間算出:** 最初と最後の単語タイムスタンプから開始（`start`）・終了（`end`）時間を正確に算出。

### 2.3 Evaluator (ルール判定モジュール)

* **役割:** 時間軸を短縮せず保持したまま、カットすべきムダ区間にフラグ・目印を付与。
* **判定ルール:**
* **無音・ギャップ:** 前後間隔が $0.4\text{秒}$ 超のポーズ検出。
* **フィラー:** 辞書一致（`["えーと", "あのー", "んー", "まー", "えっ"]`）または正規表現判定。
* **雑音・非音声:** `no_speech_prob > 0.8` のセグメント。

### 2.4 FCP7 XML Generator (XML構築モジュール)

* **役割:** DaVinci Resolve / Premiere Pro向けのFCP7 XMLを生成。

---

## 3. トラック構造および音声スライス・リンク仕様

### 3.1 3層トラック構造（話者 $N$ あたり）

話者（または音声ソース）ごとに、ビデオ2トラック・オーディオ1トラックの計3トラックを展開する。

| トラック種別 | トラック名 | 役割・構造仕様 |
| --- | --- | --- |
| **Video Track $2N$** | **目印用トラック (Flag Track)** | カット候補時のみ `[要カット:理由]` クリップを配置。赤色ラベル（`<label2>Red</label2>`）および詳細理由のマーカーを包含。正常区間は空欄。 |
| **Video Track $2N-1$** | **本番字幕トラック (Subtitle Track)** | 画面表示用のクリーンな字幕テキスト（目印文字は一切含めない）。 |
| **Audio Track $N$** | **音声トラック (Audio Track)** | 元音声を時間軸上で削除・短縮せず全区間維持。ただし、**字幕の区切り（`in` / `out`）と完全に一致する境界で細分化（スライス）された個別の音声クリップ**を隙間なく並べる。 |

### 3.2 3要素相互リンク (`<link>`) 法則

同一セグメント $i$ における **「目印クリップ」「本番字幕クリップ」「スライスされた音声クリップ」** の3要素の内部に、互いのIDを指す同一の `<link>` ブロックを埋め込む。

* **効果:** 編集ソフト上で1つのクリップを選択・削除・移動すると、連動して3つのクリップがまとめて操作される。

---

## 4. データ構造設計 (Data Models)

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
    start_frame: int            # タイムライン上開始位置
    end_frame: int              # タイムライン上終了位置
    in_frame: int               # ソース音声内切り出し開始位置
    out_frame: int              # ソース音声内切り出し終了位置
    subtitle_text: str          # 本番字幕テキスト
    is_cut_candidate: bool = False
    flag_label: str = ""        # 例: "[要カット: フィラー]"
    marker_reason: str = ""     # 例: "判定: フィラー (えーと)"
    color_label: str = "None"   # カット候補時は "Red"

```

---

## 5. 出力 XML 構造サンプル（1話者・2セグメント分）

※音声ファイル `file_audio_t1` が字幕の境界に合わせてスライスされ、目印・字幕と相互リンクされている構造を示す。

```xml
<xmeml version="5">
  <sequence>
    <name>Whisper_Pipeline_Timeline</name>
    <rate><timebase>30</timebase><ntsc>FALSE</ntsc></rate>
    <media>
      <video>
        <!-- Track 2: 目印専用トラック -->
        <track>
          <!-- セグメント1: 正常区間（目印なし・空欄） -->

          <!-- セグメント2: 要カット区間 -->
          <clip id="flag_t1_c2">
            <name>[要カット: フィラー]</name>
            <start>150</start><end>210</end>
            <in>0</in><out>60</out>
            <labels><label2>Red</label2></labels>
            <marker>
              <name>カット候補</name>
              <comment>判定: フィラー (えーと)</comment>
              <in>0</in><out>0</out>
            </marker>
            <link><linkclipid>flag_t1_c2</linkclipid><mediatype>video</mediatype><trackindex>2</trackindex><clipitemid>flag_t1_c2</clipitemid></link>
            <link><linkclipid>sub_t1_c2</linkclipid><mediatype>video</mediatype><trackindex>1</trackindex><clipitemid>sub_t1_c2</clipitemid></link>
            <link><linkclipid>audio_t1_c2</linkclipid><mediatype>audio</mediatype><trackindex>1</trackindex><clipitemid>audio_t1_c2</clipitemid></link>
          </clip>
        </track>

        <!-- Track 1: 本番字幕トラック -->
        <track>
          <clip id="sub_t1_c1">
            <name>こんにちは</name>
            <start>0</start><end>150</end>
            <in>0</in><out>150</out>
            <link><linkclipid>sub_t1_c1</linkclipid><mediatype>video</mediatype><trackindex>1</trackindex><clipitemid>sub_t1_c1</clipitemid></link>
            <link><linkclipid>audio_t1_c1</linkclipid><mediatype>audio</mediatype><trackindex>1</trackindex><clipitemid>audio_t1_c1</clipitemid></link>
          </clip>
          <clip id="sub_t1_c2">
            <name>えーと、本題に入ります</name>
            <start>150</start><end>210</end>
            <in>0</in><out>60</out>
            <link><linkclipid>flag_t1_c2</linkclipid><mediatype>video</mediatype><trackindex>2</trackindex><clipitemid>flag_t1_c2</clipitemid></link>
            <link><linkclipid>sub_t1_c2</linkclipid><mediatype>video</mediatype><trackindex>1</trackindex><clipitemid>sub_t1_c2</clipitemid></link>
            <link><linkclipid>audio_t1_c2</linkclipid><mediatype>audio</mediatype><trackindex>1</trackindex><clipitemid>audio_t1_c2</clipitemid></link>
          </clip>
        </track>
      </video>

      <audio>
        <!-- Track 1: 音声トラック（連続スライス配置） -->
        <track>
          <!-- セグメント1の音声をスライス -->
          <clip id="audio_t1_c1">
            <name>話者A_音声_01</name>
            <start>0</start><end>150</end>
            <in>0</in><out>150</out>
            <file id="file_audio_t1">
              <pathurl>file://localhost/path/to/speaker_a.wav</pathurl>
            </file>
            <link><linkclipid>sub_t1_c1</linkclipid><mediatype>video</mediatype><trackindex>1</trackindex><clipitemid>sub_t1_c1</clipitemid></link>
            <link><linkclipid>audio_t1_c1</linkclipid><mediatype>audio</mediatype><trackindex>1</trackindex><clipitemid>audio_t1_c1</clipitemid></link>
          </clip>

          <!-- セグメント2の音声をスライス（目印2・字幕2と相互リンク） -->
          <clip id="audio_t1_c2">
            <name>話者A_音声_02</name>
            <start>150</start><end>210</end>
            <in>150</in><out>210</out>
            <file id="file_audio_t1" />
            <link><linkclipid>flag_t1_c2</linkclipid><mediatype>video</mediatype><trackindex>2</trackindex><clipitemid>flag_t1_c2</clipitemid></link>
            <link><linkclipid>sub_t1_c2</linkclipid><mediatype>video</mediatype><trackindex>1</trackindex><clipitemid>sub_t1_c2</clipitemid></link>
            <link><linkclipid>audio_t1_c2</linkclipid><mediatype>audio</mediatype><trackindex>1</trackindex><clipitemid>audio_t1_c2</clipitemid></link>
          </clip>
        </track>
      </audio>
    </media>
  </sequence>
</xmeml>

```
