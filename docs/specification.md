# Podcast Preprocessor Specification

## 1. アプリケーション概要・選定規格

* **目的:** 全音声および文字起こしテキストをカットせずに保持しつつ、判定されたムダ区間に視覚的な目印を付与し、1編集アクションで「目印・字幕・音声」を一括操作できる環境を構築する。
* **採用規格:** **FCP7 XML (xmeml)**
* **選定理由:** DaVinci Resolve（無償版含む）やPremiere Proにインポートした際、`<link>` タグにより「目印」「本番字幕」「音声」の3要素を同一タイムコード上で完璧にグループ化（同期）できるため。

---

## 2. トラック構成（話者 $N$ あたり3レイヤー構成）

各話者（または音声ソース）ごとに、ビデオ2トラック・オーディオ1トラックの計3トラックを展開する。

| トラック種別 | トラック名 | 役割・内容 |
| --- | --- | --- |
| **Video Track $2N$** | **目印用トラック (Flag Track)** | ムダ区間と判定された箇所のみ `[要カット:理由]` クリップを配置。不要時はトラック非表示/削除で一括除去可能。 |
| **Video Track $2N-1$** | **本番字幕トラック (Subtitle Track)** | 画面表示用の純粋な文字起こしテキスト（`[要カット]` などの目印文字は含めない）。 |
| **Audio Track $N$** | **音声トラック (Audio Track)** | 元音声ファイル（全区間をカットせず保持）。 |

---

## 3. 目印・フラグの付与仕様

ムダな区間（無音・フィラー・言い間違い・雑音）と判定されたセグメントには、以下の3重の目印を適用する。

1. **目印専用トラックへのクリップ配置:**

* 例: `[要カット: フィラー]` / `[要カット: 無音 1.2秒]`

1. **クリップカラー（Color Label）の指定:**

* 赤系ラベル（`<labels><label2>Red</label2></labels>`）を付与し、タイムライン上で目立たせる。

1. **マーカー（Marker）の埋め込み:**

* コメント欄に判定詳細（`no_speech_prob: 0.88`, `辞書一致: えーと`）を保持。

---

## 4. クリップの相互リンク（グループ化）法則

同一セグメント $i$ における **「目印クリップ」「本番字幕クリップ」「音声クリップ」の3要素** を相互に `<link>` タグで紐付ける。

* **効果:** 編集ソフト（DaVinci Resolve / Premiere Pro）上でどれか1つのクリップを選択して削除・移動すると、連動して3つのクリップがまとめて操作される。

---

## 5. FCP7 XML 構造サンプル（1話者・1セグメント分）

```xml
<xmeml version="5">
  <sequence>
    <name>Whisper_Inspection_Timeline</name>
    <rate><timebase>30</timebase><ntsc>FALSE</ntsc></rate>
    <media>
      <video>
        <!-- Track 2: 目印専用トラック -->
        <track>
          <clip id="flag_t1_c1">
            <name>[要カット: フィラー]</name>
            <start>150</start><end>210</end>
            <in>0</in><out>60</out>
            <labels><label2>Red</label2></labels>
            <marker>
              <name>カット候補</name>
              <comment>判定: フィラー (えーと)</comment>
              <in>0</in><out>0</out>
            </marker>
            <!-- 3要素相互リンク -->
            <link><linkclipid>flag_t1_c1</linkclipid><mediatype>video</mediatype><trackindex>2</trackindex><clipitemid>flag_t1_c1</clipitemid></link>
            <link><linkclipid>sub_t1_c1</linkclipid><mediatype>video</mediatype><trackindex>1</trackindex><clipitemid>sub_t1_c1</clipitemid></link>
            <link><linkclipid>audio_t1_c1</linkclipid><mediatype>audio</mediatype><trackindex>1</trackindex><clipitemid>audio_t1_c1</clipitemid></link>
          </clip>
        </track>

        <!-- Track 1: 本番字幕トラック（クリーンテキスト） -->
        <track>
          <clip id="sub_t1_c1">
            <name>えーと、本題に入ります</name>
            <start>150</start><end>210</end>
            <in>0</in><out>60</out>
            <link><linkclipid>flag_t1_c1</linkclipid><mediatype>video</mediatype><trackindex>2</trackindex><clipitemid>flag_t1_c1</clipitemid></link>
            <link><linkclipid>sub_t1_c1</linkclipid><mediatype>video</mediatype><trackindex>1</trackindex><clipitemid>sub_t1_c1</clipitemid></link>
            <link><linkclipid>audio_t1_c1</linkclipid><mediatype>audio</mediatype><trackindex>1</trackindex><clipitemid>audio_t1_c1</clipitemid></link>
          </clip>
        </track>
      </video>

      <audio>
        <!-- Track 1: 音声トラック -->
        <track>
          <clip id="audio_t1_c1">
            <name>話者A_音声_01</name>
            <start>150</start><end>210</end>
            <in>150</in><out>210</out>
            <file id="file_audio_t1">
              <pathurl>file://localhost/path/to/speaker_a.wav</pathurl>
            </file>
            <link><linkclipid>flag_t1_c1</linkclipid><mediatype>video</mediatype><trackindex>2</trackindex><clipitemid>flag_t1_c1</clipitemid></link>
            <link><linkclipid>sub_t1_c1</linkclipid><mediatype>video</mediatype><trackindex>1</trackindex><clipitemid>sub_t1_c1</clipitemid></link>
            <link><linkclipid>audio_t1_c1</linkclipid><mediatype>audio</mediatype><trackindex>1</trackindex><clipitemid>audio_t1_c1</clipitemid></link>
          </clip>
        </track>
      </audio>
    </media>
  </sequence>
</xmeml>

```
