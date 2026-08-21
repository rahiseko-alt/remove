# Remove - 漫画コマ キャラクター個別分離＆アニメ連携パイプライン

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

漫画の1コマ画像から、複数キャラクター（例: 五条、花御など）をピンポイントで個別に切り抜き、アルファ透過PNG化・裏側背景の穴埋め補完（Clean Plate）・アニメーションツール（Cartoon Animator 5 / After Effects / Spine 2D）連携用マニフェストを一括生成するWeb/CLIツールです。

---

## 🎯 主な機能
1. **Promptable / Bounding Box によるキャラ別高精度切り抜き**:
   - 全体背景透過（Rembg等）では不可能な「同じコマ内の五条だけ / 花御だけ」を個別指定して分離。
2. **背景クリーンプレート（Clean Plate）自動生成**:
   - キャラを切り抜いた後の背後背景をインペインティング（Inpainting）で自動穴埋め補完。
3. **アニメーションツール連携 (CTA5 / AE / Spine)**:
   - 透過レイヤー構造および座標情報マニフェスト（JSON）を自動出力。
4. **直感的な Web UI ＆ 即座に試せる Fast Demo**:
   - ブラウザ上でマウスドラッグしてキャラを囲むだけの簡単操作。

---

## 🚀 クイックスタート

### 1. Docker Compose（推奨・ワンコマンド起動）
```bash
docker compose up --build
```
ブラウザで `http://localhost:8000` を開く。

### 2. ローカル直接起動 (FastAPI Web UI)
```bash
pip install -r requirements.txt
uvicorn apps.api.main:app --reload --port 8000
```
ブラウザで `http://localhost:8000` を開く。

### 3. CLI バッチ実行
```bash
python scripts/extract_characters.py \
  --image panel.jpg \
  --chars "gojo:50,50,350,550;hanami:450,50,750,550" \
  --output output/
```

---

## 📖 ガイドドキュメント
- [`docs/workflows/gui_animation_guide.md`](docs/workflows/gui_animation_guide.md): Cartoon Animator 5 / After Effects (Puppet Tool) / Spine への素材取り込み・アニメ化手順
- [`docs/workflows/manga_separation_guide.md`](docs/workflows/manga_separation_guide.md): 漫画特有の線画・トーン・集中線・吹き出し除去の実践ノウハウ
