import os
import sys
import torch
import numpy as np
from PIL import Image

# GUI表示（ポップアップウィンドウ）のための設定
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- パス問題を強制的に解決するおまじない ---
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 必要なSAM2の関数をインポート ---
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.vis import show_mask, show_points

# --- メインの処理開始 ---
print(">>> プログラムを開始します。")

try:
    # 1. デバイス設定
    print("--- デバイスを設定しています ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 2. モデルのロード（相対パスを使用）
    print("--- SAM2モデルを読み込んでいます（数秒かかります）---")
    sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
    config_name = "configs/sam2.1/sam2_1_hiera_b+.yaml"
    sam2 = build_sam2(config_name, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2)

    # 3. 画像の読み込み
    print("--- 画像を読み込んでいます ---")
    image_path = "notebooks/images/truck.jpg"
    image_pil = Image.open(image_path).convert("RGB")
    image = np.array(image_pil)
    predictor.set_image(image)
    print("画像のセットが完了しました。")

    # 4. 1点のプロンプトでマスクを予測
    print("--- 1点のプロンプトでマスクを予測しています ---")
    point_coords = np.array([[400, 320]])  # トラックの荷台あたり
    point_labels = np.array([1])
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    single_mask = masks[0]

    # 5. 結果をシンプルに描画
    print("--- 結果を画像に描画しています ---")
    overlay = image.copy()
    color = np.array([255, 0, 0], dtype=np.uint8)  # 赤色
    overlay[single_mask > 0] = (overlay[single_mask > 0] * 0.5 + color * 0.5).astype(np.uint8)
    
    # 6. 結果のウィンドウを表示
    print(">>> 成功！結果のウィンドウを表示します。")
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Truck Segmentation")
    plt.show(block=True)  # ウィンドウが閉じられるまで待機

    print("\n>>> 全ての処理が正常に完了しました！お疲れ様でした！")

except Exception as e:
    print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"予期せぬエラーが発生しました。")
    print(f"エラー詳細: {e}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    import traceback
    traceback.print_exc()