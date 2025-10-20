# ===================================================================
# SAM2ImagePredictor を使用して、指定したオブジェクトを色分けマスクするサンプル
# 背景は黒、オブジェクトはそれぞれ別の色でマスクされ、最終的な画像1枚のみを保存します。
# ===================================================================
# cd notebooks（～\notebooks>って形でやって）

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

# SAM2の必要なモジュールをインポート
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# =========================
# 1. デバイス設定
# =========================
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# AMP 等の設定
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# =========================
# 2. モデルのロード
# =========================
sam2_checkpoint = "C:/Project_sam/sam2/checkpoints/sam2.1_hiera_large.pt"
config_name = "C:/Project_sam/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

print("Loading model...")
start_time = time.time()
sam2 = build_sam2(config_name, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2)
print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

# =========================
# 3. 画像の読み込み
# =========================
image_path = "C:/Project_sam/sam2/notebooks/images/test-img/tube (38).jpg"
image = np.array(Image.open(image_path).convert("RGB"))
print(f"Image loaded from: {image_path}")

# =========================
# 4. プロンプト（点の座標）の準備
# =========================
points_per_tube = [
    [[40, 2119],[1661, 1881]],  # 1本目のチューブ上の点
    [[65, 2188],[1748,2113]],  # 2本目のチューブ上の点
    [[90, 2247],[1881, 2188]], # 3本目のチューブ上の点
    [[1436, 2277]], # 障害物要因2
]
#point_labels = [1] * len(points_per_tube)
print(f"{len(points_per_tube)} tubes specified.")

# =========================
# 5. マスクの生成と描画
# =========================
predictor.set_image(image)

# 結果を保存するディレクトリを作成
output_dir = "C:/Project_sam/sam2/notebooks/images/point_prompt_masks/"
os.makedirs(output_dir, exist_ok=True)

# 最終的な描画用の画像を、元の画像と同じサイズで全て0（黒）の配列として作成
# これが最終的に保存される画像になります
final_display_image = np.zeros_like(image)

print("Generating masks for each point...")

for tube_points in points_per_tube:
    # ランダムな色を「チューブごと」に1回だけ生成
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    
    # 1つのチューブに属する点すべてでループ
    for point_coord in tube_points:
        # プロンプトの形式をNumPy配列に変換
        points = np.array([point_coord])
        labels = np.array([1]) # ラベルは常に1

        # 点プロンプトを与えて予測を実行
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        # 最もスコアが高いマスクを1つ選ぶ
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        # あいまいなマスクを白黒ハッキリさせる
        boolean_mask = best_mask > 0

        # マスクされた部分を、先ほど生成したチューブの色で塗りつぶす
        final_display_image[boolean_mask] = color

print("All masks generated.")

# =========================
# 6. 最終結果の保存
# =========================

# Matplotlibを使って画像を表示する準備
plt.figure(figsize=(image.shape[1]/100, image.shape[0]/100), dpi=100) # 元の画像のサイズを基にdpi=100で設定

# final_display_imageはすでに背景が黒く、チューブが色分けされている
plt.imshow(final_display_image)
plt.axis("off") # 軸の表示をオフにする

# 余計な余白なしで保存するための設定
plt.gca().set_position([0, 0, 1, 1])
plt.tight_layout(pad=0)

# 最終的な画像をファイルに保存

# 元のファイル名を取得 (例: "tube (1).jpg" -> "tube (1)")
base_filename = os.path.basename(image_path).split('.')[0]

# 元のファイル名を付けて保存 (例: "tube (1)_mask.jpg")
new_filename = f"{base_filename}_mask.jpg"
overlay_save_path = os.path.join(output_dir, new_filename)

plt.savefig(overlay_save_path, bbox_inches='tight', pad_inches=0)
print(f"Overlay image saved to: {overlay_save_path}")

# plt.show() # 必要であればウィンドウで表示