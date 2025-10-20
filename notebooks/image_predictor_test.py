#### cd notebooks（～\notebooks>って形でやって）

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ★ 先ほど保存した Automatic Mask Generator の実装をインポート
from automatic_mask_generator import SAM2AutomaticMaskGenerator

# =========================
# 1. デバイス設定
# =========================
# OMP: Error #15 対策
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# AMP 等の設定（必要に応じて）
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS."
    )

# =========================
# 2. モデルのロード
# =========================
sam2_checkpoint = "C:/Project_sam/sam2/checkpoints/sam2.1_hiera_large.pt"
config_name = "C:/Project_sam/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(config_name, sam2_checkpoint, device=device)

# =========================
# 3. 画像の読み込み
# =========================
for i in range(4, 5):
    image_path = f"C:/Project_sam/sam2/notebooks/images/test-img/tube{i}.jpg"
    image_pil = Image.open(image_path).convert("RGB")
    image = np.array(image_pil)
    start_time = time.time()

    # =========================
    # 4. SAM2AutomaticMaskGenerator のインスタンス作成
    # =========================
    # ポイントの分割数 (points_per_side) や、スコア閾値などを調整しながら試してください。
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=15,           # グリッド数。大きいほど細かくマスクが生成される
        pred_iou_thresh=0.7,          # モデルが予測するマスクの品質閾値
        stability_score_thresh=0.70,  # マスクの安定度閾値
        mask_threshold=0.1,           # ロジットをバイナリに変換する際の閾値
        box_nms_thresh=0.7,           # 同一crop内でのNMS閾値
        crop_n_layers=0,              # 追加でcropを使うかどうか。0ならcropしない
        min_mask_region_area=30000,   # 小さすぎるマスクを除外
        output_mode="binary_mask",    # 出力マスクの形式。binary_mask / uncompressed_rle / coco_rle
        multimask_output=True,        # Trueにすると1点につき複数マスク候補を出す
    )

    # =========================
    # 5. 画像全体に対して自動マスク生成
    # =========================
    # 画像全体に対して自動マスク生成
    annotations = mask_generator.generate(image)
    print(f"Mask count: {len(annotations)}")

    # オーバーレイ用に元画像のコピーを作成
    overlay = image.copy()

    # 各マスクについて処理
    for idx, ann in enumerate(annotations):
        # segmentation はすでに binary_mask (NumPy 配列) である前提
        segmentation = ann["segmentation"]

        # 二値マスク（0と255の画像）に変換
        mask = (segmentation > 0).astype(np.uint8) * 255

        # 輪郭を抽出（外側の輪郭のみ）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print(f"Mask {idx+1}: 輪郭が見つかりませんでした。")
            continue

        # 最大の輪郭を取得
        contour = max(contours, key=cv2.contourArea)

        # 回転可能な最小外接矩形（OBB）を計算
        rect = cv2.minAreaRect(contour)  # ((center_x, center_y), (width, height), angle)
        (center_x, center_y), (w, h), angle = rect

        if min(w, h) == 0:
            print(f"Mask {idx+1}: 幅または高さが0のためアスペクト比を計算できません。")
            continue

        # 幅と高さの大きい方で割ってアスペクト比を計算（常に1以上）
        aspect_ratio = max(w, h) / min(w, h)
        area = cv2.contourArea(contour)

        # アスペクト比が10未満ならスキップ
        if aspect_ratio < 0:
            continue
        elif area < 0:
            continue
            
        print(f"Mask {idx+1}: OBB の中心=({center_x:.1f}, {center_y:.1f}), 幅={w:.1f}, 高さ={h:.1f}, angle={angle:.1f}°, アスペクト比={aspect_ratio:.2f}, 面積={cv2.contourArea(contour):.1f}")
        
        # アスペクト比が10以上のマスクのみオーバーレイに描画
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + color * 0.5).astype(np.uint8)
        
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} sec")

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

    # =========================
    # 7. 保存先ディレクトリに書き出し (任意)
    # =========================
    output_dir = "C:/Project_sam/sam2/notebooks/images/test-img/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"auto_mask_overlay_img{i}.jpg")
    Image.fromarray(overlay).save(save_path)
    print(f"Saved auto mask overlay to: {save_path}")