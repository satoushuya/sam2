import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from automatic_mask_generator import SAM2AutomaticMaskGenerator

# =========================
# 1. デバイス設定
# =========================
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
sam2_checkpoint = "C:/Users/syuuy/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
config_name = "C:/Users/syuuy/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2 = build_sam2(config_name, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2)

# =========================
# 3. 画像の読み込み
# =========================
image_path = "C:/Users/syuuy/sam2/notebooks/images/truck.jpg"
image_pil = Image.open(image_path).convert("RGB")
image = np.array(image_pil)
height, width, _ = image.shape

# 画像を予測器にセット
predictor.set_image(image)

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
    min_mask_region_area=30000,    # 小さすぎるマスクを除外
    output_mode="binary_mask",    # 出力マスクの形式。binary_mask / uncompressed_rle / coco_rle
    multimask_output=True,        # Trueにすると1点につき複数マスク候補を出す
)

# =========================
# 5. SAM2 でマスク推定を繰り返し取得
# =========================
# 各ポイントを個別に predict() すると計算量が大きいので、
# SAM2 で一括推定できるかを確認し、不可ならループで回す
# （ここではループ例を示します）

all_masks = []
all_scores = []

# SAM2 の実装が "一括" で複数ポイントを受け付けるかどうか要確認。
# 受け付けるならまとめて呼ぶほうが速い場合もあります。
# ここでは一例として1点ずつ呼ぶ方法を示します。

for coord, label in zip(point_coords, point_labels):
    masks, scores, _ = predictor.predict(
        point_coords=coord[None, :],  # shape: (1, 2)
        point_labels=np.array([label]),
        box=None,
        multimask_output=True  # 複数マスク候補を出す
    )
    # ここでは全候補をとりあえずリストに追加
    for m, s in zip(masks, scores):
        all_masks.append(m)
        all_scores.append(s)

# =========================
# 6. 簡易的なマスク可視化
# =========================
#  - 重複マスクの統合やフィルタリングは最小限。
#  - 重なりがある場合、後から塗られた色が上書きされます。
#  - 実運用ではNMSやスコア判定などを行ってマージすることを推奨。

overlay = image.copy()

for mask in all_masks:
    # mask shape: (H, W)
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + color * 0.5).astype(np.uint8)

# 結果表示
plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.axis("off")
plt.show()

# =========================
# 7. 保存先ディレクトリに書き出し (任意)
# =========================
output_dir = "C:/Users/syuuy/sam2/notebooks/images/auto_masks"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "auto_mask_overlay.png")
Image.fromarray(overlay).save(save_path)
print(f"Saved auto mask overlay to: {save_path}")