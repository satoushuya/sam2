import os
import time
import numpy as np  # ← これを追加！
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

# SAM2の関数をインポート
from sam2.build_sam import build_sam2_video_predictor

# --- ここからメインの処理 ---

def show_mask(mask, ax, obj_id=None, random_color=False):
    """
    SAM2の実行結果のセグメンテーションをマスクとして描画する。

    Args:
        mask (numpy.ndarray): 実行結果のセグメンテーション
        ax (matplotlib.axes._axes.Axes): matplotlibのAxis
        obj_id (int): オブジェクトID
        random_color (bool): マスクの色をランダムにするかどうか
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    """
    指定した座標に星を描画する。
    labelsがPositiveの場合は緑、Negativeの場合は赤。

    Args:
        coords (numpy.ndarray): 指定した座標
        labels (numpy.ndarray): Positive or Negative
        ax (matplotlib.axes._axes.Axes): matplotlibのAxis
        marker_size (int, optional): マーカーのサイズ
    """
    print(type(coords))
    print(type(labels))
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


# 1. 【準備】画像フレームのパスを準備（これは既存のコード）
video_dir = "input/dog_images"
# ...（フレーム名を読み込む処理）...
# ...（video_dirの定義の後）...
# ディレクトリ内のJPEGファイルをスキャンする
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

# ファイル名でソートする（この行でエラーが出ていました）
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# 2. 【AIの準備】SAM2のAIモデル（predictor）を作成する
# （image_b89a60.png のコード）
device = torch.device("cpu") # あるいは "cuda"
# ↓↓ パスを修正したのを忘れないように！
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt" 
model_cfg = "configs/sam2.1/sam2_1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# 3. 【動画の読み込み】動画の全フレームをAIに読み込ませる
# （image_b89a60.png のコード）
inference_state = predictor.init_state(video_path=video_dir)

# 4. 【物体検出の実行】犬の位置をクリックして、マスクを生成させる
# （image_b89a7e.png のコード）
ann_frame_idx = 0  # 最初のフレームを対象
ann_obj_id = 0     # 犬をオブジェクトID 0とする
# ↓↓ この座標が「星」の位置になる
points = np.array([[539.9, 408.1]], dtype=np.float32)
labels = np.array([1], np.int32) # 1はPositive（緑の星）
_, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# AIの出力を、表示できる形式に変換
out_mask = (out_mask_logits[0, 0] > 0).cpu().numpy()

# 5. 【結果の描画】元の画像、マスク、星を重ねて表示
# まずは元の画像を取得
img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
img = np.array(Image.open(img_path))

# 描画の準備
plt.figure(figsize=(10, 10))
plt.imshow(img)

# マスクと星を描画
show_mask(out_mask, plt.gca()) # ← 作成したマスクを描画
show_points(points, labels, plt.gca()) # ← クリックした星を描画

# 最終結果を表示
plt.title(f"Frame {ann_frame_idx}")
plt.axis('off')
plt.show()

# ウィンドウが消えないように一時停止
time.sleep(20)