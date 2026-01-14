"""
使用SAM2将YOLO格式的bbox注释转换为像素级标签

用法:
    python scripts/bbox_to_mask_sam2.py \
        --data_dir /home/c/yuqyan/data/TreeAI/0_RGB_fL_processed/coco \
        --sam2_checkpoint /path/to/sam2_checkpoint.pth \
        --sam2_config /path/to/sam2_config.yaml \
        --device cuda

输出路径:
    默认输出目录: /zfs/ai4good/datasets/tree/TreeAI/{dataset_name}/
    可通过 --output_base_dir 参数自定义输出根目录
    
    数据集名称自动从 data_dir 路径提取（例如: 0_RGB_fL_processed_coco）
    
    输出目录结构:
        {output_base_dir}/{dataset_name}/
        ├── train/
        │   ├── images/      (软链接到原始图像)
        │   ├── labels_txt/  (软链接到原始txt/xml标签)
        │   └── labels/      (生成的SAM2像素级labels，PNG或TIF)
        ├── val/
        │   ├── images/
        │   ├── labels_txt/
        │   └── labels/
        └── test/  (如果指定了 --splits test)
            ├── images/
            ├── labels_txt/
            └── labels/
    
    注意:
    - 完整标注模式: labels/ 中为 PNG 格式（背景=0）
    - 部分标注模式: labels/ 中为 TIF 格式（未标注=-1，背景=0）
      使用 --partial_annotation 启用部分标注模式
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from tqdm import tqdm
import cv2
import yaml
from datetime import datetime
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SKIP_LOG_PATH = REPO_ROOT / 'debugs' / 'bbox_mask.log'

try:
    from dataset_label_inspector import (
        collect_dataset_class_stats,
        load_class_metadata,
        suggest_classes_to_ignore,
        update_data_value_mapping,
        format_stats_table,
    )

    LABEL_ANALYSIS_AVAILABLE = True
except ImportError:
    LABEL_ANALYSIS_AVAILABLE = False

try:
    from sam2.build_sam import build_sam2, _load_checkpoint
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM2 not available. Please install SAM2 first.")


def build_sam2_from_config_path(config_path: str, ckpt_path: str, device: str = "cuda"):
    """
    Build SAM2 model from absolute config file path.
    This is a workaround for Hydra's initialize_config_module not finding custom configs.
    
    Args:
        config_path: Absolute path to config YAML file
        ckpt_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        SAM2 model instance
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config directly with OmegaConf (avoid Hydra's resolvers)
    cfg = OmegaConf.load(config_path)
    
    # Extract model config - it could be at cfg.model or cfg.trainer.model
    if 'trainer' in cfg and 'model' in cfg.trainer:
        # Training config format
        model_cfg = cfg.trainer.model
    elif 'model' in cfg:
        # Inference config format
        model_cfg = cfg.model
    else:
        raise ValueError(f"Cannot find 'model' config in {config_path}")
    
    # Resolve only the model config (avoid resolving training-specific interpolations)
    try:
        OmegaConf.resolve(model_cfg)
    except Exception as e:
        # If resolution fails, try to manually resolve common patterns
        print(f"Warning: Config resolution had issues: {e}")
        print("Attempting to resolve critical model params only...")
        # Manually set image_size if it references scratch.resolution
        if 'image_size' in model_cfg and isinstance(model_cfg.image_size, str) and '$' in str(model_cfg.image_size):
            if 'scratch' in cfg and 'resolution' in cfg.scratch:
                model_cfg.image_size = cfg.scratch.resolution
                print(f"Set image_size to {model_cfg.image_size}")
    
    # Instantiate model
    model = instantiate(model_cfg, _recursive_=True)
    
    # Load checkpoint
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()
    
    return model


def yolo_to_pixel_coords(yolo_bbox, img_width, img_height):
    """
    将YOLO格式的归一化坐标转换为像素坐标
    
    Args:
        yolo_bbox: (class_id, x_center, y_center, width, height) 归一化坐标 [0,1]
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        (x1, y1, x2, y2, class_id): 像素坐标
    """
    class_id, x_center, y_center, width, height = yolo_bbox
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    # 确保坐标在图像范围内
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    
    return (x1, y1, x2, y2, int(class_id))


def load_yolo_labels(label_path, valid_class_ids=None):
    """
    加载YOLO格式的标签文件
    
    Args:
        label_path: 标签文件路径
        valid_class_ids: 若提供，仅保留该集合内的类别
    
    Returns:
        list of (class_id, x_center, y_center, width, height)
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if valid_class_ids is not None and class_id not in valid_class_ids:
                    continue
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                bboxes.append((class_id, x_center, y_center, width, height))
    
    return bboxes


def load_valid_class_ids(class_config_path, fallback_max_class_id=61):
    """
    从treeAI_classes配置加载合法类别ID集合。
    如果配置缺失或解析失败，将使用[0, fallback_max_class_id]范围。
    """
    default_ids = set(range(fallback_max_class_id + 1))
    if not class_config_path or not os.path.exists(class_config_path):
        print(f"Warning: class config {class_config_path} not found. Using default IDs 0-{fallback_max_class_id}.")
        return default_ids
    try:
        with open(class_config_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        classes = data.get('classes', {})
        class_ids = {int(k) for k in classes.keys()}
        # Ensure background 0 always available
        class_ids.add(0)
        return class_ids
    except Exception as exc:
        print(f"Warning: failed to parse {class_config_path} ({exc}). Using default IDs 0-{fallback_max_class_id}.")
        return default_ids


def load_pascal_voc_bboxes(xml_path, img_width, img_height, valid_class_ids=None):
    """
    解析Pascal VOC XML标签，返回像素坐标bbox列表。
    
    Args:
        xml_path: XML文件路径
        img_width: 图像宽度（用于裁剪）
        img_height: 图像高度
        valid_class_ids: 若提供，仅保留该集合内的类别
    
    Returns:
        list of (x1, y1, x2, y2, class_id)
    """
    if not os.path.exists(xml_path):
        return []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as exc:
        print(f"Warning: Failed to parse XML {xml_path}: {exc}")
        return []
    
    bboxes = []
    for obj in root.findall('object'):
        name = obj.findtext('name', '').strip()
        try:
            class_id = int(name)
        except ValueError:
            continue
        if valid_class_ids is not None and class_id not in valid_class_ids:
            continue
        bbox_node = obj.find('bndbox')
        if bbox_node is None:
            continue
        try:
            xmin = float(bbox_node.findtext('xmin', '0'))
            ymin = float(bbox_node.findtext('ymin', '0'))
            xmax = float(bbox_node.findtext('xmax', '0'))
            ymax = float(bbox_node.findtext('ymax', '0'))
        except ValueError:
            continue
        
        x1 = max(0, min(int(xmin), img_width - 1))
        y1 = max(0, min(int(ymin), img_height - 1))
        x2 = max(0, min(int(xmax), img_width - 1))
        y2 = max(0, min(int(ymax), img_height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        bboxes.append((x1, y1, x2, y2, class_id))
    return bboxes


def log_skipped_sample(log_path, dataset_name, split, image_name, reason):
    """
    将被舍弃的样本记录到调试日志中。
    """
    if not log_path:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    dataset_label = dataset_name or 'unknown_dataset'
    log_line = f"{timestamp}\t{dataset_label}\t{split}\t{image_name}\t{reason}\n"
    with log_path.open('a') as f:
        f.write(log_line)


def cleanup_output_files(*paths):
    """
    移除指定路径（若存在），确保跳过的样本不会残留旧输出。
    """
    for path in paths:
        if not path:
            continue
        try:
            if os.path.lexists(path):
                os.remove(path)
        except OSError as exc:
            print(f"Warning: Failed to remove {path}: {exc}")


def ensure_symlink(source_path, target_path, description="file"):
    """
    创建从 target_path 指向 source_path 的软链接（如果不存在）。
    如果 source_path 本身是软链接，会解析其真实目标。
    """
    if not source_path:
        return
    if os.path.lexists(target_path):
        return
    try:
        resolved_source = os.path.abspath(source_path)
        if os.path.islink(source_path):
            link_target = os.readlink(source_path)
            if not os.path.isabs(link_target):
                link_target = os.path.join(os.path.dirname(source_path), link_target)
            resolved_source = os.path.abspath(link_target)
        os.symlink(resolved_source, target_path)
    except (OSError, FileNotFoundError) as exc:
        print(f"Warning: Failed to create symlink for {description} ({source_path} -> {target_path}): {exc}")


def sort_bboxes_by_area(bboxes, ascending=False):
    """
    按面积排序bboxes（用于Z-Order控制）
    
    Args:
        bboxes: list of (x1, y1, x2, y2, class_id)
        ascending: True=小到大（小框先画），False=大到小（大框先画，小框覆盖）
    
    Returns:
        排序后的bboxes列表
    """
    def bbox_area(bbox):
        x1, y1, x2, y2, _ = bbox
        return (x2 - x1) * (y2 - y1)
    
    return sorted(bboxes, key=bbox_area, reverse=not ascending)


def create_label_map_from_sam2(
    image_path,
    bboxes,
    sam2_predictor,
    restrict_to_bbox=True,
    bbox_margin=0.15,
    sort_by_area=True,
    dtype=np.uint8,
    partial_annotation=False,
):
    """
    使用SAM2为bbox生成像素级标签
    
    Args:
        image_path: 图像路径
        bboxes: list of (x1, y1, x2, y2, class_id) 像素坐标
        sam2_predictor: SAM2ImagePredictor实例
        restrict_to_bbox: 是否将预测限制在bbox框内（带margin）
        bbox_margin: bbox扩展margin比例（0.15=15%），仅当restrict_to_bbox=True时生效
                     允许SAM2预测超出bbox一定比例，解决人工标注不精确的问题
        sort_by_area: 是否按面积排序（大框先画，小框后画覆盖），
                      用于正确处理重叠树木的前后遮挡关系
        dtype: 输出标签图的numpy dtype
        partial_annotation: 是否为部分标注模式，若True则未标注区域标为-1
    
    Returns:
        label_map: numpy array (H, W), 像素值为类别ID
                   - 完整标注模式(partial_annotation=False): 背景=0
                   - 部分标注模式(partial_annotation=True): 未标注=-1, 背景=0
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    
    dtype = np.dtype(dtype)

    # 初始化标签图
    # 部分标注模式：未标注区域=-1；完整标注模式：背景=0
    if partial_annotation:
        label_map = np.full((height, width), -1, dtype=dtype)
    else:
        label_map = np.zeros((height, width), dtype=dtype)
    
    if len(bboxes) == 0:
        return label_map
    
    dtype_info = np.iinfo(dtype)
    dtype_max = dtype_info.max
    dtype_min = dtype_info.min
    
    # Z-Order优化：按面积排序，大框先画，小框后画（覆盖大框）
    # 这样小树的mask会覆盖在大树之上，符合视觉遮挡关系
    if sort_by_area:
        bboxes = sort_bboxes_by_area(bboxes, ascending=False)
   
    # 设置SAM2的图像（只需要设置一次）
    sam2_predictor.set_image(image_rgb)
    
    # 为每个bbox生成标签区域
    for x1, y1, x2, y2, class_id in bboxes:
        # 确保bbox有效
        if x2 <= x1 or y2 <= y1:
            continue
        
        # 创建bbox框（作为prompt），格式为 [x1, y1, x2, y2]
        box = np.array([x1, y1, x2, y2])
        
        try:
            # 使用SAM2预测区域
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],  # SAM2 expects (N, 4)
                multimask_output=False,
            )
            
            # 获取最佳区域（第一个）
            sam_mask = masks[0].astype(bool)
            
            if restrict_to_bbox:
                # 计算带margin的扩展bbox（允许SAM2预测超出原始bbox一定比例）
                # 这解决了人工标注不精确、树枝延伸出框等问题
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                margin_x = int(bbox_w * bbox_margin)
                margin_y = int(bbox_h * bbox_margin)
                
                # 扩展后的bbox坐标（限制在图像范围内）
                exp_x1 = max(0, x1 - margin_x)
                exp_y1 = max(0, y1 - margin_y)
                exp_x2 = min(width - 1, x2 + margin_x)
                exp_y2 = min(height - 1, y2 + margin_y)
                
                # 创建扩展后的bbox区域
                bbox_mask = np.zeros((height, width), dtype=bool)
                bbox_mask[exp_y1:exp_y2+1, exp_x1:exp_x2+1] = True
                
                # 将SAM2区域限制在扩展后的bbox框内
                sam_mask = sam_mask & bbox_mask
            
            # Z-Order处理：后处理的小框会覆盖先处理的大框
            # 将SAM2生成的区域设置为对应的类别ID
            if class_id > dtype_max:
                raise ValueError(f"Class ID {class_id} exceeds dtype capacity ({dtype_max})")
            label_map[sam_mask] = class_id
            
        except Exception as e:
            print(f"Warning: Failed to process bbox {box} for class {class_id}: {e}")
            # 如果SAM2失败，至少填充bbox矩形区域（带margin）
            if restrict_to_bbox:
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                margin_x = int(bbox_w * bbox_margin)
                margin_y = int(bbox_h * bbox_margin)
                fb_x1 = max(0, x1 - margin_x)
                fb_y1 = max(0, y1 - margin_y)
                fb_x2 = min(width - 1, x2 + margin_x)
                fb_y2 = min(height - 1, y2 + margin_y)
                label_map[fb_y1:fb_y2+1, fb_x1:fb_x2+1] = class_id
    
    return label_map


def process_dataset(
    data_dir,
    sam2_predictor,
    output_dir,
    split='val',
    restrict_to_bbox=True,
    bbox_margin=0.15,
    sort_by_area=True,
    valid_class_ids=None,
    label_dtype=np.uint8,
    partial_annotation=False,
    dataset_name=None,
    skip_log_path=None,
):
    """
    处理整个数据集
    
    Args:
        data_dir: 数据根目录（包含train/val/test/images和labels）
        sam2_predictor: SAM2ImagePredictor实例
        output_dir: 输出根目录（将创建 {split}/images/labels_txt/labels 结构）
        split: 'train', 'val' 或 'test'
        restrict_to_bbox: 是否将SAM2预测限制在bbox边界内（带margin）
        bbox_margin: bbox扩展margin比例（默认0.15=15%）
        sort_by_area: 是否按面积排序bbox（大框先画，小框后覆盖）
        valid_class_ids: 需要保留的类别ID集合（其余bbox将被丢弃）
        label_dtype: 保存标签PNG的dtype（默认为uint8）
        partial_annotation: 是否为部分标注模式（未标注区域=-1，输出TIF格式）
        dataset_name: 当前数据集名称，用于日志
        skip_log_path: 跳过样本的日志路径
    """
    # 部分标注模式必须使用有符号整数类型以支持-1
    if partial_annotation:
        label_dtype = np.int16
    label_dtype = np.dtype(label_dtype)

    images_dir = os.path.join(data_dir, split, 'images')
    labels_dir = os.path.join(data_dir, split, 'labels')
    
    # 创建输出目录结构：output_dir/{split}/images, labels_txt, labels
    output_split_dir = os.path.join(output_dir, split)
    output_images_dir = os.path.join(output_split_dir, 'images')
    output_labels_txt_dir = os.path.join(output_split_dir, 'labels_txt')
    output_seg_labels_dir = os.path.join(output_split_dir, 'labels')
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_txt_dir, exist_ok=True)
    os.makedirs(output_seg_labels_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images in {split} split")
    
    # 确定输出标签格式：部分标注用TIF（支持-1），完整标注用PNG
    seg_label_ext = '.tif' if partial_annotation else '.png'
    
    # 处理每张图像
    for image_path in tqdm(image_files, desc=f"Processing {split}"):
        # 获取对应的标签文件（支持 txt 和 xml）
        txt_label_path = os.path.join(labels_dir, image_path.stem + '.txt')
        xml_label_path = os.path.join(labels_dir, image_path.stem + '.xml')
        seg_label_path = os.path.join(output_seg_labels_dir, image_path.stem + seg_label_ext)
        output_image_path = os.path.join(output_images_dir, image_path.name)
        possible_label_symlinks = [
            os.path.join(output_labels_txt_dir, image_path.stem + '.txt'),
            os.path.join(output_labels_txt_dir, image_path.stem + '.xml'),
        ]
        # 旧格式的seg label文件（用于清理格式切换后的残留）
        old_seg_label_path = os.path.join(
            output_seg_labels_dir,
            image_path.stem + ('.png' if partial_annotation else '.tif')
        )
        
        label_path = None
        label_format = None
        if os.path.exists(txt_label_path):
            label_path = txt_label_path
            label_format = 'txt'
        elif os.path.exists(xml_label_path):
            label_path = xml_label_path
            label_format = 'xml'
        
        if not label_path:
            log_skipped_sample(skip_log_path, dataset_name, split, image_path.name, 'label_missing')
            cleanup_output_files(output_image_path, *possible_label_symlinks, seg_label_path, old_seg_label_path)
            continue
        
        with Image.open(image_path) as pil_img:
            img_width, img_height = pil_img.width, pil_img.height
        
        label_basename = os.path.basename(label_path)
        output_label_path = os.path.join(output_labels_txt_dir, label_basename)
        
        if os.path.exists(seg_label_path):
            # 清理非当前格式的旧标签链接，并确保必要的软链接存在
            for candidate in possible_label_symlinks:
                if os.path.basename(candidate) != label_basename:
                    cleanup_output_files(candidate)
            ensure_symlink(image_path, output_image_path, description="image")
            ensure_symlink(label_path, output_label_path, description="label")
            continue
        
        if label_format == 'txt':
            yolo_bboxes = load_yolo_labels(label_path, valid_class_ids)
            if len(yolo_bboxes) == 0:
                log_skipped_sample(skip_log_path, dataset_name, split, image_path.name, 'no_valid_bbox')
                cleanup_output_files(output_image_path, *possible_label_symlinks, seg_label_path, old_seg_label_path)
                continue
            pixel_bboxes = [
                yolo_to_pixel_coords(bbox, img_width, img_height)
                for bbox in yolo_bboxes
            ]
        else:
            pixel_bboxes = load_pascal_voc_bboxes(label_path, img_width, img_height, valid_class_ids)
            if len(pixel_bboxes) == 0:
                log_skipped_sample(skip_log_path, dataset_name, split, image_path.name, 'no_valid_bbox')
                cleanup_output_files(output_image_path, *possible_label_symlinks, seg_label_path, old_seg_label_path)
                continue
        
        # 清理可能存在的其它标签扩展名和旧格式seg label
        for candidate in possible_label_symlinks:
            if os.path.basename(candidate) != label_basename:
                cleanup_output_files(candidate)
        cleanup_output_files(old_seg_label_path)  # 清理旧格式残留
        
        # 创建图像软链接（如果不存在）
        ensure_symlink(image_path, output_image_path, description="image")
        
        # 创建标签软链接（如果不存在）
        ensure_symlink(label_path, output_label_path, description="label")
        
        # 使用SAM2生成像素级标签
        label_map = create_label_map_from_sam2(
            str(image_path),
            pixel_bboxes,
            sam2_predictor,
            restrict_to_bbox=restrict_to_bbox,
            bbox_margin=bbox_margin,
            sort_by_area=sort_by_area,
            dtype=label_dtype,
            partial_annotation=partial_annotation,
        )
        
        # 保存标签图
        if partial_annotation:
            # 部分标注模式：保存为TIF格式（支持有符号整数，-1表示未标注）
            # 使用cv2保存以确保正确处理int16
            cv2.imwrite(seg_label_path, label_map.astype(np.int16))
        else:
            # 完整标注模式：保存为PNG格式
            pil_mode = 'L' if label_dtype == np.uint8 else 'I;16'
            label_image = Image.fromarray(label_map.astype(label_dtype), mode=pil_mode)
            label_image.save(seg_label_path)


def extract_dataset_name(data_dir):
    """
    从数据目录路径中提取数据集名称
    
    Args:
        data_dir: 数据目录路径，例如 /home/c/yuqyan/data/TreeAI/0_RGB_fL_processed/coco
    
    Returns:
        数据集名称，例如 0_RGB_fL_processed_coco
    """
    # 规范化路径
    data_dir = os.path.normpath(data_dir)
    path_parts = data_dir.split(os.sep)
    
    # 尝试找到 TreeAI 相关的部分
    # 例如: /home/c/yuqyan/data/TreeAI/0_RGB_fL_processed/coco
    # 提取: 0_RGB_fL_processed_coco
    
    # 查找 TreeAI 或包含 TreeAI 的部分
    treeai_idx = -1
    for i, part in enumerate(path_parts):
        if 'TreeAI' in part or 'treeAI' in part.lower():
            treeai_idx = i
            break
    
    if treeai_idx >= 0 and treeai_idx + 1 < len(path_parts):
        # 提取 TreeAI 之后的所有部分并组合
        dataset_parts = path_parts[treeai_idx + 1:]
        dataset_name = '_'.join(dataset_parts) if dataset_parts else 'dataset'
    else:
        # 如果没有找到 TreeAI，使用最后两个目录名
        dataset_name = '_'.join(path_parts[-2:]) if len(path_parts) >= 2 else path_parts[-1]
    
    # 清理名称（移除特殊字符）
    dataset_name = dataset_name.replace('/', '_').replace('\\', '_')
    
    return dataset_name


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO bbox annotations to pixel-level labels using SAM2')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file (e.g., configs/bbox_to_mask.yaml)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Single data directory containing train/val/images and labels')
    parser.add_argument('--data_dirs', type=str, nargs='+', default=None,
                       help='Multiple data directories to process (e.g., --data_dirs /path/to/dataset1 /path/to/dataset2)')
    parser.add_argument('--data_list', type=str, default=None,
                       help='File containing list of data directories (one per line)')
    parser.add_argument('--sam2_checkpoint', type=str, default="/home/c/yuqyan/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_tree_finetune_full/checkpoints/checkpoint.pt",
                       help='Path to SAM2 checkpoint file (overrides config)')
    parser.add_argument('--sam2_config', type=str, default="/home/c/yuqyan/code/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_tree_finetune_full.yaml",
                       help='SAM2 config file name or full path (overrides config)')
    parser.add_argument('--output_base_dir', type=str, default="/zfs/ai4good/datasets/tree/TreeAI/TreeAI_new",
                       help='Base directory for output (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu, overrides config)')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                       help='Data splits to process (e.g., train val test, overrides per-dataset splits in config)')
    parser.add_argument('--restrict_to_bbox', action='store_true', default=None,
                       help='Restrict SAM2 label predictions to bbox boundaries (with margin)')
    parser.add_argument('--no_restrict_to_bbox', dest='restrict_to_bbox', action='store_false',
                       help='Allow SAM2 label predictions to extend beyond bbox boundaries')
    parser.add_argument('--bbox_margin', type=float, default=None,
                       help='BBox margin ratio (default 0.15=15%%). Allows SAM2 predictions to extend beyond '
                            'bbox by this proportion. Helps with imprecise manual annotations.')
    parser.add_argument('--sort_by_area', action='store_true', default=None,
                       help='Sort bboxes by area (large first) for correct Z-order occlusion handling')
    parser.add_argument('--no_sort_by_area', dest='sort_by_area', action='store_false',
                       help='Process bboxes in original order without area-based sorting')
    parser.add_argument('--partial_annotation', action='store_true', default=False,
                       help='Partial annotation mode: output TIF format with -1 for unlabeled areas. '
                            'Use this when only some objects are annotated in each image.')
    parser.add_argument('--analyze_labels', action='store_true',
                       help='Print class distribution summary for each dataset based on YOLO labels.')
    parser.add_argument('--auto_update_value_mapping', action='store_true',
                       help='After label analysis, automatically update configs/data/data.yaml with ignore mappings.')
    parser.add_argument('--class_config_path', type=str, default='configs/data/treeAI_classes.yaml',
                       help='Path to configs/data/treeAI_classes.yaml for readable class names.')
    parser.add_argument('--data_config_path', type=str, default='configs/data/data.yaml',
                       help='Path to configs/data/data.yaml for value_mapping updates.')
    parser.add_argument('--class_min_instances', type=int, default=5,
                       help='Minimum number of total bbox instances required to keep a class.')
    parser.add_argument('--class_min_images', type=int, default=3,
                       help='Minimum number of distinct images containing the class required to keep it.')
    parser.add_argument('--force_ignore_classes', type=int, nargs='*', default=None,
                       help='Class IDs that should always be ignored.')
    parser.add_argument('--keep_classes', type=int, nargs='*', default=None,
                       help='Class IDs that must never be ignored.')
    parser.add_argument('--analysis_label_dir_name', type=str, default='labels',
                       help='Name of the label subdirectory inside each split (default: labels).')
    
    args = parser.parse_args()
    
    # Load config file if provided
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            return
        try:
            config = OmegaConf.load(args.config)
            print(f"Loaded config from: {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            return
    
    # Merge config with command line arguments (CLI args take precedence)
    sam2_checkpoint = args.sam2_checkpoint or config.get('sam2', {}).get('checkpoint')
    sam2_config = args.sam2_config or config.get('sam2', {}).get('config')
    output_base_dir = args.output_base_dir or config.get('output', {}).get('base_dir', '/zfs/ai4good/datasets/tree/TreeAI')
    device = args.device or config.get('device', 'cuda')
    restrict_to_bbox = args.restrict_to_bbox if args.restrict_to_bbox is not None else config.get('restrict_to_bbox', True)
    bbox_margin = args.bbox_margin if args.bbox_margin is not None else config.get('bbox_margin', 0.15)
    sort_by_area = args.sort_by_area if args.sort_by_area is not None else config.get('sort_by_area', True)
    partial_annotation = args.partial_annotation or config.get('partial_annotation', False)
    
    # Print output directory information
    print(f"\n{'='*60}")
    print(f"Output Configuration")
    print(f"{'='*60}")
    print(f"Base output directory: {output_base_dir}")
    print(f"  → Final output will be: {output_base_dir}/<dataset_name>/<split>/")
    print(f"     - images/      (symlinks to original images)")
    print(f"     - labels_txt/  (symlinks to original txt/xml annotations)")
    print(f"     - labels/       (generated SAM2 segmentation labels)")
    print(f"{'='*60}\n")
    
    # Collect datasets to process
    datasets_to_process = []
    
    # Priority 1: Command line arguments (--data_dir, --data_dirs, --data_list)
    if args.data_list:
        if not os.path.exists(args.data_list):
            print(f"Error: Data list file not found: {args.data_list}")
            return
        with open(args.data_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    datasets_to_process.append({
                        'path': line,
                        'splits': args.splits or ['train', 'val']
                    })
        print(f"Loaded {len(datasets_to_process)} datasets from {args.data_list}")
    elif args.data_dirs:
        datasets_to_process = [
            {'path': d, 'splits': args.splits or ['train', 'val']}
            for d in args.data_dirs
        ]
        print(f"Processing {len(datasets_to_process)} datasets from command line")
    elif args.data_dir:
        datasets_to_process = [{
            'path': args.data_dir,
            'splits': args.splits or ['train', 'val']
        }]
        print(f"Processing single dataset: {args.data_dir}")
    # Priority 2: Config file datasets
    elif 'datasets' in config and config.datasets:
        for dataset_cfg in config.datasets:
            dataset_path = dataset_cfg.get('path')
            dataset_name = dataset_cfg.get('name')  # Optional custom name
            dataset_splits = args.splits or dataset_cfg.get('splits', ['train', 'val'])
            if not dataset_path:
                print(f"Warning: Dataset config missing 'path', skipping: {dataset_cfg}")
                continue
            
            datasets_to_process.append({
                'path': dataset_path,
                'name': dataset_name,  # Will use extract_dataset_name if None
                'splits': dataset_splits,
            })
        print(f"Loaded {len(datasets_to_process)} datasets from config file")
    else:
        print("Error: Must specify datasets via:")
        print("  - --config with datasets list in config file, OR")
        print("  - --data_dir, --data_dirs, or --data_list")
        parser.print_help()
        return
    
    if len(datasets_to_process) == 0:
        print("Error: No datasets specified")
        return

    analyze_labels = (args.analyze_labels or args.auto_update_value_mapping)
    if analyze_labels and not LABEL_ANALYSIS_AVAILABLE:
        print("Warning: dataset_label_inspector module not available. Skipping label analysis.")
        analyze_labels = False

    class_names = {}
    if analyze_labels:
        class_config_path = args.class_config_path
        if not os.path.exists(class_config_path):
            print(f"Warning: Class config not found at {class_config_path}. Skipping label analysis.")
            analyze_labels = False
        else:
            try:
                class_names = load_class_metadata(class_config_path)
            except Exception as e:
                print(f"Warning: Failed to load class metadata ({class_config_path}): {e}")
                analyze_labels = False

    if args.auto_update_value_mapping and not analyze_labels:
        print("Warning: auto_update_value_mapping requested but label analysis is disabled.")
    
    # 合法类别集合与输出dtype
    valid_class_ids = load_valid_class_ids(args.class_config_path)
    max_class_id = max(valid_class_ids) if valid_class_ids else 0
    label_dtype = np.uint8 if max_class_id < 256 else np.uint16
    dtype_name = 'uint8' if label_dtype == np.uint8 else 'uint16'
    skip_log_path = DEFAULT_SKIP_LOG_PATH
    print(f"Valid class IDs: {len(valid_class_ids)} loaded from {args.class_config_path} (max={max_class_id}).")
    if partial_annotation:
        print(f"Partial annotation mode: labels saved as int16 TIF (unlabeled=-1, background=0)")
    else:
        print(f"Full annotation mode: labels saved as {dtype_name} PNG (background=0)")
    print(f"BBox margin: {bbox_margin:.0%} (restrict_to_bbox={restrict_to_bbox})")
    print(f"Z-order sorting by area: {sort_by_area}\n")
    
    # Validate SAM2 checkpoint and config
    if not sam2_checkpoint:
        print("Error: SAM2 checkpoint not specified. Use --sam2_checkpoint or provide in config file.")
        return
    
    if not sam2_config:
        print("Error: SAM2 config not specified. Use --sam2_config or provide in config file.")
        return
    
    if not SAM2_AVAILABLE:
        print("Error: SAM2 is not available. Please install it first.")
        print("Installation: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        return
    
    if not os.path.exists(sam2_checkpoint):
        print(f"Error: SAM2 checkpoint not found: {sam2_checkpoint}")
        return
    
    # Initialize SAM2
    print(f"\n{'='*60}")
    print(f"Loading SAM2 model...")
    print(f"  Config: {sam2_config}")
    print(f"  Checkpoint: {sam2_checkpoint}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    # Load SAM2 model
    if os.path.isabs(sam2_config) and os.path.exists(sam2_config):
        print(f"Using config file: {sam2_config}")
        sam2_model = build_sam2_from_config_path(
            sam2_config,
            sam2_checkpoint,
            device=device
        )
    else:
        # Use standard build_sam2 function (for relative config names)
        sam2_model_cfg = sam2_config
        if sam2_model_cfg.endswith('.yaml'):
            sam2_model_cfg = sam2_model_cfg[:-5]
        print(f"Using SAM2 config: {sam2_model_cfg}")
        sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("✓ SAM2 loaded successfully!\n")
    
    # Process each dataset
    print(f"{'='*60}")
    print(f"Processing {len(datasets_to_process)} dataset(s)")
    print(f"{'='*60}\n")
    
    for idx, dataset_info in enumerate(datasets_to_process, 1):
        data_dir = dataset_info['path']
        dataset_name = dataset_info.get('name')  # May be None
        dataset_splits = dataset_info['splits']
        
        print(f"\n{'='*60}")
        print(f"Dataset {idx}/{len(datasets_to_process)}: {data_dir}")
        print(f"{'='*60}")
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"Warning: Data directory not found: {data_dir}")
            print("Skipping...\n")
            continue
        
        # Determine output directory
        if dataset_name:
            final_dataset_name = dataset_name
        else:
            final_dataset_name = extract_dataset_name(data_dir)
        output_dir = os.path.join(output_base_dir, final_dataset_name)
        
        print(f"Dataset name: {final_dataset_name}")
        print(f"Splits to process: {dataset_splits}")
        print(f"Output directory: {output_dir}\n")
        
        # Process each split
        for split in dataset_splits:
            print(f"Processing {split} split...")
            try:
                process_dataset(
                    data_dir,
                    sam2_predictor,
                    output_dir,
                    split=split,
                    restrict_to_bbox=restrict_to_bbox,
                    bbox_margin=bbox_margin,
                    sort_by_area=sort_by_area,
                    valid_class_ids=valid_class_ids,
                    label_dtype=label_dtype,
                    partial_annotation=partial_annotation,
                    dataset_name=final_dataset_name,
                    skip_log_path=skip_log_path,
                )
            except Exception as e:
                print(f"Error processing {split} split: {e}")
                import traceback
                traceback.print_exc()
                continue

        if analyze_labels:
            try:
                stats = collect_dataset_class_stats(
                    data_dir,
                    dataset_name=final_dataset_name,
                    splits=dataset_splits,
                    label_dir_name=args.analysis_label_dir_name,
                )
                suggestions = suggest_classes_to_ignore(
                    stats,
                    min_instances=args.class_min_instances,
                    min_images=args.class_min_images,
                    force_ignore=args.force_ignore_classes,
                    keep_classes=args.keep_classes,
                )
                print("\nLabel statistics based on YOLO annotations:")
                print(format_stats_table(stats, class_names, highlight=suggestions))
                if suggestions:
                    suggestion_list = ", ".join(str(cid) for cid in suggestions.keys())
                    print(f"Suggested ignore classes: {suggestion_list}")
                    if args.auto_update_value_mapping:
                        try:
                            updates = update_data_value_mapping(
                                args.data_config_path,
                                final_dataset_name,
                                suggestions,
                                class_names,
                                write_changes=True,
                            )
                            if updates:
                                print(f"Updated {updates} entries in {args.data_config_path}")
                            else:
                                print(f"{final_dataset_name}: value_mapping already up-to-date in {args.data_config_path}")
                        except Exception as cfg_err:
                            print(f"Warning: Failed to update {args.data_config_path}: {cfg_err}")
                else:
                    print("No ignore suggestions for this dataset.")
            except Exception as analysis_err:
                print(f"Warning: Failed to analyze labels for {final_dataset_name}: {analysis_err}")
        
        print(f"\n✓ Completed dataset {idx}/{len(datasets_to_process)}: {final_dataset_name}")
        print(f"  Output: {output_dir}\n")
    
    print(f"\n{'='*60}")
    print("All datasets processed!")
    print(f"{'='*60}")
    print(f"\nOutput directories:")
    for dataset_info in datasets_to_process:
        data_dir = dataset_info['path']
        if os.path.exists(data_dir):
            dataset_name = dataset_info.get('name') or extract_dataset_name(data_dir)
            output_dir = os.path.join(output_base_dir, dataset_name)
            print(f"  - {output_dir}")


if __name__ == '__main__':
    main()
