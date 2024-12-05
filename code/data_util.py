import os
import math
import cv2
import numpy as np
import mahotas.polygon as ploygon_to_mask
from lxml import etree
import re
import tifffile
import tempfile

def check_policy(target_str, mask_policy_dict):
    in_policy = False
    mask_key = None
    for policy_key in mask_policy_dict.keys():
        in_policy = policy_key in target_str
        if in_policy is True:
            mask_key = policy_key
            break
    return in_policy, mask_key


def get_mask_image(image_shape, mask_xml_path, level, downsizing_per_level=2, mask_policy_dict={'None': 255}):
    image_shape = np.array(image_shape) // (downsizing_per_level ** level)

    mask_array = np.zeros(image_shape, dtype=np.uint8)
    if os.path.exists(mask_xml_path) is False:
        print(f"{mask_xml_path} not exist.")
        return mask_array
    mask_xml_tree = etree.parse(mask_xml_path)
    ASAP_Annotations_tree = mask_xml_tree.getroot()
    Annotations_tree = ASAP_Annotations_tree[0]

    for Annotation_tree in Annotations_tree:
        Annotaion_Type = Annotation_tree.attrib['Type']
        Annotation_PartOfGroup = Annotation_tree.attrib['PartOfGroup']
        Coordinates_tree = Annotation_tree[0]
        is_in_policy, mask_key = check_policy(Annotation_PartOfGroup,
                                              mask_policy_dict)
        if (Annotaion_Type == "Spline") and (is_in_policy == True):
            mask_value = mask_policy_dict[mask_key]
            polygon_points = []
            for Coordinate in Coordinates_tree.iter("Coordinate"):
                x = float(Coordinate.attrib["X"])
                y = float(Coordinate.attrib["Y"])
                polygon_points.append((round(
                    y) // (downsizing_per_level ** level), round(x) // (downsizing_per_level ** level)))
            ploygon_to_mask.fill_polygon(
                polygon_points, mask_array, mask_value)

    return mask_array

def get_rec_info_list(xml_path, patch_size, stride_ratio=0.5, conservative=False, include_edge=False):
    Rec_info_list = []
    mask_xml_tree = etree.parse(xml_path)
    ASAP_Annotations_tree = mask_xml_tree.getroot()
    Annotations_tree = ASAP_Annotations_tree[0]

    for Annotation_tree in Annotations_tree:
        Annotation_Name = Annotation_tree.attrib["Name"].replace("\n", "")
        Annotaion_Type = Annotation_tree.attrib['Type']
        Annotation_PartOfGroup = Annotation_tree.attrib['PartOfGroup']
        Coordinates_tree = Annotation_tree[0]

        if Annotaion_Type == "Rectangle" and Annotation_PartOfGroup != "lvi":
            x_list, y_list = [], []
            for Coordinate in Coordinates_tree.iter("Coordinate"):
                x = float(Coordinate.attrib["X"])
                y = float(Coordinate.attrib["Y"])
                x_list.append(x), y_list.append(y)
            row_min, row_max = math.floor(min(y_list)), math.ceil(max(y_list))
            col_min, col_max = math.floor(min(x_list)), math.ceil(max(x_list))

            stride = int(patch_size * stride_ratio)

            row_idx_list = [row_min + stride * row_idx for row_idx in range(0, 1 + (row_max - row_min - patch_size) // stride)]
            col_idx_list = [col_min + stride * col_idx for col_idx in range(0, 1 + (col_max - col_min - patch_size) // stride)]

            if conservative:
                row_idx_list = row_idx_list[1:-1]
                col_idx_list = col_idx_list[1:-1]
                if include_edge:
                    row_idx_list = [row_min] + row_idx_list
                    col_idx_list = [col_min] + col_idx_list
            if include_edge:
                row_idx_list = row_idx_list + [row_max - patch_size]
                col_idx_list = col_idx_list + [col_max - patch_size]
            
            Rec_info = [Annotation_Name, row_idx_list, col_idx_list, [row_min, row_max, col_min, col_max]]
            Rec_info_list.append(Rec_info)
    return Rec_info_list

def remove_orange_peel(mask_array, mask_policy_dict, remove_region_ratio=0.01):
    mask_area = np.prod(mask_array.shape[:2])
    for key, value in mask_policy_dict.items():
        mask_region = np.prod(mask_array == value).astype("uint8")
        if np.sum(mask_region) == 0:
            continue
        else:
            mask_num, mask_region = cv2.connectedComponents(mask_region)
            for mask_index in range(1, mask_num):
                mask_index_region = mask_region == mask_index
                mask_boundary_sum = np.sum(mask_index_region[0, :]) + np.sum(mask_index_region[-1, :]) + \
                    np.sum(mask_index_region[:, 0]) + np.sum(mask_index_region[:, -1])
                if mask_boundary_sum > 0:
                    region_ratio = np.sum(mask_index_region) / mask_area
                    if region_ratio <= remove_region_ratio:
                        mask_array[mask_index_region] = 0
    
    return mask_array

def resize_with_preserve_rgb_value(mask_array, target_dsize, mask_policy_dict):
    x, y = target_dsize
    resize_mask_array = np.zeros((y, x, 3), dtype=mask_array.dtype)
    for key, value in mask_policy_dict.items():
        rgb_value, label_key = value[:-1], value[-1]
        mask_region = np.prod(mask_array == rgb_value, axis=-1).astype("uint8") * 255
        if np.sum(mask_region) == 0:
            continue
        else:
            mask_region = cv2.resize(mask_region, (x, y), cv2.INTER_LINEAR)
            resize_mask_array[mask_region > 127.5] = rgb_value
    return resize_mask_array

"""
output: int
"""
def get_mpp_from_description(text, regex_str='PhysicalSize.="(0.[0-9]{1,10})"', base_mpp_value=0.5):
    regex = re.compile(regex_str)
    matched_group = regex.findall(text)
    
    if matched_group:
        assert matched_group[0] == matched_group[1], f'{matched_group}'
        mpp_value = float(matched_group[0])
    else:
        mpp_value = base_mpp_value
    return mpp_value

def get_size4mpp(size_patch_level0, mpp_standard, mpp_value) :

    fov = float(mpp_standard * size_patch_level0)
    new_patch_size = int(fov / mpp_value)
    
    return new_patch_size

def get_wsi_info_read_region(wsi_path, downsize_scale=4, use_memmap=False):
    """
    OpenSlide의 read_region을 사용하여 특정 level의 이미지를 읽고 MPP 값을 반환합니다.

    Parameters:
        wsi_path (str): WSI 파일 경로
        level (int): 읽고자 하는 해상도 레벨 (기본값: 2)

    Returns:
        wsi_array (np.ndarray): 지정된 레벨의 Numpy 배열 형식의 이미지
        mpp (tuple): 지정된 레벨의 MPP 값 (mpp_x, mpp_y)
    """
    tiff_object = tifffile.TiffFile(wsi_path)
    # OpenSlide로 WSI 로드
    tiff_tags = {}
    
    out = 'memmap' if use_memmap else None
    
    wsi_array_list = []
    for tiff_page in  tiff_object.pages:
        try:
            for tag in tiff_page.tags.values():
                name, value = tag.name, tag.value
                tiff_tags[name] = value
        except:
            continue
        wsi_array_each = tiff_page.asarray(out=out)
        wsi_array_list.append(wsi_array_each)
    if len(wsi_array_list) == 1:
        wsi_array = wsi_array_list[0]
    else:
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            wsi_array_each = wsi_array_list[0]
            wsi_hw, dtype = wsi_array_each.shape, wsi_array_each.dtype
            assert len(wsi_array_list) == 3
            assert len(wsi_hw) == 2
            wsi_shape = tuple((*wsi_hw, 3))
            wsi_array = np.memmap(temp_file, dtype=dtype, mode='w+', shape=wsi_shape)
            for channel_idx, wsi_array_each in enumerate(wsi_array_list):
                wsi_array[..., channel_idx] = wsi_array_each
    if wsi_array.shape[0] == 3:
        wsi_array = np.transpose(wsi_array, (2, 1, 0))

    wsi_hw = np.array(wsi_array.shape[:2])
    if downsize_scale > 1:
        downsampled_shape = np.round(wsi_hw / downsize_scale).astype("int32")
        wsi_array = cv2.resize(wsi_array, downsampled_shape[::-1], interpolation=cv2.INTER_LINEAR)

    mpp_value = get_mpp_from_description(tiff_tags["ImageDescription"])
    return wsi_array, mpp_value