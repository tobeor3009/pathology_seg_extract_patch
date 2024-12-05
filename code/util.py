import os
import cv2
import json
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    pass

xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])
xyz_ref_white = np.array((0.950456, 1., 1.088754))


def RGB_to_LAB(image_array, use_gpu=False):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    arr = image_array[:, :, ::-1]
    arr = arr / 255
    mask = arr > 0.04045
    arr[mask] = compat_cp.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    arr = arr @ xyz_from_rgb.T
    # scale by CIE XYZ tristimulus values of the reference white point
    arr = arr / xyz_ref_white

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    mask = arr > 0.008856
    mask_x, mask_y, mask_z = mask[..., 0], mask[..., 1], mask[..., 2]

    arr_converted = compat_cp.zeros_like(arr)
    arr_converted[mask] = compat_cp.cbrt(arr[mask])
    arr_converted[~mask] = 7.787 * arr[~mask] + (16 / 116)

    x_converted, y_converted, z_converted = arr_converted[...,
                                                          0], arr_converted[..., 1], arr_converted[..., 2]

    L = compat_cp.zeros_like(y)

    # Nonlinear distortion and linear transformation
    L[mask_y] = 116 * compat_cp.cbrt(y[mask_y]) - 16
    L[~mask_y] = 903.3 * y[~mask_y]
    L *= 2.55
    # if want to see this formula, go to https://docs.opencv.org/3.4.15/de/d25/imgproc_color_conversions.html RGB <-> CIELab
    a = 500 * (x_converted - y_converted) + 128
    b = 200 * (y_converted - z_converted) + 128

    return compat_cp.round(compat_cp.stack([L, a, b], axis=-1)).astype("uint8")


def get_tissue_mask(I, luminosity_threshold=0.8, use_gpu=False):
    #     I_LAB = RGB_to_LAB(I, use_gpu=use_gpu)
    L = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)[:, :, 0].astype("float16")
    L = L / 255.0  # Convert to range [0,1].
    mask = L < luminosity_threshold

    # Check it's not empty
#     if mask.sum() == 0:
#         raise Exception("Empty tissue mask computed")

    return mask


def RGB_to_OD(I, use_gpu=False):

    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    """
    Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    I[I == 0] = 1
    return -1 * compat_cp.log(I / 255)


def OD_to_RGB(OD, use_gpu=False):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    """
    Convert from optical density (OD_RGB) to RGB
    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, 'Negative optical density'
    return (255 * compat_cp.exp(-1 * OD)).astype(np.uint8)


def soft_threshold(rho, lamda, use_gpu):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    '''Soft threshold function used for normalized data and lasso regression'''
    lamda_below_index = rho < - lamda
    lamda_upper_index = rho > lamda

    new_rho = compat_cp.zeros_like(rho)
    new_rho[lamda_below_index] = rho[lamda_below_index] + lamda
    new_rho[lamda_upper_index] = rho[lamda_upper_index] - lamda

    return new_rho


def coordinate_descent_lasso(H, W, Y, lamda=0.1, num_iters=100, min_delta=0.9999, max_patience=10, use_gpu=False):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    '''Coordinate gradient descent for lasso regression - for normalized data. 
    The intercept parameter allows to specifY whether or not we regularize H_0'''

    # Initialisation of useful values
    m, n = W.shape
    # normalizing W in case it was not done before
    W = W / (compat_cp.linalg.norm(W, axis=0))

    H_patience = 0
    H_previos_error = 1000
    H_current_error = compat_cp.linalg.norm(Y - W @ H)
    H_loss_decrease_ratio = H_current_error / H_previos_error

    # Looping until max number of iterations
    for iter_index in range(num_iters):
        # Looping through each coordinate
        for j in range(n):

            # Vectorized implementation
            W_j = W[:, j].reshape(-1, 1)
            Y_pred = W @ H
            rho = W_j.T @ (Y - Y_pred + H[j] * W_j)
            rho = rho.squeeze()
            H[j] = soft_threshold(rho, lamda, use_gpu=use_gpu)

        H_current_error = compat_cp.linalg.norm(Y - W @ H)
        H_loss_decrease_ratio = H_current_error / H_previos_error
        if H_loss_decrease_ratio > min_delta:
            H_patience += 1
            if H_patience > max_patience:
                break
        H_previos_error = H_current_error
    return H, iter_index


def coordinate_descent_lasso(H, W, Y, lamda=0.1, num_iters=100, use_gpu=False):
    global np, cp
    if use_gpu:
        compat_cp = cp
    else:
        compat_cp = np
    '''Coordinate gradient descent for lasso regression - for normalized data. 
    The intercept parameter allows to specifY whether or not we regularize H_0'''

    # Initialisation of useful values
    m, n = W.shape
    # normalizing W in case it was not done before
    W = W / (compat_cp.linalg.norm(W, axis=0))

    # Looping until max number of iterations
    for i in range(num_iters):
        # Looping through each coordinate
        for j in range(n):

            # Vectorized implementation
            W_j = W[:, j].reshape(-1, 1)
            Y_pred = W @ H
            rho = W_j.T @ (Y - Y_pred + H[j] * W_j)
            rho = rho.squeeze()
            H[j] = soft_threshold(rho, lamda, use_gpu=use_gpu)
    return H


def get_basis_concentraion_matrix(img_array, tissue_mask,
                                  input_channel_num, converted_channel_num,
                                  stop_ratio=0.9999, max_iter=200,
                                  h_lambda=0.1, use_gpu=False):
    if use_gpu:
        cp.cuda.runtime.setDevice(0)
        compat_cp = cp
    else:
        compat_cp = np

    img_array = img_array[tissue_mask]
    img_array = RGB_to_OD(img_array)
    img_array = compat_cp.array(img_array, dtype="float32")
    img_array = compat_cp.rollaxis(img_array, 1, 0)
    img_shape = img_array.shape
    elemenet_num = img_shape[-1]

    previos_error = 10000
    if use_gpu:
        w = compat_cp.random.random(
            size=(input_channel_num, converted_channel_num), dtype="float32")
        h = compat_cp.random.random(
            size=(converted_channel_num, elemenet_num), dtype="float32")
    else:
        w = compat_cp.random.random(
            size=(input_channel_num, converted_channel_num))
        h = compat_cp.random.random(size=(converted_channel_num, elemenet_num))
    for total_index in range(max_iter):

        for w_index in range(100):
            w = w * (img_array @ h.T) / (w @ h @ h.T + 1e-16)

        h = coordinate_descent_lasso(h, w, img_array,
                                     lamda=h_lambda, num_iters=100,
                                     use_gpu=use_gpu)

        for h2_index in range(5):
            h = h * (w.T @ img_array) / (w.T @ w @ h + 1e-16)

        current_error = compat_cp.linalg.norm(img_array - w @ h)
        print(f"iter {total_index}: {current_error}")

        if stop_ratio is not None:
            if current_error / previos_error > stop_ratio:
                break
        previos_error = current_error

    # order H and E.
    # H on first row.
#     if w[0, 0] < w[0, 1]:
#         w = w[:, [1, 0]]

    return w, h


def get_concentraion_matrix(img_array, w,
                            stop_ratio=0.9999, max_iter=200,
                            h_lambda=0.1, use_gpu=False):
    if use_gpu:
        cp.cuda.runtime.setDevice(0)
        compat_cp = cp
    else:
        compat_cp = np

    img_array = RGB_to_OD(img_array).reshape(-1, 3)
    img_array = compat_cp.array(img_array, dtype="float32")
    img_array = compat_cp.rollaxis(img_array, 1, 0)
    img_shape = img_array.shape
    elemenet_num = img_shape[-1]

    converted_channel_num = w.shape[1]
    previos_error = 10000
    w = compat_cp.array(w)
    if use_gpu:
        h = compat_cp.random.random(
            size=(converted_channel_num, elemenet_num), dtype="float32")
    else:
        h = compat_cp.random.random(size=(converted_channel_num, elemenet_num))

    for total_index in range(max_iter):

        h = h * (w.T @ img_array) / (w.T @ w @ h + 1e-16)

        current_error = compat_cp.linalg.norm(img_array - w @ h)
        print(f"stage2_iter {total_index}: {current_error}")

        if stop_ratio is not None:
            if current_error / previos_error > stop_ratio:
                break
        previos_error = current_error
    return h

def get_random_index_permutation_range(image_shape, patch_stride):
    row_index_range = range(0, image_shape[0], patch_stride)
    col_index_range = range(0, image_shape[1], patch_stride)

    random_permutation_len = len(row_index_range) * len(col_index_range)
    random_permutation = np.zeros((random_permutation_len, 2), dtype="int32")

    for row_index, row_num in enumerate(row_index_range):
        for col_index, col_num in enumerate(col_index_range):
            current_index = row_index * len(col_index_range) + col_index 
            random_permutation[current_index] = (row_num, col_num)
    np.random.shuffle(random_permutation)
    
    return random_permutation


def imread(img_path, channel=None):
    img_byte_stream = open(img_path.encode("utf-8"), "rb")
    img_byte_array = bytearray(img_byte_stream.read())
    img_numpy_array = np.asarray(img_byte_array, dtype=np.uint8)
    img_numpy_array = cv2.imdecode(
        img_numpy_array, cv2.IMREAD_UNCHANGED)
    if channel == "rgb":
        img_numpy_array = cv2.cvtColor(
            img_numpy_array, cv2.COLOR_BGR2RGB)

    return img_numpy_array

def read_json_as_dict(json_path):
    json_file = open(json_path, encoding="utf-8")
    json_str = json_file.read()
    json_dict = json.loads(json_str)
    
    return json_dict

def get_parent_dir_name(path, level=1):

    path_spliter = os.path.sep
    abs_path = os.path.abspath(path)

    return abs_path.split(path_spliter)[-(1 + level)]

##############################################
################## Grad CAM ##################
##############################################
from tensorflow import keras
import matplotlib.cm as cm

def get_last_conv_name(model):
    layer_names = [layer.name for layer in model.layers]
    conv_layer_name = [layer_name for layer_name in layer_names if layer_name.find("conv") >= 0]
    last_conv_name = conv_layer_name[-1]
    
    return last_conv_name

def make_grad_model(model, target_layer_name):
    # Model input: model.input
    # Model output: [last_conv_layer_outputs, model.outputs]
    grad_model = keras.models.Model(
        model.input, [model.get_layer(target_layer_name).output, model.output]
    )
    return grad_model

def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    # compute
    # 1. last_conv_layer_output
    # 2. class_channel value in prediction
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        # compute gradient 
        grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # compute gradient channel mean 
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # compute matmul(last_conv_layer_output, pooled_grads)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_array_normalized, heatmap, cam_path="cam.jpg", alpha=0.4, display_cam=True):
    # Load the original image
    img_array = (img_array_normalized + 1) * 127.5
    img_array = img_array.astype('uint8')
    img_array = img_array[0]
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    if display_cam:
        display(Image.open(cam_path))

def decode_classify_predictions(pred_array, index_to_label_dict):
    predicted_index = np.sum(np.argmax(preds, axis=1))
    predicted = index_to_label_dict[predicted_index]
    
    return predicted

def decode_binary_classify_predictions(pred_array):
    predicted = np.round(pred_array).astype("int32")
    predicted = predicted.flatten()[0]
    return predicted

def restore_sigmoid_value(y, decrease_ratio=1e1):
    original_x = np.log(y/(1-y))
    restored_x = decrease_ratio*original_x
    new_y = 1/(1+np.exp(-restored_x))
    return new_y

def save_overlayed_cam(img_array, heatmap, overlayed_path, heatmap_path, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap.save(heatmap_path)
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    
    superimposed_img.save(overlayed_path)        
   