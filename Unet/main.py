import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import tensorflow as tf
import os
import random
from keras.models import Model
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D, Reshape, core, \
    Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.metrics import jaccard_score
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
from collections import defaultdict
from tensorflow.keras.layers import Conv2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
num_cls = 10
N_Cls = 10
inDir = 'dstl'
DF = pd.read_csv(inDir + '/train_wkt_v4.csv')
GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
ISZ = 160
smooth = 1e-12


def _convert_coordinates_to_raster(coords, img_size, xymax):
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask


def M(image_id):
    filename = os.path.join(inDir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img


def get_crop_shape(target, refer):
    cw = target.get_shape()[2] - refer.get_shape()[2]
    print('cw')
    print(cw)
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = cw // 2, cw // 2 + 1
    else:
        cw1, cw2 = cw // 2, cw // 2

    ch = target.get_shape()[1] - refer.get_shape()[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = ch // 2, ch // 2 + 1
    else:
        ch1, ch2 = ch // 2, ch // 2

    return (ch1, ch2), (cw1, cw2)


def stretch_n(bands, lower_percent=5, higher_percent=95):
    out = np.zeros_like(bands)
    n = bands.shape[2]
    for i in range(n):
        a = 0
        b = 1
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = 0.2 + (intersection / (sum_ - intersection))
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def stick_all_train():
    print("let's stick all imgs together")
    s = 835
    x = np.zeros((5 * s, 5 * s, 8))
    y = np.zeros((5 * s, 5 * s, N_Cls))
    ids = sorted(DF.ImageId.unique())
    print(len(ids))
    for i in range(5):
        for j in range(5):
            id = ids[5 * i + j]

            img = M(id)
            img = stretch_n(img)
            print(img.shape, id, np.amax(img), np.amin(img))
            x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
            for z in range(N_Cls):
                y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
                    (img.shape[0], img.shape[1]), id, z + 1)[:s, :s]

    print(np.amax(y), np.amin(y))

    np.save('x_trn_%d' % N_Cls, x)
    np.save('y_trn_%d' % N_Cls, y)


def get_patches(img, msk, amt=10000, aug=True):
    is2 = int(1.0 * ISZ)
    xm, ym = img.shape[0] - is2, img.shape[1] - is2

    x, y = [], []

    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
    for i in range(amt):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]

        for j in range(N_Cls):
            sm = np.sum(ms[:, :, j])
            if 1.0 * sm / is2 ** 2 > tr[j]:
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                        ms = ms[::-1]
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]

                x.append(im)
                y.append(ms)

    x, y = 2 * np.transpose(x, (0, 3, 1, 2)) - 1, np.transpose(y, (0, 3, 1, 2))
    print(x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))
    return x, y


def make_val():
    print("let's pick some samples for validation")
    img = np.load('x_trn_%d.npy' % N_Cls)
    msk = np.load('y_trn_%d.npy' % N_Cls)
    x, y = get_patches(img, msk, amt=3000)

    np.save('x_tmp_%d' % N_Cls, x)
    np.save('y_tmp_%d' % N_Cls, y)


def get_unet():
    inputs = Input((8, ISZ, ISZ))

    conv1 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=23), data_format='channels_first')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=43), data_format='channels_first')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=26), data_format='channels_first')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_uniform(seed=45), data_format='channels_first')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=54), data_format='channels_first')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=25), data_format='channels_first')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=63), data_format='channels_first')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_uniform(seed=32), data_format='channels_first')(conv4)
    drop4 = Dropout(0.5, seed=38)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(drop4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=32), data_format='channels_first')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=27), data_format='channels_first')(conv5)
    drop5 = Dropout(0.5, seed=41)(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal(seed=28),
                 data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(drop5))
    merge6 = concatenate([drop4, up6], axis=1)
    conv6 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=39), data_format='channels_first')(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=21), data_format='channels_first')(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal(seed=11),
                 data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(conv6))
    merge7 = concatenate([conv3, up7], axis=1)
    conv7 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=17), data_format='channels_first')(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=53), data_format='channels_first')(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal(seed=63),
                 data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(conv7))
    merge8 = concatenate([conv2, up8], axis=1)
    conv8 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=29), data_format='channels_first')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=22), data_format='channels_first')(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=tf.keras.initializers.he_normal(seed=54),
                 data_format='channels_first')(UpSampling2D(size=(2, 2), data_format='channels_first')(conv8))
    merge9 = concatenate([conv1, up9], axis=1)
    conv9 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=67), data_format='channels_first')(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=56), data_format='channels_first')(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer=tf.keras.initializers.he_normal(seed=64), data_format='channels_first')(conv9)
    conv10 = Conv2D(num_cls, (1, 1), strides=1, activation='sigmoid', data_format='channels_first')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=[jaccard_coef,'accuracy'])

    return model


def calc_jacc(model1):
    img = np.load('x_tmp_%d.npy' % N_Cls)
    msk = np.load('y_tmp_%d.npy' % N_Cls)

    prd = model1.predict(img, batch_size=4)
    print(prd.shape, msk.shape)
    avg, trs = [], []

    for i in range(N_Cls):
        t_msk = msk[:, i, :, :]
        t_prd = prd[:, i, :, :]
        t_msk = t_msk.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])

        m, b_tr = 0, 0
        for j in range(10):
            tr = j / 10.0
            pred_binary_mask = t_prd > tr

            jk = jaccard_score(t_msk, pred_binary_mask,average='weighted')
            if jk > m:
                m = jk
                b_tr = tr
        print(i, m, b_tr)
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / 10.0
    return score, trs


def mask_for_polygons(polygons, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def mask_to_polygons(mask, epsilon=5, min_area=1.):
    contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()

    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1

    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])

    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)

    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)

        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def get_scalers(im_size, x_max, y_min):
    h, w = im_size
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


def train_net():
    global results
    print("start train net")
    x_val, y_val = np.load('x_tmp_%d.npy' % N_Cls), np.load('y_tmp_%d.npy' % N_Cls)
    img = np.load('x_trn_%d.npy' % N_Cls)
    msk = np.load('y_trn_%d.npy' % N_Cls)

    x_trn, y_trn = get_patches(img, msk)

    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights/unet_tmp.hdf5', monitor='loss', save_best_only=True)
    for i in range(1):
        results = model.fit(x_trn, y_trn, batch_size=64, epochs=15, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint], validation_data=(x_val, y_val))
        del x_trn
        del y_trn
        x_trn, y_trn = get_patches(img, msk)
        score, trs = calc_jacc(model)
        print('val jk', score)
        model.save_weights('weights/unet_10_jk%.4f' % score)

    return model,results


def predict_id(id, model, trs):
    img = M(id)
    x = stretch_n(img)

    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((N_Cls, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ] = tmp[j]

    for i in range(N_Cls):
        prd[i] = prd[i] > trs[i]

    return prd[:, :img.shape[0], :img.shape[1]]


def predict_test(model, trs):
    print("predict test")
    for i, id in enumerate(sorted(set(SB['ImageId'].tolist()))):
        msk = predict_id(id, model, trs)
        np.save('msk/10_%s' % id, msk)
        if i % 100 == 0: print(i, id)


def make_submit():
    print("make submission file")
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    print(df.head())
    for idx, row in df.iterrows():
        id = row[0]
        kls = row[1] - 1

        msk = np.load('msk/10_%s.npy' % id)[kls]
        pred_polygons = mask_to_polygons(msk)
        x_max = GS.loc[GS['ImageId'] == id, 'Xmax'].to_numpy()[0]
        y_min = GS.loc[GS['ImageId'] == id, 'Ymin'].to_numpy()[0]

        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)

        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))

        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
        if idx % 100 == 0:
            pass
    print(df.head())
    df.to_csv('subm/1.csv', index=False)


def check_predict(id='6120_2_2'):
    model_1 = get_unet()
    model_1.load_weights('weights/unet_10_jk0.1133')

    msk = predict_id(id, model_1, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])
    img = M(id)

    plt.figure()

    ax1 = plt.subplot(131)
    ax1.set_title('image ID:6120_2_2')
    ax1.imshow(img[:, :, 5], cmap=plt.get_cmap('gist_ncar'))
    ax2 = plt.subplot(132)
    ax2.set_title('predict pixels')
    ax2.imshow(msk[9], cmap=plt.get_cmap('gray'))
    ax3 = plt.subplot(133)
    ax3.set_title('predict polygones')
    ax3.imshow(mask_for_polygons(mask_to_polygons(msk[9], epsilon=1), img.shape[:2]), cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == '__main__':
    stick_all_train()
    make_val()
    model1, res = train_net()
    score, trs = calc_jacc(model1)
    predict_test(model1, trs)
    make_submit()
    check_predict()


    plt.figure(1)
    plt.plot(res.history['loss'])
    plt.plot(res.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training','testing'], loc="upper left")
    plt.show()

    plt.figure(2)
    plt.plot(res.history['jaccard_coef'])
    plt.plot(res.history['val_jaccard_coef'])

    plt.title('Jaccard_coefficient')
    plt.ylabel('IoU')
    plt.xlabel('epochs')
    plt.legend(['training','testing'], loc="upper left")
    plt.show()

    plt.figure(3)
    plt.plot(res.history['accuracy'])
    plt.plot(res.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['training', 'testing'], loc="upper left")
    plt.show()


