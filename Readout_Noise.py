import glob as glob
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from settings import generate_int_ptc_paths
# from settings import create_rslt_json_file
from csv import writer
import scipy.ndimage as scn
import glob


def write_csvfile(name, list_name):
    """
    :param name:  Name of the csv file.

    :type name:  str

    :param list_name:  list of parameters to be included in csv file

    :type list_name: list

    """
    with open(name, "a", newline="") as f_object:
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(list_name)
        # Close the file object
        f_object.close()


def get_image(img_path, gain):
    """
    :param img_path: path to the image.

    :type img_path: str

    :param gain: high or low gain

    :type gain: str

    :return: Image numpy array

    :rtype: numpy array.

    """
    im = Image.open(img_path)
    # find th position to crop the images
    img_l = np.array(im.crop((0, 0, 4742, 4742)))
    img_h = np.array(im.crop((4742, 0, 9484, 4742)))
    # img = np.add(img, img_1)
    m, n = img_h.shape
    img_lg = img_l[4 : m - 4, 4 : n - 4]
    img_hg = img_h[4 : m - 4, 4 : n - 4]
    if gain == "low":
        return img_lg
    elif gain == "high":
        return img_hg


def get_bias_images(light_ptc_path, ptc_struct, gain):
    """

    :param light_ptc_path: path to light ptc images

    :type light_ptc_path: str

    :param ptc_struct: file name structure

    :type ptc_struct: stry

    :param gain: high or low gain

    :type gain: str

    :return: bias image numpy array

    :rtype: numpy array

    """
    # type is B or D1 or D2
    filename = light_ptc_path + ptc_struct + "_ZE_*tif"
    file_list = glob.glob(filename, recursive=True)
    # print(file_list)
    signal = np.array([get_image(file, gain) for file in file_list])
    return signal


def save_image(img, path_name, img_title):
    """

    :param img: image array

    :type img: numpy array

    :param path_name: path where image to be saved.

    :type path_name: str

    :param img_title: title of the image

    :type img_title: str

    """
    mean_img = np.mean(img)
    std_img = np.std(img)
    plt.title(path_name + "\n" + img_title)
    plt.imshow(img, cmap="jet")
    plt.colorbar()
    plt.clim(
        mean_img - 4 * std_img, mean_img + 4 * std_img,
    )
    plt.tight_layout()
    img_name = path_name + "/" + img_title + ".png"
    plt.savefig(img_name)
    # plt.show()
    plt.close()


def plot_histogram(img, path, rslt_path, gain):
    """

    :param img:  image array.

    :type img: numpy array.

    :param path: path to code.

    :type path: str.

    :param rslt_path: path.

    :type rslt_path: path to the result images.

    :param gain: high or low gain.

    :type gain: str.

    """
    max_v = np.max(img)
    min_v = np.min(img)
    if max_v > 16383:
        max_v = 16383
    if min_v < 0:
        min_v = 0
    nbin = 100
    plt.hist(img.flatten(), nbin, range=[0, 50], log=True)
    # histtype="step", log=True)
    plt.xlabel("Read out noise in  ADU/pix", fontsize=16)
    plt.ylabel("Number of pixels", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    name = rslt_path + "/" + "read_noise " + gain + " gain histogram.png"
    plt.title(
        path.replace("./data/", "") + " " + gain + " gain readout noise histogram"
    )
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

    # find the readout noise of pixel


def find_readout_noise(light_ptc_path, ptc_struct, gain):
    """

    :param light_ptc_path:  path to light ptc images

    :type light_ptc_path: str

    :param ptc_struct: image file structure

    :type ptc_struct: str

    :param gain: high or low gain

    :type gain: str

    :return: readout noise

    :rtype: float
    """
    diff_sig = []
    signal = get_bias_images(light_ptc_path, ptc_struct, gain)
    std_signal = np.std(signal, axis=0)
    return std_signal

    # find the bias signal


def find_bias(light_ptc_pat, ptc_struct, gain):
    """

    :param light_ptc_pat:  path to the light ptc images
    :type light_ptc_pat: str
    :param ptc_struct: image file structure
    :type ptc_struct: str
    :param gain: low or high gain
    :type gain: str
    :return: mean bias values and std in bias
    :rtype: float
    """
    signal = get_bias_images(light_ptc_pat, ptc_struct, gain)
    bias = np.mean(signal)
    std = np.std(signal)
    return bias, std

    # find the readout noise profile


def rn_profile(index, rn, gain, ptc_struct):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plt.suptitle(ptc_struct, fontsize=14)
    ax.set_title("Read noise at " + gain + " gain part", fontsize=16)
    ax.scatter(index, rn, marker=".")
    # ax.set_ylim([0, 2])
    # ax[0].scatter(mean_count, var_sig)
    ax.set_xlabel("Column number", fontsize=16)
    ax.set_ylabel("Read noise in ADU/pix", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    plt.show()
    return

    # find the rms and standard deviation of the noise


def rms_std(img):
    """

    :param img: image array

    :type img: numpy array

    :return: rms value, std, median value, pixels > 5RN, pixels > 8RN.

    :rtype: float
    """
    # find rms of the image
    img_sq = np.multiply(img, img)
    img_sq_fl = img_sq.flatten()
    N = len(img_sq_fl)
    img_sq_sum = np.sum(img_sq_fl)
    rms_img = np.round(np.sqrt(img_sq_sum / N), 3)
    std_img = np.round(np.std(img), 3)
    med_img = np.round(np.median(img), 3)
    # find number of pixels greater than 5 rms
    loc_5 = np.where((img > 5))
    # print(loc_5)
    loc_8 = np.where((img > 8))
    pix_5 = len(loc_5[0])
    pix_8 = len(loc_8[0])
    return rms_img, std_img, med_img, pix_5, pix_8


def get_rastor_rn(light_ptc_path, ptc_struct, gain, n):
    """

    :param light_ptc_path: Light ptc path

    :type light_ptc_path: str

    :param ptc_struct: Image file structure

    :type ptc_struct: str

    :param gain: high or low gain

    :type gain: str

    :param n: raster scan size n x n

    :type n: int

    :return: raster scanned readout noise

    :rtype: float
    """
    signal = get_bias_images(light_ptc_path, ptc_struct, gain)
    std_signal = np.std(signal, axis=0)
    # print(std_signal.shape)
    # print('signal is ')
    # print(std_signal[1, :])
    sqr_signal = np.multiply(std_signal, std_signal)
    # print(sqr_signal.shape)
    # print('squared signal is ')
    # print(sqr_signal[1, :])
    # print('squared signal shape is', sqr_signal.shape)
    # here Initially I didn't do the square signal.
    rast_rn = scn.uniform_filter(sqr_signal, size=n)
    # print(rast_rn.shape)
    # print('Raster signal shape is', rast_rn.shape)
    sqr_rast = np.sqrt(rast_rn)
    print(sqr_rast.shape)
    return sqr_rast


def plot_RN_raster(path, light_ptc_path, ptc_struct, gain):
    # high_rn_1 = find_readout_noise(light_ptc_path, ptc_struct, gain)
    high_rn_1 = get_rastor_rn(light_ptc_path, ptc_struct, gain, 1)
    high_rn_3 = get_rastor_rn(light_ptc_path, ptc_struct, gain, 3)
    high_rn_5 = get_rastor_rn(light_ptc_path, ptc_struct, gain, 5)
    rms_h1, std_h1, med_h1, pix_h51, pix_h81 = rms_std(high_rn_1)
    rms_h3, std_h3, med_h3, pix_h53, pix_h83 = rms_std(high_rn_3)
    rms_h5, std_h5, med_h5, pix_h55, pix_h85 = rms_std(high_rn_5)
    split_path = path.split("/")
    list_rn = [
        split_path[-1],
        rms_h1,
        med_h1,
        std_h1,
        rms_h3,
        med_h3,
        std_h3,
        rms_h5,
        med_h5,
        std_h5,
    ]
    list_pixels = [split_path[-1], pix_h51, pix_h81, pix_h53, pix_h83, pix_h55, pix_h85]
    if gain == "high":
        write_csvfile("./RN_high_gain.csv", list_rn)
        write_csvfile("./RN_high_gain_pixels.csv", list_pixels)
    else:
        write_csvfile("./RN_low_gain.csv", list_rn)
        write_csvfile("./RN_low_gain_pixels.csv", list_pixels)
    plt.figure(1)
    split_path = path.split("/")
    title = split_path[-1] + ' ' + gain + " gain RN"
    plt.title(title, fontsize=16)
    plt.hist(
        high_rn_1.flatten(),
        240,
        range=[0, 60],
        alpha=0.5,
        log=True,
        label=gain
        + " gain 1 x 1 rms,std RN = "
        + str(np.round(rms_h1, 2))
        + ", "
        + str(np.round(std_h1, 2)),
    )
    plt.hist(
        high_rn_3.flatten(),
        240,
        range=[0, 60],
        alpha=0.5,
        log=True,
        label=gain
        + " gain 3 x 3 rms,std RN = "
        + str(np.round(rms_h3, 2))
        + ", "
        + str(np.round(std_h3, 2)),
    )
    plt.hist(
        high_rn_5.flatten(),
        240,
        range=[0, 60],
        alpha=0.5,
        log=True,
        label=gain
        + " gain 5 x 5 rms,std RN = "
        + str(np.round(rms_h5, 2))
        + ", "
        + str(np.round(std_h5, 2)),
    )
    plt.legend(fontsize=14)
    plt.xlabel("Readout noise (ADU/pix)", fontsize=16)
    plt.ylabel("Counts", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    img_name = result_path + '/' + title + '.png'
    plt.savefig(img_name)
    plt.close()
    # plt.show()


# rootdir = './data/'
# for file in os.listdir(rootdir):
#     d = os.path.join(rootdir, file)
#     if os.path.isdir(d):
#         d.replace('\\', '/')
#         path, result_path, ptc_campaign, ptc_struct, light_ptc_path = generate_int_ptc_paths(d)
#         print(path)
#         plot_RN_raster(path, light_ptc_path, ptc_struct, "low")
#         plot_RN_raster(path, light_ptc_path, ptc_struct, "high")



