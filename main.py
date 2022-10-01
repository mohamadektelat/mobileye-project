try:
    from Configurations import *
except ImportError:
    print("Need to fix the installation")
    raise


def extract_dataset_file(path):
    """
    This function is responsible for extracting the data from the h5 file.
    :param path: The path for the h5 file.
    :return: This function returns numpy array with the rows from dataset.
    """
    df = pd.read_hdf(path)
    return np.array(df[df.keys()])


def rgb_convolve(image, kernel):
    red = convolve(image[:, :, 0], kernel)
    green = convolve(image[:, :, 1], kernel)
    return red, green


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    # Our kernel, we used 5*5 kernel for better run time.
    kernel = (1 / 13) * np.array([[-1, -1, -1, -1, -1],
                                  [-1, -1, 4, -1, -1],
                                  [-1, 4, 4, 4, -1],
                                  [-1, -1, 4, -1, -1],
                                  [-1, -1, -1, -1, -1]])

    conv_im1, conv_im2 = rgb_convolve(c_image, kernel)
    # Peak the local maximums after normalizing with Maximum filter method.
    red_image = peak_local_max(conv_im1, min_distance=15, threshold_abs=0.2, threshold_rel=0.2)
    green_image = peak_local_max(conv_im2, min_distance=15, threshold_abs=0.28, threshold_rel=0.29)
    return red_image, green_image


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def filter_method(red_list, green_list, image, data, image_path):
    for i in red_list:
        if image[i[0]][i[1]][0] > image[i[0]][i[1]][1] + 0.05 and image[i[0]][i[1]][0] > image[i[0]][i[1]][2] + 0.05:
            plt.plot(int(i[1] * 1), int(i[0] * 1), '+', color='r', markersize=5)
            data.append(["Red", (i[1], i[0]), image_path])

        print(i[1], i[0])
    for i in green_list:
        if image[i[0]][i[1]][1] > image[i[0]][i[1]][0] + 0.03 and image[i[0]][i[1]][1] > image[i[0]][i[1]][2]:
            plt.plot(int(i[1] * 1), int(i[0] * 1), '+', color='g', markersize=5)
            data.append(["Green", (i[1], i[0]), image_path])


def test_find_tfl_lights(image_path, data, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    # image = np.array(Image.open(image_path))
    image = plt.imread(image_path)
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    plt.figure(56)
    plt.clf()
    h = plt.subplot(111)
    plt.imshow(image)
    plt.figure(57)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)
    plt.imshow(image)

    # Our filtering operation
    red_list, green_list = find_tfl_lights(image)

    # filter the list
    filter_method(red_list, green_list, image, data, image_path)

    # Saving the Processed image to file(for debugging).
    # plt.savefig(f"procesed_images\{image_path}")

    plt.show()


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    counter = 0
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    # To do: change the directory according to your computer!!!
    default_base = r"Test_for_me"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '**/*_leftImg8bit.png'))

    print(len(flist))
    data = []

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, data, json_fn)

    col_names = ["path", "x", "y", "zoom", "col"]
    table = pd.DataFrame(columns=col_names, data=data)
    print(table)
    table.to_csv('table.csv')
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


def check_angle(top_right_point, top_left_point, bottom_right_point, bottom_left_point):
    counter = 0
    if top_right_point[0] == 250 and top_right_point[1] == 170 and top_right_point[2] == 30 and top_right_point[3] == 255:
        counter += 1
    if top_left_point[0] == 250 and top_left_point[1] == 170 and top_left_point[2] == 30 and top_left_point[3] == 255:
        counter += 1
    if bottom_right_point[0] == 250 and bottom_right_point[1] == 170 and bottom_right_point[2] == 30 and \
            bottom_right_point[3] == 255:
        counter += 1
    if bottom_left_point[0] == 250 and bottom_left_point[1] == 170 and bottom_left_point[2] == 30 and \
            bottom_left_point[3] == 255:
        counter += 1
    return counter


def check_bounds(x0, y0, x1, y1):
    if x0 > 2048:
        x0 = 2047
    if x1 > 2048:
        x1 = 2047
    if y0 > 1024:
        y0 = 1023
    if y1 > 1024:
        y1 = 1023
    return x0, y0, x1, y1


def crop_and_certificate(splitted_name: list, x: int, y: int, zoom_factor: float, x_factor: int,
                         y_left_factor: int, y_right_factor: int, image: Image) -> (Image, str):
    """
    This Function is responsible for cropping the image according to the x and y factors and zoom factor,then
    the function checks the labeled image to classificate the cropped image, whether traffic light or not
    :param splitted_name: The name of the image splitted according to "_" separator.
    :param x: The x coordinate.
    :param y: The y coordinate.
    :param zoom_factor: The zoom factor to crop perfectly.
    :param x_factor: The x left and right factor.
    :param y_left_factor: The y left factor.
    :param y_right_factor: The y right factor.
    :param image: The image to crop.
    :return: This function returns two arguments: the cropped image and flag with the classification whether the
             cropped image is a traffic light or not.
    """
    x0, y0, x1, y1 = check_bounds(int(x - x_factor * zoom_factor), int(y - y_left_factor * zoom_factor),int(x + x_factor * zoom_factor),
                 int(y + y_left_factor * zoom_factor))
    if_traffic_light = False
    if_ignore = False

    # open the selected labeled image
    labeled_image = Image.open("gtFine/train" + "/" + splitted_name[0] + "/" + splitted_name[0] + "_" +
                               splitted_name[1] + "_" + splitted_name[2] + "_" + "gtFine_color.png")

    image_array = np.asarray(labeled_image)[int(y)][int(x)]
    if image_array[0] == 250 and image_array[1] == 170 and image_array[2] == 30 and image_array[3] == 255:
        first_safe_zone = np.asarray(labeled_image)[int(y)][int(x) + 5]
        second_safe_zone = np.asarray(labeled_image)[int(y)][int(x) - 5]
        third_safe_zone = np.asarray(labeled_image)[int(y) + 5][int(x)]
        fourth_safe_zone = np.asarray(labeled_image)[int(y) - 5][int(x)]
        if (first_safe_zone[0] == 250 and first_safe_zone[1] == 170 and first_safe_zone[2] == 30
            and first_safe_zone[3] == 255) and (second_safe_zone[0] == 250 and second_safe_zone[1] == 170 and
                                                second_safe_zone[2] == 30 and second_safe_zone[3] == 255) and \
                (third_safe_zone[0] == 250 and third_safe_zone[1] == 170 and third_safe_zone[2] == 30 and
                 third_safe_zone[3] == 255) and (fourth_safe_zone[0] == 250 and fourth_safe_zone[1] == 170
                                                 and fourth_safe_zone[2] == 30 and fourth_safe_zone[3] == 255) and \
                (check_angle(np.asarray(labeled_image)[y0][x0], np.asarray(labeled_image)[y1][x0],
                             np.asarray(labeled_image)[y0][x1], np.asarray(labeled_image)[y1][x1]) == 0):
            pass
        else:
            if_ignore = True
        if_traffic_light = True
    box = (x - x_factor * zoom_factor, y - y_left_factor * zoom_factor, x + x_factor * zoom_factor,
           y + y_right_factor * zoom_factor)
    return image.crop(box=box).resize((150, 300)), if_traffic_light, if_ignore


def setting_up_a_crop(directory: str) -> None:
    """
    This function is responsible for setting the crop factors according to color.
    :param directory: The base directory for thew image.
    :return: None.
    """
    headers = ["Cropped Image path", "Original Image Path", "Is Traffic Light", "Is Ignore", "Color"]
    data = []
    # TODO debug
    counter = 1
    dataset = extract_dataset_file('attention_results.h5')
    path = directory + dataset[0][0].split("_")[0] + "/" + dataset[0][0]
    image = Image.open(path)
    for index, row in enumerate(dataset):
        print(index)
        if np.isnan(row[1]) and np.isnan(row[2]):
            continue
        if directory + row[0].split("_")[0] + "/" + row[0] != path:
            path = directory + row[0].split("_")[0] + "/" + row[0]
            image = Image.open(path)
            counter = 1
        if row[4] == 'r':
            x_factor = 50
            y_left_factor = 60
            y_right_factor = 350
            cropped_image, if_traffic_light, if_ignore = crop_and_certificate(row[0].split("_"), row[1], row[2], row[3],
                                                                              x_factor, y_left_factor, y_right_factor,
                                                                              image)
        else:
            x_factor = 50
            y_left_factor = 80
            y_right_factor = 20
            cropped_image, if_traffic_light, if_ignore = crop_and_certificate(row[0].split("_"), row[1], row[2], row[3],
                                                                              x_factor, y_left_factor, y_right_factor,
                                                                              image)
        if if_traffic_light:
            if not if_ignore:
                path = f"cropped_images/Traffic Light/{row[4]}_{if_traffic_light}_{if_ignore}_{counter}_{row[0]}"
                cropped_image.save(path, quality=95)
                data.append([path, path, if_traffic_light, if_ignore, row[4]])
        else:
            path = f"cropped_images/Not Traffic Light/{row[4]}_{if_traffic_light}_{if_ignore}_{counter}_{row[0]}"
            cropped_image.save(path, quality=95)
            data.append([path, path, if_traffic_light, if_ignore, row[4]])
        # crop = Image.open(f"cropped_images/{row[4]}_{if_traffic_light}_{if_ignore}_{counter}_{row[0]}")
        # plt.imshow(crop)
        # plt.title(f"Traffic Light: {if_traffic_light}" + "\n" + f"Ignore: {if_ignore}")
        # plt.show()
        counter += 1

    df = pd.DataFrame(data, columns=headers)
    df.to_hdf('cropped_table.h5', key='df', mode='w')
    print(df)

    # print(tabulate(data, headers=headers, showindex='always'))


if __name__ == '__main__':
    # main()
    # image = plt.imread("gtFine/test/berlin/berlin_000022_000019_gtFine_instanceIds.png")
    # plt.imshow(image != 0)
    # plt.show()
    base_dir = "images/train/"
    setting_up_a_crop(base_dir)
    # df = extract_dataset_file("cropped_table.h5")
    # print(len(df))
