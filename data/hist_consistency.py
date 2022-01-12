from cv2 import distanceTransform, threshold, subtract, connectedComponents, watershed, DIST_L2, erode
import numpy as np
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def postproc(mask, sure_fg_threshold=0.5):
    """
    Isolates pores in the binary mask from the background (pore walls, etc.) using watershed algorithm.

    Parameters
    ----------
    mask: ndarray(dtype='uint8') of shape (H, W).
        Binary mask with black pores (pixel=0) and white background (pixel=1).
    sure_fg_threshold: float, optional, default=0.5.
        Threshold value is used to construct sure foreground marker for watershed algorithm.
        Applied to TOTAL distance map of input binary mask.

    Returns
    -------
    ndarray(dtype='uint8') of shape (H, W).
        Result of watershed algorithm applied to input binary mask.
    """

    # invert input binary mask
    mask_postproc = np.invert(mask.astype('bool')).astype('uint8') * 255
    # 3D-version of input 2D-binary mask for watershed function
    mask_postproc_3d = np.stack([mask_postproc] * 3, axis=2)
    # sure background marker for watershed algorithm
    sure_bg = mask_postproc

    # construct sure foreground marker for watershed algorithm
    # using Euclidean distance transform followed by thresholding
    dist_transform = distanceTransform(mask_postproc, DIST_L2, 5)
    _, sure_fg = threshold(dist_transform, sure_fg_threshold * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # construct final markers for watershed algorithm
    unknown = subtract(sure_bg, sure_fg)
    _, markers = connectedComponents(sure_fg)
    markers = markers + 10
    markers[unknown == 255] = 0
    # apply watershed algorithm
    markers = watershed(mask_postproc_3d, markers)
    mask_postproc_3d[markers == -1] = [0, 255, 255]

    # resulting binary mask corresponds to the first channel
    # convert it to input binary mask's style
    return mask_postproc_3d[:, :, 0] + 1


def postproc2(mask, sure_fg_threshold=0.5):
    """
    Isolates pores in the binary mask from the background (pore walls, etc.) using watershed algorithm.

    Parameters
    ----------
    mask: ndarray(dtype='uint8') of shape (H, W).
        Binary mask with black pores (pixel=0) and white background (pixel=1).
    sure_fg_threshold: float, optional, default=0.17.
        Threshold value is used to construct sure foreground marker for watershed algorithm.
        Applied to INDIVIDUAL distance map of each connected area (pore) of input binary mask.

    Returns
    -------
    ndarray(dtype='uint8') of shape (H, W).
        Result of watershed algorithm applied to input binary mask.
    """

    # invert input binary mask
    mask_postproc = np.invert(mask.astype('bool')).astype('uint8') * 255
    # 3D-version of input 2D-binary mask for watershed function
    mask_postproc_3d = np.stack([mask_postproc] * 3, axis=2)
    # sure background marker for watershed algorithm
    sure_bg = mask_postproc

    # construct sure foreground marker for watershed algorithm using Euclidean distance transform
    # followed by thresholding applied to each connected area (pore) of input binary mask
    sure_fg = np.zeros_like(mask_postproc)
    mask_postproc_labeled = label(mask_postproc, connectivity=1, background=mask_postproc.min().item())
    for i in range(1, mask_postproc_labeled.max() + 1):
        mask_postproc_piece = (mask_postproc_labeled == i).astype('uint8') * 255
        dist_transform = distanceTransform(mask_postproc_piece, DIST_L2, 5)
        _, sure_fg_piece = threshold(dist_transform, sure_fg_threshold * dist_transform.max(), 255, 0)
        sure_fg_piece = sure_fg_piece.astype('uint8')
        sure_fg += sure_fg_piece

    # construct final markers for watershed algorithm
    unknown = subtract(sure_bg, sure_fg)
    _, markers = connectedComponents(sure_fg)
    markers = markers + 10
    markers[unknown == 255] = 0
    # apply watershed algorithm
    markers = watershed(mask_postproc_3d, markers)
    mask_postproc_3d[markers == -1] = [0, 255, 255]

    # resulting binary mask corresponds to the first channel
    # convert it to input binary mask's style
    return mask_postproc_3d[:, :, 0] + 1


def postproc3(mask, coarse_threshold=0.5, fine_threshold=0.2):
    """
    Isolates pores in the binary mask from the background (pore walls, etc.) using watershed algorithm.

    Parameters
    ----------
    mask: ndarray(dtype='uint8') of shape (H, W).
        Binary mask with black pores (pixel=0) and white background (pixel=1).
    sure_fg_threshold: float, optional, default=0.17.
        Threshold value is used to construct sure foreground marker for watershed algorithm.
        Applied to INDIVIDUAL distance map of each connected area (pore) of input binary mask.

    Returns
    -------
    ndarray(dtype='uint8') of shape (H, W).
        Result of watershed algorithm applied to input binary mask.
    """

    fine_kernel_1 = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]], dtype='uint8')
    fine_kernel_2 = np.array([[1, 0, 1],
                              [0, 1, 0],
                              [1, 0, 1]], dtype='uint8')
    coarse_kernel_1 = np.array([[0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 1, 0, 0],
                                [0, 1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 1, 1, 0],
                                [0, 0, 1, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]], dtype='uint8')
    coarse_kernel_2 = coarse_kernel_1.T

    # invert input binary mask
    mask_postproc = np.invert(mask.astype('bool')).astype('uint8') * 255
    # 3D-version of input 2D-binary mask for watershed function
    mask_postproc_3d = np.stack([mask_postproc] * 3, axis=2)
    # sure background marker for watershed algorithm
    sure_bg = mask_postproc

    # construct sure foreground marker for watershed algorithm using Euclidean distance transform
    # followed by thresholding applied to each connected area (pore) of input binary mask
    sure_fg = np.zeros_like(mask_postproc)
    mask_postproc_labeled = label(mask_postproc, connectivity=1, background=mask_postproc.min().item())
    for i in range(1, mask_postproc_labeled.max() + 1):
        mask_postproc_piece = (mask_postproc_labeled == i).astype('uint8')
        area = mask_postproc_piece.sum()
        while mask_postproc_piece.sum() > area * coarse_threshold:
            mask_postproc_piece = erode(mask_postproc_piece, coarse_kernel_1)
        while mask_postproc_piece.sum() > area * fine_threshold:
            mask_postproc_piece = erode(mask_postproc_piece, fine_kernel_1)
        sure_fg_piece = mask_postproc_piece * 255
        sure_fg += sure_fg_piece

    # construct final markers for watershed algorithm
    unknown = subtract(sure_bg, sure_fg)
    _, markers = connectedComponents(sure_fg)
    markers = markers + 10
    markers[unknown == 255] = 0
    # apply watershed algorithm
    markers = watershed(mask_postproc_3d, markers)
    mask_postproc_3d[markers == -1] = [0, 255, 255]

    # resulting binary mask corresponds to the first channel
    # convert it to input binary mask's style
    return mask_postproc_3d[:, :, 0] + 1


def area(mask):
    """
    Measures area of all connected regions (pores) in the binary mask.

    Parameters
    ----------
    mask: ndarray(dtype='uint8') of shape (H, W).
        Binary mask with black pores (pixel=0) and white background (pixel=1).

    Returns
    -------
    ndarray(dtype='uint32') of shape (N, ).
        Array of area values of all connectned regions (pores).
    """

    labeled_mask = label(mask, connectivity=1, background=mask.max().item())
    props = regionprops(labeled_mask)
    areas = [prop.area for prop in props]
    return np.array(areas, dtype=np.uint32)


def ks_test(areas1, areas2, bin_width=1000, plot_hist=False):
    """
    Computes histograms of two datasets and performs Kolmogorov-Smirnov two-sample test consistency of theirs.
    Optionally plots the histograms corresponding to two arrays of area values areas1 and areas2.

    Parameters
    ----------
    areas1: ndarray(dtype='uint32') of shape (N, ).
        Array of area values of all connectned regions (pores) in the binary mask.
    areas2: ndarray(dtype='uint32') of shape (M, ).
        Array of area values of all connectned regions (pores) in the other binary mask.
    bin_width: int, optional, default=1000.
        Width of histogram bins.
    plot_hist: bool, optional, default=False.
        If True, plots the histograms. Otherwise do not plot the histograms.

    Returns
    -------
    float.
        Kolmogorov-Smirnov two-sample statistic value.
    """

    # Define boundaries of joint range of areas1 and areas2.
    # Boundaries are multiple of bin_width.
    # Calculate number of histogram bins.
    areas_total = np.concatenate([areas1, areas2])
    range_left = areas_total.min().item() // int(bin_width) * int(bin_width)
    range_right = (areas_total.max().item() // int(bin_width) + 1) * int(bin_width)
    bin_num = (range_right - range_left) // int(bin_width)
    # Compute histograms of areas1 and areas2. They have the same range and width of bins.
    hist1 = np.histogram(areas1, bins=bin_num, range=(range_left, range_right))
    hist2 = np.histogram(areas2, bins=bin_num, range=(range_left, range_right))
    # If plot_hist=True, plot the histograms on one graph.
    if plot_hist:
        plot_two_histograms(hist1[1][:-1], hist1[0], hist2[0], 'mask1', 'mask2', bin_width)
    # Compute Kolmogorov-Smirnov two-sample statistic.
    return ks_2samp(hist1[0], hist2[0])[0]


def intersection(areas1, areas2, bin_width=1000, plot_hist=False):
    """
    Computes histograms of two datasets and performs intersection consistency of theirs.
    Optionally plots the histograms corresponding to two arrays of area values areas1 and areas2.

    Parameters
    ----------
    areas1: ndarray(dtype='uint32') of shape (N, ).
        Array of area values of all connectned regions (pores) in the binary mask.
    areas2: ndarray(dtype='uint32') of shape (M, ).
        Array of area values of all connectned regions (pores) in the other binary mask.
    bin_width: int, optional, default=1000.
        Width of histogram bins.
    plot_hist: bool, optional, default=False.
        If True, plots the histograms. Otherwise do not plot the histograms.

    Returns
    -------
    float.
        Jaccard coefficient, or IoU value, of two histograms
        corresponding to two arrays of area values areas1 and areas2.
    """

    # Define boundaries of joint range of areas1 and areas2.
    # Boundaries are multiple of bin_width.
    # Calculate number of histogram bins.
    areas_total = np.concatenate([areas1, areas2])
    range_left = areas_total.min().item() // int(bin_width) * int(bin_width)
    range_right = (areas_total.max().item() // int(bin_width) + 1) * int(bin_width)
    bin_num = (range_right - range_left) // int(bin_width)
    # Compute histograms of areas1 and areas2. They have the same range and width of bins.
    hist1 = np.histogram(areas1, bins=bin_num, range=(range_left, range_right))
    hist2 = np.histogram(areas2, bins=bin_num, range=(range_left, range_right))
    # If plot_hist=True, plot the histograms on one graph.
    if plot_hist:
        plot_two_histograms(hist1[1][:-1], hist1[0], hist2[0], 'mask1', 'mask2', bin_width)
    # Compute Jaccard coefficient, or IoU value, of two histograms.
    return np.sum(np.minimum(hist1[0], hist2[0])) / np.sum(np.maximum(hist1[0], hist2[0])).item()


def plot_two_histograms(x, height1, height2, label1, label2, bin_width=1000):
    """
    Plots two histograms on one graph.

    Parameters
    ----------
    x: ndarray of shape (N, ).
        The x coordinates of the bars.
    height1: ndarray of shape (N, ).
        Bar heights of the first histogram.
    height2: ndarray of shape (N, ).
        Bar heights of the other histogram.
    label1: str.
        Label of the first histogram.
    label2: str.
        Label of the other histogram.
    bin_width: int, optional, default=1000.
        Width of histogram bins.
    """

    plt.bar(x, height1, alpha=0.5, label=label1, color='r', align='edge', width=bin_width)
    plt.bar(x, height2, alpha=0.5, label=label2, color='b', align='edge', width=bin_width)
    plt.xlabel('Area, px^2')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
