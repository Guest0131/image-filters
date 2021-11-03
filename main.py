import sys
import numpy as np
from math import log10, sqrt, pow, exp
from PIL import Image
import cv2


def MSE(input_file_1, input_file_2):
    """
    Calculating MSE 

    Args:
        input_file_1: First picture
        input_file_2: Second picture
    """
    err = np.sum((input_file_1.astype("float") - input_file_2.astype("float")) ** 2)
    err /= float(input_file_1.shape[0] * input_file_2.shape[1])
    return err


def Gausskernel(sigma_d, size=10):
    """
    Calculate the kernel of the Gaussian convolution 

    Args:
        sigma: sigma value
    """
    gausskernel=np.zeros((size,size), np.float32)
    for i in range(size):
        for j in range(size):
            norm=pow(i - 1, 2)+pow(j - 1, 2)
            gausskernel[i, j] = exp (-norm / (2 * pow (sigma_d, 2))) # Calculate of Gaussian convolution
        sum = np.sum (gausskernel) # summ
        kernel = gausskernel / sum # normalization
    return kernel


def PSNR(input_file_1, input_file_2):
    """
    Calculating PSNR 

    Args:
        input_file_1: First picture
        input_file_2: Second picture
    """
    mse = MSE(input_file_1, input_file_2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def SSIM(input_file_1, input_file_2):
    """
    Calculating SSIM 

    Args:
        input_file_1: First picture
        input_file_2: Second picture
    """
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    input_file_1 = input_file_1.astype(np.float64)
    input_file_2 = input_file_2.astype(np.float64)
    kernel = Gausskernel(1.5, 11)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(input_file_1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(input_file_2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(input_file_1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(input_file_2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(input_file_1 * input_file_2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []

    return Image.fromarray(data_final)

def gaussian(x,sigma):
    return (1.0/(2*np.pi*(sigma**2)))*np.exp(-(x**2)/(2*(sigma**2)))

def distance(x1,y1,x2,y2):
    return np.sqrt(np.abs((x1-x2)**2+(y1-y2)**2))

def bilateralfilter(image, texture, sigma_s, sigma_r):
    r = int(np.ceil(3 * sigma_s))
    # Image padding
    if image.ndim == 3:
        h, w, ch = image.shape
        I = np.pad(image, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float32)
    elif image.ndim == 2:
        h, w = image.shape
        I = np.pad(image, ((r, r), (r, r)), 'symmetric').astype(np.float32)
    else:
        print('Input image is not valid!')
        return image
    # Check texture size and do padding
    if texture.ndim == 3:
        ht, wt, cht = texture.shape
        if ht != h or wt != w:
            print('The guidance image is not aligned with input image!')
            return image
        T = np.pad(texture, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.int32)
    elif texture.ndim == 2:
        ht, wt = texture.shape
        if ht != h or wt != w:
            print('The guidance image is not aligned with input image!')
            return image
        T = np.pad(texture, ((r, r), (r, r)), 'symmetric').astype(np.int32)
    # Pre-compute
    output = np.zeros_like(image)
    scaleFactor_s = 1 / (2 * sigma_s * sigma_s)
    scaleFactor_r = 1 / (2 * sigma_r * sigma_r)
    # A lookup table for range kernel
    LUT = np.exp(-np.arange(256) * np.arange(256) * scaleFactor_r)
    # Generate a spatial Gaussian function
    x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
    kernel_s = np.exp(-(x * x + y * y) * scaleFactor_s)
    # Main body
    if I.ndim == 2 and T.ndim == 2:     # I1T1 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[np.abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    elif I.ndim == 3 and T.ndim == 2:     # I3T1 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif I.ndim == 3 and T.ndim == 3:     # I3T3 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif I.ndim == 2 and T.ndim == 3:     # I1T3 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    else:
        print('Something wrong!')
        return image

    # return np.clip(output, 0, 255)
    return output

class GaussianFilter():
    def _pad(self, x, k, mode="edge"):
        c = x.shape[-1]
        pad = [(k,), (k,)]
        per_chan = [np.pad(x[:, :, chan], pad, mode=mode) for chan in range(c)]
        return np.stack(per_chan, axis=2)

    def __init__(self, sigma, truncate=3.0):
        super().__init__()

        assert sigma >= 0., "[!] Standard deviation must be positive."

        self.sigma = float(sigma)
        self.var = self.sigma * self.sigma
        self.truncate = truncate

        self.k = self._compute_kernel_radius()
        self.kernel = self._sample_kernel_gaussian_function()

    def _compute_kernel_radius(self, trunc=True):
        if trunc:  # from [2]
            radius = int(self.truncate * self.sigma + 0.5)
        else:  # from [1]
            thresh = 0.005
            radius = int(np.floor(1.0 + 2.0*np.sqrt(-2.0*self.var*np.log(thresh)))) + 1
        return radius

    def _sample_kernel_pascal_triangle(self):
        from scipy.linalg import pascal
        kernel = pascal(2*self.k+1, kind='lower')[-1]
        kernel = kernel / np.sum(kernel)
        return kernel

    def _sample_kernel_gaussian_function(self):
        x = np.linspace(-self.k, self.k, 2*self.k+1)
        kernel = np.exp(-0.5 * x * x / self.var)
        kernel = kernel / np.sum(kernel)
        return kernel

    @staticmethod
    def _loop(h, w, c, k, kernel, x_pad, out):
        for row in range(h):
            for col in range(w):
                cumsum = np.zeros(c)
                for i in range(-k, k+1):
                    cumsum += kernel[i+k] * x_pad[row+k, col+i+k, :]
                out[row, col, :] = cumsum
        return out

    def _filter1d(self, x):
        h, w, c = x.shape
        x_pad = self._pad(x, self.k)
        out = np.empty_like(x, dtype="float64")
        out = self._loop(h, w, c, self.k, self.kernel, x_pad, out)
        return out

    def filter(self, x):
        if x.ndim == 2:
            x = x[:, :, np.newaxis]
        h, w, c = x.shape
        horiz = self._filter1d(x)
        vert = self._filter1d(horiz.transpose((1, 0, 2)))
        out = vert.transpose((1, 0, 2)).astype("uint8")
        return out.squeeze()


if __name__ == '__main__':

    # Read args from command line arguments
    args = sys.argv[1:]
    
    # Selecting the operating mode 
    if args[0] == "mse":
        original = cv2.imread(args[1])
        contrast = cv2.imread(args[2])
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
        print(MSE(original, contrast))
    if args[0] == "psnr":
        print(PSNR(cv2.imread(args[1], 0), cv2.imread(args[2], 0)))
    if args[0] == "ssim":
        original = cv2.imread(args[1])
        contrast = cv2.imread(args[2])
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
        print(SSIM(original, contrast))
    if args[0] == "median":
        img = Image.open(args[2]).convert("L")
        arr = np.array(img)
        removed_noise = median_filter(arr, int(args[1])) 
        img = removed_noise
        img.convert("RGB").save(args[3])

    if args[0] == "gauss":
        gauss_filter = GaussianFilter(float(args[1]))
        img = np.asarray(Image.open(args[2])).astype("uint8")
        blurred_gauss = Image.fromarray(gauss_filter.filter(img))
        blurred_gauss.convert("RGB").save(args[3])
        
    if args[0] == "bilateral":
        img = cv2.imread(args[3])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bf = bilateralfilter(img, img_gray, float(args[1]), float(args[2]))
        cv2.imwrite(args[4], img_bf)
        