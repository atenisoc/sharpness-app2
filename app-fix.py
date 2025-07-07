from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter
import numpy as np
import cv2
from skimage import restoration
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 処理関数（すべてグレースケール）
def apply_unsharp(img):
    return img.convert('L').filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

def apply_median_blur(img):
    np_img = np.array(img.convert('L'))
    blurred = cv2.medianBlur(np_img, 3)
    return Image.fromarray(blurred)

def apply_laplacian_sharpen(img):
    np_img = np.array(img.convert('L'))
    laplacian = cv2.Laplacian(np_img, cv2.CV_64F)
    sharpened = np.clip(np_img - 0.5 * laplacian, 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened)

def apply_clahe(img):
    np_img = np.array(img.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(np_img)
    return Image.fromarray(enhanced)

def apply_deconvolution(img):
    np_img = np.array(img.convert('L')) / 255.0
    psf = np.ones((5, 5)) / 25
    deconv, _ = restoration.unsupervised_wiener(np_img, psf)
    deconv = np.clip(deconv * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(deconv)

# 評価関数
def calculate_laplacian_energy(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return round(np.var(laplacian), 2)

def calculate_psnr(original_path, processed_path):
    orig = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    proc = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
    return round(peak_signal_noise_ratio(orig, proc), 2)

def calculate_ssim(original_path, processed_path):
    orig = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    proc = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
    ssim_score, _ = structural_similarity(orig, proc, full=True)
    return round(ssim_score, 3)

def process_and_save(img, func, filename):
    result_img = func(img)
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    result_img.convert('RGB').save(result_path, 'JPEG')
    return result_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='ファイルが見つかりません')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='ファイルが選択されていません')
        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            base_filename = os.path.splitext(original_filename)[0]
            jpg_filename = base_filename + '.jpg'
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], jpg_filename)

            img = Image.open(file)
            img_rgb = img.convert('RGB')
            img_rgb.save(upload_path, 'JPEG')

            evaluations = {}

            process_list = [
                ('01_unsharp_mask.jpg', apply_unsharp, '輪郭が強調された'),
                ('02_median.jpg',       apply_median_blur, 'ノイズが抑えられたが少しぼやけた'),
                ('03_laplacian.jpg',    apply_laplacian_sharpen, '細部が浮き上がって見える'),
                ('04_clahe.jpg',        apply_clahe, '全体のコントラストが改善された'),
                ('05_deconv.jpg',       apply_deconvolution, 'やや人工的だが明瞭さが増した')
            ]

            for fname, func, comment in process_list:
                path = process_and_save(img_rgb, func, fname)
                lap = calculate_laplacian_energy(path)
                psnr = calculate_psnr(upload_path, path)
                ssim = calculate_ssim(upload_path, path)
                key = fname.split('_')[0]  # '01'〜'05'
                evaluations[key] = {
                    'lap': lap,
                    'psnr': psnr,
                    'ssim': ssim,
                    'desc': comment
                }

            return render_template('index.html', filename=jpg_filename, evaluations=evaluations)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
