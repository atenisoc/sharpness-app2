from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.signal import convolve2d

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def apply_unsharp_mask(img):
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

def apply_median(img):
    return cv2.medianBlur(img, 5)

def apply_laplacian_filter(img):
    # ガウシアンでノイズを抑制
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # ラプラシアン適用
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    lap_abs = np.abs(lap)

    # 強調（倍率10倍）＋ヒストグラム均等化
    lap_scaled = np.clip(lap_abs * 10, 0, 255).astype(np.uint8)
    lap_eq = cv2.equalizeHist(lap_scaled)

    return lap_eq


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def deconvolution(img, psf, iterations):
    img = img.astype(np.float64)
    estimate = np.full_like(img, 0.5)
    psf_mirror = np.flip(psf)
    for _ in range(iterations):
        conv_est = convolve2d(estimate, psf, 'same')
        conv_est[conv_est == 0] = 1e-5
        relative_blur = img / conv_est
        estimate *= convolve2d(relative_blur, psf_mirror, 'same')
    return np.clip(estimate, 0, 255).astype(np.uint8)

def qualitative_comment(name, lap, psnr, ssim):
    if "laplacian" in name:
        if lap > 800:
            return "エッジが非常に強調されています"
        elif lap > 400:
            return "適度にエッジが抽出されています"
        else:
            return "輪郭が弱く、視認性が低いです"
    elif "deconv" in name:
        if lap > 1000:
            return "やや強調しすぎています"
        elif lap < 200:
            return "鮮鋭化が控えめです"
        else:
            return "適切に鮮鋭化されています"
    elif "unsharp" in name:
        return "輪郭が明瞭になっています"
    elif "median" in name:
        return "ノイズが軽減されています"
    elif "clahe" in name:
        return "局所的にコントラストが強調されています"
    else:
        return "処理済み画像"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='ファイルが選択されていません')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='ファイル名が空です')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            evaluations = {}

            def save_and_evaluate(processed, name):
                out_path = os.path.join(app.config['RESULT_FOLDER'], f'{name}.jpg')
                cv2.imwrite(out_path, processed)
                lap = apply_laplacian(processed)
                psnr = peak_signal_noise_ratio(img, processed)
                ssim = structural_similarity(img, processed)
                desc = qualitative_comment(name, lap, psnr, ssim)
                evaluations[name] = {
                    'lap': round(lap, 2),
                    'psnr': round(psnr, 2),
                    'ssim': round(ssim, 4),
                    'desc': desc
                }

            save_and_evaluate(apply_unsharp_mask(img), '01_unsharp_mask')
            save_and_evaluate(apply_median(img), '02_median')
            save_and_evaluate(apply_laplacian_filter(img), '03_laplacian')  # ← 🔄ここが修正対象
            save_and_evaluate(apply_clahe(img), '04_clahe')

            psf = np.ones((5,5)) / 25
            save_and_evaluate(deconvolution(img, psf, 10), '05_deconv')
            save_and_evaluate(deconvolution(img, psf, 30), '05-2_strong')
            save_and_evaluate(deconvolution(img, psf, 5), '05-3_mild')

            return render_template('index.html', filename=filename, evaluations=evaluations)
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
