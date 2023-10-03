
from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder


@app.route('/')
def home():
    return render_template('histogram_equalization.html')


@app.route('/histogram', methods=['GET', 'POST'])
def histogram_equ():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Menghitung histogram untuk masing-masing saluran (R, G, B)
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])

        # Normalisasi histogram
        hist_r /= hist_r.sum()
        hist_g /= hist_g.sum()
        hist_b /= hist_b.sum()

        # Simpan histogram sebagai gambar PNG
        hist_image_path = os.path.join(app.config['UPLOAD'], 'histogram.png')
        plt.figure()
        plt.title("RGB Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_r, color='red', label='Red')
        plt.plot(hist_g, color='green', label='Green')
        plt.plot(hist_b, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_image_path)

        # Hasil equalisasi
        # Ubah ke ruang warna YCrCb
        img_equalized = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_equalized[:, :, 0] = cv2.equalizeHist(
            img_equalized[:, :, 0])  # Equalisasi komponen Y (luminance)
        # Kembalikan ke ruang warna BGR
        img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_YCrCb2BGR)

        # Menyimpan gambar hasil equalisasi ke folder "static/uploads"
        equalized_image_path = os.path.join(
            'static', 'uploads', 'img-equalized.jpg')
        cv2.imwrite(equalized_image_path, img_equalized)

        # Menghitung histogram untuk gambar yang sudah diequalisasi
        hist_equalized_r = cv2.calcHist(
            [img_equalized], [0], None, [256], [0, 256])
        hist_equalized_g = cv2.calcHist(
            [img_equalized], [1], None, [256], [0, 256])
        hist_equalized_b = cv2.calcHist(
            [img_equalized], [2], None, [256], [0, 256])

        # Normalisasi histogram
        hist_equalized_r /= hist_equalized_r.sum()
        hist_equalized_g /= hist_equalized_g.sum()
        hist_equalized_b /= hist_equalized_b.sum()

        # Simpan histogram hasil equalisasi sebagai gambar PNG
        hist_equalized_image_path = os.path.join(
            app.config['UPLOAD'], 'histogram_equalized.png')
        plt.figure()
        plt.title("RGB Histogram (Equalized)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_equalized_r, color='red', label='Red')
        plt.plot(hist_equalized_g, color='green', label='Green')
        plt.plot(hist_equalized_b, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_equalized_image_path)

        return render_template('histogram_equalization.html', img=img_path, img2=equalized_image_path, histogram=hist_image_path, histogram2=hist_equalized_image_path)

    return render_template('histogram_equalization.html')


def edge_detection(img):
    # Menerapkan deteksi tepi menggunakan algoritma Canny
    edges = cv2.Canny(img, 100, 200)

    # Menyimpan gambar hasil deteksi tepi ke folder "static/uploads"
    edge_image_path = os.path.join(app.config['UPLOAD'], 'edge_detected.jpg')
    cv2.imwrite(edge_image_path, edges)

    return edge_image_path


@app.route('/edge', methods=['GET', 'POST'])
def edge():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Memanggil fungsi edge_detection
        edge_image_path = edge_detection(img)

        return render_template('edge.html', img=img_path, edge=edge_image_path)

    return render_template('edge.html')


def blur_faces(image_path, blur_level):
    # Membaca gambar dengan OpenCV
    img = cv2.imread(image_path)

    # Menggunakan Cascade Classifier untuk mendeteksi wajah
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Menerapkan deteksi wajah dengan parameter yang diatur
    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=[30, 30])

    # Menerapkan efek blur ke setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Ambil bagian wajah dari gambar
        face = img[y:y+h, x:x+w]
        # Hitung ukuran kernel berdasarkan tingkat blur yang diatur
        kernel_size = (blur_level, blur_level)
        # Terapkan efek blur Gaussian dengan kernel yang sesuai
        blurred_face = cv2.GaussianBlur(face, kernel_size, 0)
        img[y:y+h, x:x+w] = blurred_face

    # Menyimpan gambar dengan wajah-wajah yang telah di-blur
    blurred_image_path = os.path.join(
        app.config['UPLOAD'], 'blurred_image.jpg')
    cv2.imwrite(blurred_image_path, img)

    return blurred_image_path


@app.route('/faceBlur', methods=['GET', 'POST'])
def face_blur():
    error = None
    if request.method == 'POST':
        # Check if the 'img' file is in the request
        if 'img' not in request.files:
            error = 'Please Select a Picture'
            return render_template('blur.html', error=error)

        file = request.files['img']

        # Check if the file name is empty
        if file.filename == '':
            error = 'Please Select a Picture'
            return render_template('blur.html', error=error)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        image_path = os.path.join(app.config['UPLOAD'], filename)

        # Get blur level from the form
        blur_level = int(request.form.get('tingkatan', 1))

        # Call the function to blur faces
        blurred_image_path = blur_faces(image_path, blur_level)

        return render_template('blur.html', img=image_path, img2=blurred_image_path)

    return render_template('blur.html')


@app.route('/segmentation', methods=['GET', 'POST'])
def segment():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Melakukan segmentasi citra
        # Dapat diganti dengan algoritma segmentasi yang sesuai dengan kebutuhan
        # Sebagai contoh, di sini digunakan metode ambang batas sederhana.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, segmented_image = cv2.threshold(
            gray_img, 128, 255, cv2.THRESH_BINARY)

        # Menyimpan gambar hasil segmentasi ke folder "static/uploads"
        segmented_image_path = os.path.join(
            app.config['UPLOAD'], 'segmented_image.jpg')
        cv2.imwrite(segmented_image_path, segmented_image)

        return render_template('segment.html', img=img_path, img2=segmented_image_path)

    return render_template('segment.html')


@app.route('/enhancement', methods=['GET', 'POST'])
def image_enhancement():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Mendapatkan nilai kontras dan kecerahan dari formulir
        alpha = float(request.form.get('kontras', 1.0))
        beta = float(request.form.get('kecerahan', 0.0))

        # Melakukan image enhancement dengan mengatur kontras dan kecerahan
        enhanced_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Menyimpan gambar hasil enhancement ke folder "static/uploads"
        enhanced_image_path = os.path.join(
            app.config['UPLOAD'], 'enhanced_image.jpg')
        cv2.imwrite(enhanced_image_path, enhanced_image)

        return render_template('enhancement.html', img=img_path, img2=enhanced_image_path)

    return render_template('enhancement.html')


if __name__ == '__main__':
    app.run(debug=True, port=8001)
