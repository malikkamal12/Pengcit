
from scipy import ndimage
from flask import Flask, redirect, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
def enhancement():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Mendapatkan nilai kontras dan kecerahan dari formulir
        alpha = float(request.form.get('kontras', 2.0))
        beta = float(request.form.get('kecerahan', 0.0))

        # Melakukan image enhancement dengan mengatur kontras dan kecerahan
        enhanced_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Menyimpan gambar hasil enhancement ke folder "static/uploads"
        enhanced_image_path = os.path.join(
            app.config['UPLOAD'], 'enhanced_image.jpg')
        cv2.imwrite(enhanced_image_path, enhanced_image)

        return render_template('enhancement.html', img=img_path, img2=enhanced_image_path)

    return render_template('enhancement.html')


@app.route('/noise', methods=['GET', 'POST'])
def noise():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Pilih metode reduksi noise (Median Smoothing)
        kernel_size = 5  # Ukuran kernel untuk Median Blur
        denoised_img = cv2.medianBlur(img, kernel_size)

        # Menyimpan gambar hasil reduksi noise ke folder "static/uploads"
        denoised_image_path = os.path.join(
            app.config['UPLOAD'], 'denoised_image.jpg')
        cv2.imwrite(denoised_image_path, denoised_img)

        return render_template('noise.html', img=img_path, img2=denoised_image_path)

    return render_template('noise.html')


@app.route('/erosion', methods=['GET', 'POST'])
def erosion():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Read the image with OpenCV
        img = cv2.imread(img_path)

        # Denoise the image
        denoised_img = cv2.fastNlMeansDenoisingColored(
            img, None, 10, 10, 7, 21)

        # Resize the image if necessary
        # resized_img = cv2.resize(denoised_img, (new_width, new_height))

        # Define the kernel erosion
        kernel_size = int(request.form.get('kernel_size', 7))  # Kernel size
        kernel_shape = request.form.get(
            'kernel_shape', 'cv2.MORPH_RECT')  # Kernel shape
        iterations = int(request.form.get('iterations', 1)
                         )  # Number of iterations
        kernel = cv2.getStructuringElement(
            eval(kernel_shape), (kernel_size, kernel_size))
        # Perform the erosion operation on the image
        eroded_img = cv2.erode(denoised_img, kernel, iterations=iterations)
        # Save the eroded image
        eroded_image_path = os.path.join(
            app.config['UPLOAD'], 'eroded_image.jpg')
        cv2.imwrite(eroded_image_path, eroded_img)

        return render_template('erosion.html', img=img_path, img2=eroded_image_path)

    return render_template('erosion.html')


@app.route('/dilation', methods=['GET', 'POST'])
def dilation():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Read the image with OpenCV
        img = cv2.imread(img_path)

        # Denoise the image
        denoised_img = cv2.fastNlMeansDenoisingColored(
            img, None, 10, 10, 7, 21)

        # Resize the image if necessary
        # resized_img = cv2.resize(denoised_img, (new_width, new_height))

        # Define the kernel dilation
        kernel_size = int(request.form.get('kernel_size', 7))  # Kernel size
        kernel_shape = request.form.get(
            'kernel_shape', 'cv2.MORPH_RECT')  # Kernel shape
        iterations = int(request.form.get('iterations', 1)
                         )  # Number of iterations
        kernel = cv2.getStructuringElement(
            eval(kernel_shape), (kernel_size, kernel_size))

        # Perform the dilation operation on the image
        dilated_img = cv2.dilate(denoised_img, kernel, iterations=iterations)

        # Save the dilated image
        dilated_image_path = os.path.join(
            app.config['UPLOAD'], 'dilated_image.jpg')
        cv2.imwrite(dilated_image_path, dilated_img)

        return render_template('dilation.html', img=img_path, img2=dilated_image_path)

    return render_template('dilation.html')


@app.route('/bicubic', methods=['GET', 'POST'])
def bicubic():
    if request.method == 'POST':
        # Mendapatkan file gambar dari form
        file = request.files['img']
        filename = secure_filename(file.filename)
        # Menyimpan file gambar di direktori upload
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Mendapatkan faktor skala dari formulir
        scale_factor = float(request.form.get('scale_factor', 2.0))

        # Menghitung dimensi baru berdasarkan faktor skala
        new_dimensions = (
            int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))

        # Melakukan interpolasi Bicubic pada gambar
        img_bicubic = cv2.resize(
            img, new_dimensions, interpolation=cv2.INTER_CUBIC)

        # Menyimpan gambar hasil interpolasi Bicubic ke folder "static/uploads"
        bicubic_image_path = os.path.join(
            app.config['UPLOAD'], 'bicubic_interpolated_image.jpg')
        cv2.imwrite(bicubic_image_path, img_bicubic)

        # Menampilkan nilai pixel di titik yang dipilih oleh pengguna
        x = int(request.form.get('x_coordinate', 0))
        y = int(request.form.get('y_coordinate', 0))
        pixel_value = img[y, x]

        # Menghitung jumlah pixel di gambar asli dan gambar hasil interpolasi
        original_pixel_count = img.shape[0] * img.shape[1]
        bicubic_pixel_count = img_bicubic.shape[0] * img_bicubic.shape[1]

        # Mengembalikan template dengan gambar asli, hasil interpolasi, dan nilai pixel
        return render_template('bicubic.html', img=img_path, img2=bicubic_image_path, pixel_value=pixel_value, x_coordinate=x, y_coordinate=y, original_pixel_count=original_pixel_count, bicubic_pixel_count=bicubic_pixel_count)

    # Menampilkan form jika metode request adalah GET
    return render_template('bicubic.html')


@app.route('/nearest_neighbor', methods=['GET', 'POST'])
def nearest_neighbor():
    if request.method == 'POST':
        # Mendapatkan file gambar dari form
        file = request.files['img']
        filename = secure_filename(file.filename)
        # Menyimpan file gambar di direktori upload
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Mendapatkan faktor skala dari formulir
        scale_factor = float(request.form.get('scale_factor', 2.0))

        # Menghitung dimensi baru berdasarkan faktor skala
        new_dimensions = (
            int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))

        # Melakukan interpolasi Nearest Neighbor pada gambar
        img_nearest = cv2.resize(
            img, new_dimensions, interpolation=cv2.INTER_NEAREST)

        # Menyimpan gambar hasil interpolasi Nearest Neighbor ke folder "static/uploads"
        nearest_image_path = os.path.join(
            app.config['UPLOAD'], 'nearest_neighbor_interpolated_image.jpg')
        cv2.imwrite(nearest_image_path, img_nearest)

        # Menghitung jumlah pixel di gambar asli dan gambar hasil interpolasi Nearest Neighbor
        original_pixel_count = img.shape[0] * img.shape[1]
        nearest_pixel_count = img_nearest.shape[0] * img_nearest.shape[1]

        # Mengembalikan template dengan gambar asli, hasil interpolasi, dan jumlah pixel
        return render_template('nearest_neighbor.html', img=img_path, img2=nearest_image_path, scale_factor=scale_factor, original_pixel_count=original_pixel_count, nearest_pixel_count=nearest_pixel_count)

    # Menampilkan form jika metode request adalah GET
    return render_template('nearest_neighbor.html')


if __name__ == '__main__':
    app.run(debug=True, port=8001)
