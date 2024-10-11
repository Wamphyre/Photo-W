import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import ctypes
import threading
import time
import sys
import requests
import subprocess
import tempfile
import re
from typing import Optional, Tuple

VERSION = "1.2"

class UpdateSystem:
    def __init__(self, current_version: str):
        self.current_version = current_version
        self.repo_url = "Wamphyre/Photo-W"
        self.api_url = f"https://api.github.com/repos/{self.repo_url}/releases/latest"

    def check_for_updates(self) -> Tuple[Optional[str], Optional[str]]:
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()
            latest_release = response.json()
            latest_version = latest_release['tag_name']
            
            if self._is_newer_version(latest_version):
                return latest_version, self._get_windows_exe_url(latest_release)
            return None, None
        except requests.RequestException:
            return None, None

    def _is_newer_version(self, latest_version: str) -> bool:
        current = [int(x) for x in self.current_version.split('.')]
        latest = [int(x) for x in latest_version.split('.')]
        return latest > current

    def _get_windows_exe_url(self, release: dict) -> Optional[str]:
        for asset in release['assets']:
            if asset['name'].endswith('.exe'):
                return asset['browser_download_url']
        return None

    def download_update(self, download_url: str) -> Optional[str]:
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            filename = self._get_filename_from_response(response, download_url)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                update_file = os.path.join(temp_dir, filename)
                
                with open(update_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                return update_file
        except requests.RequestException as e:
            print(f"Error al descargar la actualización: {e}")
            return None

    def _get_filename_from_response(self, response: requests.Response, url: str) -> str:
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition:
            filename = re.findall("filename=(.+)", content_disposition)
            if filename:
                return filename[0].strip('"')
        return os.path.basename(url)

    def install_update(self, update_file: str) -> None:
        if sys.platform.startswith('win'):
            subprocess.Popen([update_file], shell=True)
            sys.exit()

class ImageProcessor:
    def __init__(self):
        self.use_gpu = self._check_gpu_support()

    def _check_gpu_support(self) -> bool:
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False

    def process_image(self, img: np.ndarray, brightness: float, contrast: float, 
                      saturation: float, angle: int, is_grayscale: bool) -> np.ndarray:
        if self.use_gpu:
            return self._process_image_gpu(img, brightness, contrast, saturation, angle, is_grayscale)
        else:
            return self._process_image_cpu(img, brightness, contrast, saturation, angle, is_grayscale)

    def _process_image_gpu(self, img: np.ndarray, brightness: float, contrast: float, 
                           saturation: float, angle: int, is_grayscale: bool) -> np.ndarray:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        gpu_img = cv2.cuda.multiply(gpu_img, contrast)
        gpu_img = cv2.cuda.add(gpu_img, (brightness-1)*100)

        gpu_hsv = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.cuda.split(gpu_hsv)
        s = cv2.cuda.multiply(s, saturation)
        gpu_hsv = cv2.cuda.merge([h, s, v])
        gpu_img = cv2.cuda.cvtColor(gpu_hsv, cv2.COLOR_HSV2RGB)

        if angle != 0:
            h, w = gpu_img.size()
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            gpu_img = cv2.cuda.warpAffine(gpu_img, M, (w, h))

        if is_grayscale:
            gpu_img = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_RGB2GRAY)
            gpu_img = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_GRAY2RGB)

        return gpu_img.download()

    def _process_image_cpu(self, img: np.ndarray, brightness: float, contrast: float, 
                           saturation: float, angle: int, is_grayscale: bool) -> np.ndarray:
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=(brightness-1)*100)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, saturation)
        hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if angle != 0:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

        if is_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

class PhotoW:
    def __init__(self, master: ttk.Window):
        self.master = master
        self.master.title(f"Photo-W v{VERSION}")
        self.master.geometry("1000x600")
        
        if self._check_for_update():
            return

        self._set_icon()
        self.style = ttk.Style("darkly")
        
        self.cv_image: Optional[np.ndarray] = None
        self.original_cv_image: Optional[np.ndarray] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.zoom = 1.0
        self.angle = 0
        self.brightness = 1.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.is_grayscale = False

        self.image_processor = ImageProcessor()

        self._create_widgets()
        self._setup_threading()

    def _check_for_update(self) -> bool:
        updater = UpdateSystem(VERSION)
        new_version, download_url = updater.check_for_updates()
        
        if new_version and download_url:
            if messagebox.askyesno("Actualización disponible", 
                                   f"Hay una nueva versión disponible: {new_version}. ¿Desea actualizar?"):
                update_file = updater.download_update(download_url)
                if update_file:
                    updater.install_update(update_file)
                else:
                    messagebox.showerror("Error", "No se pudo descargar la actualización.")
            return True
        return False

    def _set_icon(self) -> None:
        if getattr(sys, 'frozen', False):
            application_path = sys._MEIPASS
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(application_path, 'icon.ico')
        self.master.iconbitmap(icon_path)

    def _create_widgets(self) -> None:
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=BOTH, expand=YES)

        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(side=LEFT, fill=BOTH, expand=YES)

        self.canvas = tk.Canvas(self.image_frame, bg='#2f3640')
        self.canvas.pack(fill=BOTH, expand=YES)

        control_frame = ttk.Frame(self.main_frame, padding=10, width=200)
        control_frame.pack(side=RIGHT, fill=Y)
        control_frame.pack_propagate(False)

        ttk.Button(control_frame, text="Abrir Imagen", command=self.open_image).pack(fill=X, pady=5)
        ttk.Button(control_frame, text="Guardar Imagen", command=self.save_image).pack(fill=X, pady=5)

        ttk.Separator(control_frame, orient=HORIZONTAL).pack(fill=X, pady=10)

        self._create_slider(control_frame, "Brillo", 0, 2, 1, 0.01, self.update_brightness)
        self._create_slider(control_frame, "Contraste", 0, 2, 1, 0.01, self.update_contrast)
        self._create_slider(control_frame, "Saturación", 0, 2, 1, 0.01, self.update_saturation)

        ttk.Separator(control_frame, orient=HORIZONTAL).pack(fill=X, pady=10)

        ttk.Button(control_frame, text="Rotar 90°", command=self.rotate_image).pack(fill=X, pady=5)
        ttk.Button(control_frame, text="Zoom In", command=self.zoom_in).pack(fill=X, pady=5)
        ttk.Button(control_frame, text="Zoom Out", command=self.zoom_out).pack(fill=X, pady=5)
        
        self.grayscale_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Escala de Grises", variable=self.grayscale_var, 
                        command=self.update_grayscale).pack(fill=X, pady=5)

        self.master.bind("<Configure>", self.on_window_resize)

    def _create_slider(self, parent: ttk.Frame, label: str, from_: float, to: float, 
                       initial: float, resolution: float, command: callable) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=X, pady=5)
        ttk.Label(frame, text=label).pack(side=TOP, fill=X)
        slider = ttk.Scale(frame, from_=from_, to=to, value=initial, command=command, orient=HORIZONTAL)
        slider.pack(side=BOTTOM, fill=X)

    def _setup_threading(self) -> None:
        self.update_thread = None
        self.update_lock = threading.Lock()
        self.update_event = threading.Event()
        self.processing = False
        self.last_update_time = 0
        self.update_interval = 0.05
        self.full_quality_thread = None
        self.full_quality_event = threading.Event()

    def open_image(self, file_path: Optional[str] = None) -> None:
        if not file_path:
            file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.cv_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                self.original_cv_image = self.cv_image.copy()
                self._reset_image_parameters()
                self.update_image(force=True)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo abrir la imagen: {str(e)}")

    def _reset_image_parameters(self) -> None:
        self.zoom = 1
        self.angle = 0
        self.brightness = 1
        self.contrast = 1
        self.saturation = 1
        self.is_grayscale = False

    def update_image(self, force: bool = False, preview: bool = False) -> None:
        if self.cv_image is None:
            return

        current_time = time.time()
        if force or (current_time - self.last_update_time) >= self.update_interval:
            self.last_update_time = current_time
            if self.update_thread is None or not self.update_thread.is_alive():
                self.update_thread = threading.Thread(target=self._process_and_display, args=(preview,))
                self.update_thread.start()
            else:
                self.update_event.set()

        if self.full_quality_thread is None or not self.full_quality_thread.is_alive():
            self.full_quality_thread = threading.Thread(target=self._process_full_quality)
            self.full_quality_thread.start()

    def _process_and_display(self, preview: bool) -> None:
        while True:
            with self.update_lock:
                if self.processing:
                    continue
                self.processing = True

            processed = self.image_processor.process_image(
                self.original_cv_image, self.brightness, self.contrast, 
                self.saturation, self.angle, self.is_grayscale
            )
            if processed is not None:
                self._resize_and_display(processed)

            with self.update_lock:
                self.processing = False

            if not preview:
                break

            if not self.update_event.wait(timeout=0.01):
                break
            self.update_event.clear()

    def _process_full_quality(self) -> None:
        time.sleep(0.5)
        processed = self.image_processor.process_image(
            self.original_cv_image, self.brightness, self.contrast, 
            self.saturation, self.angle, self.is_grayscale
        )
        if processed is not None:
            self.master.after_idle(lambda: self._resize_and_display(processed))

    def _resize_and_display(self, img: np.ndarray) -> None:
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_height, img_width = img.shape[:2]
        
        scale = min(canvas_width/img_width, canvas_height/img_height) * self.zoom
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
        
        self.master.after_idle(self._update_canvas)

    def _update_canvas(self) -> None:
        if self.photo:
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, anchor=CENTER, image=self.photo)

    def update_brightness(self, value: str) -> None:
        self.brightness = float(value)
        self._update_image_and_notify()

    def update_contrast(self, value: str) -> None:
        self.contrast = float(value)
        self._update_image_and_notify()

    def update_saturation(self, value: str) -> None:
        self.saturation = float(value)
        self._update_image_and_notify()

    def update_grayscale(self) -> None:
        self.is_grayscale = self.grayscale_var.get()
        self._update_image_and_notify()

    def _update_image_and_notify(self) -> None:
        self.update_image(preview=True)
        self.full_quality_event.set()

    def rotate_image(self) -> None:
        self.angle += 90
        self.angle %= 360
        self.update_image(force=True)

    def zoom_in(self) -> None:
        self.zoom *= 1.2
        self.update_image(force=True)

    def zoom_out(self) -> None:
        self.zoom /= 1.2
        self.update_image(force=True)

    def save_image(self) -> None:
        if self.cv_image is not None:
            file_types = [
                ('PNG', '*.png'),
                ('JPEG', '*.jpg'),
                ('BMP', '*.bmp'),
                ('TIFF', '*.tiff'),
            ]
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=file_types)
            if file_path:
                try:
                    processed = self.image_processor.process_image(
                        self.original_cv_image, self.brightness, self.contrast, 
                        self.saturation, self.angle, self.is_grayscale
                    )
                    cv2.imwrite(file_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
                    messagebox.showinfo("Éxito", "Imagen guardada correctamente.")
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo guardar la imagen: {str(e)}")
        else:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar.")

    def on_window_resize(self, event: tk.Event) -> None:
        if self.cv_image is not None:
            self.update_image(force=True)

def main():
    # Evitar que se abra una consola en Windows
    if sys.platform.startswith('win'):
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

    root = ttk.Window("Photo-W")
    app = PhotoW(root)
    
    # Verificar si se pasó una imagen como argumento
    if len(sys.argv) > 1:
        app.open_image(sys.argv[1])
    
    root.mainloop()

if __name__ == "__main__":
    main()