import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import ctypes
import sys
import requests
import subprocess
import tempfile
import re
from typing import Optional, Tuple
import win32api
from functools import lru_cache
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

VERSION = "1.3.0"

def get_windows_file_version(filename):
    info = win32api.GetFileVersionInfo(filename, "\\")
    ms = info['FileVersionMS']
    ls = info['FileVersionLS']
    return f"{HIWORD(ms)}.{LOWORD(ms)}.{HIWORD(ls)}.{LOWORD(ls)}"

def HIWORD(x):
    return (x >> 16) & 0xffff

def LOWORD(x):
    return x & 0xffff

def check_current_version():
    try:
        current_version = get_windows_file_version(sys.executable)
        if current_version != VERSION:
            print(f"Advertencia: La versión del ejecutable ({current_version}) no coincide con la versión declarada ({VERSION})")
    except Exception as e:
        print(f"Error al verificar la versión: {e}")

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
            try:
                ctypes.windll.shell32.ShellExecuteW(None, "runas", update_file, None, None, 1)
                self.exit_application()
            except Exception as e:
                print(f"Error al iniciar la actualización: {e}")
                messagebox.showerror("Error", "No se pudo iniciar la actualización. Por favor, inténtelo manualmente.")

    def exit_application(self):
        sys.exit()

class ImageProcessor:
    def __init__(self):
        self.use_gpu = self._check_gpu_support()
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

    def _check_gpu_support(self) -> bool:
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False

    @lru_cache(maxsize=32)
    def process_image(self, img_bytes: bytes, brightness: float, contrast: float, 
                      saturation: float, angle: int, is_grayscale: bool) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if self.use_gpu:
            return self._process_image_gpu(img, brightness, contrast, saturation, angle, is_grayscale)
        else:
            return self._process_image_cpu(img, brightness, contrast, saturation, angle, is_grayscale)

    def _process_image_gpu(self, img: np.ndarray, brightness: float, contrast: float, 
                           saturation: float, angle: int, is_grayscale: bool) -> np.ndarray:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        # Aplicar brillo y contraste
        gpu_img = cv2.cuda.multiply(gpu_img, contrast)
        gpu_img = cv2.cuda.add(gpu_img, brightness - 1)

        # Aplicar saturación
        if saturation != 1:
            gpu_hsv = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.cuda.split(gpu_hsv)
            s = cv2.cuda.multiply(s, saturation)
            gpu_hsv = cv2.cuda.merge((h, s, v))
            gpu_img = cv2.cuda.cvtColor(gpu_hsv, cv2.COLOR_HSV2BGR)

        # Aplicar rotación
        if angle != 0:
            h, w = gpu_img.size()
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            gpu_img = cv2.cuda.warpAffine(gpu_img, M, (w, h))

        # Aplicar escala de grises
        if is_grayscale:
            gpu_img = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
            gpu_img = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_GRAY2BGR)

        return gpu_img.download()

    def _process_image_cpu(self, img: np.ndarray, brightness: float, contrast: float, 
                           saturation: float, angle: int, is_grayscale: bool) -> np.ndarray:
        def process_chunk(chunk):
            chunk = chunk.astype(np.float32) / 255.0
            chunk = cv2.multiply(chunk, contrast)
            chunk = cv2.add(chunk, brightness - 1)
            
            if saturation != 1:
                hsv = cv2.cvtColor(chunk, cv2.COLOR_BGR2HSV)
                hsv[:,:,1] *= saturation
                chunk = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return (np.clip(chunk, 0, 1) * 255).astype(np.uint8)

        # Dividir la imagen en chunks para procesamiento paralelo
        chunks = np.array_split(img, multiprocessing.cpu_count())
        processed_chunks = list(self.executor.map(process_chunk, chunks))
        img = np.vstack(processed_chunks)

        if angle != 0:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

        if is_grayscale:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        return img

    def crop_image(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        return img[y1:y2, x1:x2]

class HistoryManager:
    def __init__(self):
        self.previous_state = None

    def add_state(self, state):
        self.previous_state = state

    def undo(self):
        return self.previous_state

class CropTool:
    def __init__(self, canvas, on_crop):
        self.canvas = canvas
        self.on_crop = on_crop
        self.start_x = self.start_y = 0
        self.end_x = self.end_y = 0
        self.rect_id = None

    def start(self):
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

    def on_drag(self, event):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, 
                                                    self.end_x, self.end_y, 
                                                    outline='red')

    def on_release(self, event):
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)
        self.on_crop(self.start_x, self.start_y, self.end_x, self.end_y)
        self.canvas.delete(self.rect_id)

class PhotoW:
    def __init__(self, master: ttk.Window):
        self.master = master
        self.master.title(f"Photo-W v{VERSION}")
        self.master.geometry("1000x600")
        
        check_current_version()
        
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
        self.history_manager = HistoryManager()

        self._create_widgets()
        self.crop_tool = CropTool(self.canvas, self.apply_crop)

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
        ttk.Button(control_frame, text="Recortar", command=self.start_crop).pack(fill=X, pady=5)
        ttk.Button(control_frame, text="Deshacer", command=self.undo).pack(fill=X, pady=5)

        ttk.Separator(control_frame, orient=HORIZONTAL).pack(fill=X, pady=10)

        self.sliders = []
        self.sliders.append(self._create_slider(control_frame, "Brillo", 0, 2, 1, 0.01, self.update_brightness))
        self.sliders.append(self._create_slider(control_frame, "Contraste", 0, 2, 1, 0.01, self.update_contrast))
        self.sliders.append(self._create_slider(control_frame, "Saturación", 0, 2, 1, 0.01, self.update_saturation))

        ttk.Separator(control_frame, orient=HORIZONTAL).pack(fill=X, pady=10)

        ttk.Button(control_frame, text="Rotar 90°", command=self.rotate_image).pack(fill=X, pady=5)
        ttk.Button(control_frame, text="Zoom In", command=self.zoom_in).pack(fill=X, pady=5)
        ttk.Button(control_frame, text="Zoom Out", command=self.zoom_out).pack(fill=X, pady=5)
        
        self.grayscale_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Escala de Grises", variable=self.grayscale_var, 
                        command=self.update_grayscale).pack(fill=X, pady=5)

        self.master.bind("<Configure>", self.on_window_resize)

    def _create_slider(self, parent: ttk.Frame, label: str, from_: float, to: float, 
                       initial: float, resolution: float, command: callable) -> ttk.Scale:
        frame = ttk.Frame(parent)
        frame.pack(fill=X, pady=5)
        ttk.Label(frame, text=label).pack(side=TOP, fill=X)
        slider = ttk.Scale(frame, from_=from_, to=to, value=initial, command=command, orient=HORIZONTAL)
        slider.pack(side=BOTTOM, fill=X)
        return slider

    def open_image(self, file_path: Optional[str] = None) -> None:
        if not file_path:
            file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.cv_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                self.original_cv_image = self.cv_image.copy()
                self._reset_image_parameters()
                self._reset_sliders()
                self.update_image()
                self.history_manager.add_state(self.get_current_state())
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo abrir la imagen: {str(e)}")

    def _reset_image_parameters(self) -> None:
        self.zoom = 1
        self.angle = 0
        self.brightness = 1
        self.contrast = 1
        self.saturation = 1
        self.is_grayscale = False

    def _reset_sliders(self) -> None:
        for slider in self.sliders:
            slider.set(1)  # Asumiendo que 1 es el valor predeterminado
        self.grayscale_var.set(False)

    def update_image(self) -> None:
        if self.cv_image is None:
            return

        img_bytes = cv2.imencode('.png', self.cv_image)[1].tobytes()
        processed = self.image_processor.process_image(
            img_bytes, self.brightness, self.contrast, 
            self.saturation, self.angle, self.is_grayscale
        )
        self._resize_and_display(processed)

    def _resize_and_display(self, img: np.ndarray) -> None:
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_height, img_width = img.shape[:2]
        
        scale = min(canvas_width/img_width, canvas_height/img_height) * self.zoom
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        if new_width > 0 and new_height > 0:
            pil_img = Image.fromarray(img)
            resized = pil_img.resize((new_width, new_height), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(image=resized)
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width/2, canvas_height/2, anchor=CENTER, image=self.photo)

    def update_brightness(self, value: str) -> None:
        self.history_manager.add_state(self.get_current_state())
        self.brightness = float(value)
        self.update_image()

    def update_contrast(self, value: str) -> None:
        self.history_manager.add_state(self.get_current_state())
        self.contrast = float(value)
        self.update_image()

    def update_saturation(self, value: str) -> None:
        self.history_manager.add_state(self.get_current_state())
        self.saturation = float(value)
        self.update_image()

    def update_grayscale(self) -> None:
        self.history_manager.add_state(self.get_current_state())
        self.is_grayscale = self.grayscale_var.get()
        self.update_image()

    def rotate_image(self) -> None:
        self.history_manager.add_state(self.get_current_state())
        self.angle += 90
        self.angle %= 360
        self.update_image()

    def zoom_in(self) -> None:
        self.zoom *= 1.2
        self.update_image()

    def zoom_out(self) -> None:
        self.zoom /= 1.2
        self.update_image()

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
                    img_bytes = cv2.imencode('.png', self.cv_image)[1].tobytes()
                    processed = self.image_processor.process_image(
                        img_bytes, self.brightness, self.contrast, 
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
            self.update_image()

    def start_crop(self) -> None:
        self.crop_tool.start()

    def apply_crop(self, start_x: float, start_y: float, end_x: float, end_y: float) -> None:
        if self.cv_image is not None:
            img_height, img_width = self.cv_image.shape[:2]
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            scale = min(canvas_width/img_width, canvas_height/img_height)
            
            x1 = max(0, int((start_x - (canvas_width - img_width*scale)/2) / scale))
            y1 = max(0, int((start_y - (canvas_height - img_height*scale)/2) / scale))
            x2 = min(img_width, int((end_x - (canvas_width - img_width*scale)/2) / scale))
            y2 = min(img_height, int((end_y - (canvas_height - img_height*scale)/2) / scale))
            
            self.history_manager.add_state(self.get_current_state())
            self.cv_image = self.image_processor.crop_image(self.cv_image, x1, y1, x2, y2)
            self.update_image()

    def get_current_state(self) -> dict:
        return {
            'image': self.cv_image.copy() if self.cv_image is not None else None,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'saturation': self.saturation,
            'angle': self.angle,
            'is_grayscale': self.is_grayscale
        }

    def set_state(self, state: dict) -> None:
        if state is None:
            return
        self.cv_image = state['image'].copy() if state['image'] is not None else None
        self.brightness = state['brightness']
        self.contrast = state['contrast']
        self.saturation = state['saturation']
        self.angle = state['angle']
        self.is_grayscale = state['is_grayscale']
        self._update_sliders()
        self.update_image()

    def _update_sliders(self) -> None:
        self.sliders[0].set(self.brightness)
        self.sliders[1].set(self.contrast)
        self.sliders[2].set(self.saturation)
        self.grayscale_var.set(self.is_grayscale)

    def undo(self) -> None:
        state = self.history_manager.undo()
        if state:
            self.set_state(state)

def main():
    if sys.platform.startswith('win'):
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

    root = ttk.Window("Photo-W")
    app = PhotoW(root)
    
    if len(sys.argv) > 1:
        app.open_image(sys.argv[1])
    
    root.mainloop()

if __name__ == "__main__":
    main()