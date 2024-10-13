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
import logging
import hashlib

VERSION = "1.5.0"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        current_version_parts = current_version.split('.')[:3]
        declared_version_parts = VERSION.split('.')
        if current_version_parts != declared_version_parts:
            logger.warning(f"La versión del ejecutable ({'.'.join(current_version_parts)}) no coincide con la versión declarada ({VERSION})")
    except Exception as e:
        logger.error(f"Error al verificar la versión: {e}")

class UpdateSystem:
    def __init__(self, current_version: str):
        self.current_version = current_version
        self.repo_url = "Wamphyre/Photo-W"
        self.api_url = f"https://api.github.com/repos/{self.repo_url}/releases/latest"

    def check_for_updates(self) -> Tuple[Optional[str], Optional[str]]:
        try:
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            latest_release = response.json()
            latest_version = latest_release['tag_name']
            
            if self._is_newer_version(latest_version):
                return latest_version, self._get_windows_exe_url(latest_release)
            return None, None
        except requests.RequestException as e:
            logger.error(f"Error al verificar actualizaciones: {e}")
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
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            
            filename = self._get_filename_from_response(response, download_url)
            
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            update_file = os.path.join(downloads_dir, filename)
            
            with open(update_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return update_file
        except requests.RequestException as e:
            logger.error(f"Error al descargar la actualización: {e}")
            return None

    def _get_filename_from_response(self, response: requests.Response, url: str) -> str:
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition:
            filename = re.findall("filename=(.+)", content_disposition)
            if filename:
                return filename[0].strip('"')
        return os.path.basename(url)

class ImageProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.image_cache = {}

    @lru_cache(maxsize=10)
    def _process_image_cached(self, img_hash, brightness, contrast, saturation, angle, is_grayscale):
        img = self.image_cache.get(img_hash)
        if img is None:
            return None
        
        processed = img.copy()

        if angle != 0:
            processed = self._rotate_image(processed, angle)

        processed = self._adjust_image(processed, brightness, contrast, saturation, is_grayscale)

        return processed

    def process_image(self, img: np.ndarray, brightness: float, contrast: float, 
                      saturation: float, angle: int, is_grayscale: bool) -> np.ndarray:
        img_hash = hash(img.tobytes())
        self.image_cache[img_hash] = img
        return self._process_image_cached(img_hash, brightness, contrast, saturation, angle, is_grayscale)

    def _adjust_image(self, img: np.ndarray, brightness: float, contrast: float, 
                      saturation: float, is_grayscale: bool) -> np.ndarray:
        # Convertir a float32 para cálculos precisos
        img = img.astype(np.float32) / 255.0

        # Ajustar brillo y contraste
        img = img * contrast + (brightness - 1)

        # Ajustar saturación
        if saturation != 1:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 1)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Convertir a escala de grises si es necesario
        if is_grayscale:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

        # Asegurar que los valores estén en el rango [0, 1]
        img = np.clip(img, 0, 1)

        # Convertir de vuelta a uint8
        return (img * 255).astype(np.uint8)

    def _rotate_image(self, img: np.ndarray, angle: int) -> np.ndarray:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    def crop_image(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        return img[y1:y2, x1:x2]

class HistoryManager:
    def __init__(self):
        self.states = []
        self.current_index = -1

    def add_state(self, state):
        self.states = self.states[:self.current_index + 1]
        self.states.append(state)
        self.current_index = len(self.states) - 1

    def undo(self):
        if self.current_index > 0:
            self.current_index -= 1
            return self.states[self.current_index]
        return None

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
        
        if check_for_update(VERSION, self.master):
            self.master.quit()
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
                if self.cv_image is None:
                    raise ValueError("No se pudo cargar la imagen")
                
                if len(self.cv_image.shape) == 2:  # Imagen en escala de grises
                    self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2RGB)
                elif self.cv_image.shape[2] == 3:  # Imagen BGR
                    self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                elif self.cv_image.shape[2] == 4:  # Imagen BGRA
                    self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGRA2RGBA)
                
                self.original_cv_image = self.cv_image.copy()
                logger.debug(f"Imagen cargada: shape={self.cv_image.shape}, dtype={self.cv_image.dtype}")
                self._reset_image_parameters()
                self._reset_sliders()
                self.update_image()
                self.history_manager.add_state(self.get_current_state())
            except Exception as e:
                logger.error(f"No se pudo abrir la imagen: {str(e)}")
                messagebox.showerror("Error", f"No se pudo abrir la imagen: {str(e)}")

    def _reset_image_parameters(self) -> None:
        self.zoom = 1.0
        self.angle = 0
        self.brightness = 1.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.is_grayscale = False

    def _reset_sliders(self) -> None:
        for slider in self.sliders:
            slider.set(1)
        self.grayscale_var.set(False)

    def update_image(self) -> None:
        if self.cv_image is None:
            return

        processed = self.image_processor.process_image(
            self.cv_image, self.brightness, self.contrast, 
            self.saturation, self.angle, self.is_grayscale
        )
        logger.debug(f"Imagen procesada: shape={processed.shape}, dtype={processed.dtype}")
        
        self._resize_and_display(processed)

    def _resize_and_display(self, img: np.ndarray) -> None:
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_height, img_width = img.shape[:2]
        
        # Calcular la escala para ajustar la imagen al canvas manteniendo la relación de aspecto
        scale = min(canvas_width/img_width, canvas_height/img_height) * self.zoom
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        if new_width > 0 and new_height > 0:
            # Redimensionar la imagen
            resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convertir la imagen a formato PIL
            pil_img = Image.fromarray(resized)
            
            self.photo = ImageTk.PhotoImage(image=pil_img)
            self.canvas.delete("all")
            
            # Centrar la imagen en el canvas
            x = max(0, (canvas_width - new_width) // 2)
            y = max(0, (canvas_height - new_height) // 2)
            
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            logger.debug(f"Imagen mostrada en el canvas: shape={resized.shape}, dtype={resized.dtype}")
        else:
            logger.warning(f"Dimensiones de imagen inválidas: {new_width}x{new_height}")

    def update_brightness(self, value: str) -> None:
        self.brightness = float(value)
        self.update_image()
        self.history_manager.add_state(self.get_current_state())

    def update_contrast(self, value: str) -> None:
        self.contrast = float(value)
        self.update_image()
        self.history_manager.add_state(self.get_current_state())

    def update_saturation(self, value: str) -> None:
        self.saturation = float(value)
        self.update_image()
        self.history_manager.add_state(self.get_current_state())

    def update_grayscale(self) -> None:
        self.is_grayscale = self.grayscale_var.get()
        self.update_image()
        self.history_manager.add_state(self.get_current_state())

    def rotate_image(self) -> None:
        self.angle += 90
        self.angle %= 360
        self.update_image()
        self.history_manager.add_state(self.get_current_state())

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
                    processed = self.image_processor.process_image(
                        self.cv_image, self.brightness, self.contrast, 
                        self.saturation, self.angle, self.is_grayscale
                    )
                    cv2.imwrite(file_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
                    messagebox.showinfo("Éxito", "Imagen guardada correctamente.")
                except Exception as e:
                    logger.error(f"No se pudo guardar la imagen: {str(e)}")
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
            
            scale = min(canvas_width/img_width, canvas_height/img_height) * self.zoom
            
            x1 = max(0, int((start_x - (canvas_width - img_width*scale)/2) / scale))
            y1 = max(0, int((start_y - (canvas_height - img_height*scale)/2) / scale))
            x2 = min(img_width, int((end_x - (canvas_width - img_width*scale)/2) / scale))
            y2 = min(img_height, int((end_y - (canvas_height - img_height*scale)/2) / scale))
            
            self.cv_image = self.image_processor.crop_image(self.cv_image, x1, y1, x2, y2)
            self.update_image()
            self.history_manager.add_state(self.get_current_state())

    def get_current_state(self) -> dict:
        return {
            'image': self.cv_image.copy() if self.cv_image is not None else None,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'saturation': self.saturation,
            'angle': self.angle,
            'is_grayscale': self.is_grayscale,
            'zoom': self.zoom
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
        self.zoom = state['zoom']
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

def check_for_update(version: str, root: tk.Tk) -> bool:
    updater = UpdateSystem(version)
    new_version, download_url = updater.check_for_updates()
    
    if new_version and download_url:
        if messagebox.askyesno("Actualización Disponible", 
                               f"Hay una nueva versión disponible: {new_version}. ¿Desea descargarla?"):
            update_file = updater.download_update(download_url)
            if update_file:
                if messagebox.askyesno("Actualización Descargada", 
                                       f"La actualización ha sido descargada a:\n{update_file}\n\n¿Desea ejecutar el instalador ahora?"):
                    root.quit()
                    os.startfile(update_file)
                    return True
                else:
                    messagebox.showinfo("Información", f"Puede instalar la actualización más tarde ejecutando:\n{update_file}")
            else:
                messagebox.showerror("Error", "No se pudo descargar la actualización.")
        return False
    return False

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