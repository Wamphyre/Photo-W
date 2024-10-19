import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageEnhance, ImageOps
import os
import sys
from functools import partial
import threading

class PhotoW:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo-W v1.6")
        self.root.geometry("1024x768")
        
        self.style = ttk.Style(theme="darkly")
        
        if getattr(sys, 'frozen', False):
            application_path = sys._MEIPASS
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
        
        icon_path = os.path.join(application_path, 'icon.ico')
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        
        self.current_image = None
        self.photo = None
        self.original_image = None
        self.initial_image = None  # Para guardar la imagen inicial sin ajustes
        self.zoom_factor = 1.0
        self.rotation = 0
        
        self.crop_start = None
        self.crop_rect = None
        
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=BOTH, expand=YES)
        
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=LEFT, fill=BOTH, expand=YES)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray20", highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=YES)
        
        self.side_panel = ttk.Frame(self.main_frame, padding=10, width=200)
        self.side_panel.pack(side=RIGHT, fill=Y)
        self.side_panel.pack_propagate(False)
        
        self.create_side_panel_controls()
        self.bind_events()
        
        # Verificar si se pasó un archivo como argumento
        if len(sys.argv) > 1:
            self.load_image(sys.argv[1])
        
    def create_side_panel_controls(self):
        ttk.Button(self.side_panel, text="Abrir imagen", command=self.open_image).pack(fill=X, pady=(0, 5))
        ttk.Button(self.side_panel, text="Guardar como...", command=self.save_image).pack(fill=X, pady=(0, 5))
        ttk.Button(self.side_panel, text="Ajustar a ventana", command=self.fit_to_window).pack(fill=X, pady=(0, 5))
        ttk.Button(self.side_panel, text="Tamaño original", command=self.original_size).pack(fill=X, pady=(0, 5))
        ttk.Button(self.side_panel, text="Recortar", command=self.crop_image).pack(fill=X, pady=(0, 15))
        
        ttk.Separator(self.side_panel).pack(fill=X, pady=10)
        
        ttk.Label(self.side_panel, text="Ajustes", font=("TkDefaultFont", 12, "bold")).pack(pady=(0, 10))
        
        ttk.Label(self.side_panel, text="Brillo:").pack(anchor=W)
        self.brightness_scale = ttk.Scale(self.side_panel, from_=0, to=2, value=1, command=self.update_image)
        self.brightness_scale.pack(fill=X, pady=(0, 10))
        
        ttk.Label(self.side_panel, text="Contraste:").pack(anchor=W)
        self.contrast_scale = ttk.Scale(self.side_panel, from_=0, to=2, value=1, command=self.update_image)
        self.contrast_scale.pack(fill=X, pady=(0, 10))
        
        ttk.Label(self.side_panel, text="Saturación:").pack(anchor=W)
        self.saturation_scale = ttk.Scale(self.side_panel, from_=0, to=2, value=1, command=self.update_image)
        self.saturation_scale.pack(fill=X, pady=(0, 10))
        
        ttk.Label(self.side_panel, text="Rotación:").pack(anchor=W)
        rotation_frame = ttk.Frame(self.side_panel)
        rotation_frame.pack(fill=X, pady=(0, 10))
        ttk.Button(rotation_frame, text="⟲ 90°", command=partial(self.rotate_image, -90)).pack(side=LEFT, expand=YES)
        ttk.Button(rotation_frame, text="⟳ 90°", command=partial(self.rotate_image, 90)).pack(side=LEFT, expand=YES)
        
        ttk.Button(self.side_panel, text="Reiniciar ajustes", command=self.reset_adjustments).pack(fill=X, pady=(10, 0))
        
    def bind_events(self):
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Button-4>", self.mouse_wheel)
        self.canvas.bind("<Button-5>", self.mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_image())
        self.root.bind("<Left>", lambda e: self.move_image(-10, 0))
        self.root.bind("<Right>", lambda e: self.move_image(10, 0))
        self.root.bind("<Up>", lambda e: self.move_image(0, -10))
        self.root.bind("<Down>", lambda e: self.move_image(0, 10))
        
    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        try:
            self.original_image = Image.open(file_path)
            self.initial_image = self.original_image.copy()
            self.reset_adjustments()
            self.fit_to_window()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
    
    def fit_to_window(self):
        if self.original_image:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            max_width = int(screen_width * 0.9)
            max_height = int(screen_height * 0.9)
            
            image_ratio = self.original_image.width / self.original_image.height
            if self.original_image.width > max_width or self.original_image.height > max_height:
                if image_ratio > max_width / max_height:
                    new_width = max_width
                    new_height = int(max_width / image_ratio)
                else:
                    new_height = max_height
                    new_width = int(max_height * image_ratio)
            else:
                new_width = self.original_image.width
                new_height = self.original_image.height
            
            window_width = new_width + self.side_panel.winfo_reqwidth()
            window_height = new_height
            self.root.geometry(f"{window_width}x{window_height}")
            
            self.zoom_factor = new_width / self.original_image.width
            self.update_image()
    
    def display_image(self):
        if self.current_image:
            self.photo = ImageTk.PhotoImage(self.current_image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=NW, image=self.photo)
            self.canvas.config(scrollregion=self.canvas.bbox(ALL))
    
    def save_image(self):
        if self.current_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                     filetypes=[("PNG", "*.png"), 
                                                                ("JPEG", "*.jpg"), 
                                                                ("BMP", "*.bmp"), 
                                                                ("TIFF", "*.tiff")])
            if file_path:
                try:
                    self.current_image.save(file_path)
                    messagebox.showinfo("Guardado", "Imagen guardada exitosamente.")
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo guardar la imagen: {str(e)}")
        else:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar.")
    
    def crop_image(self):
        if self.current_image:
            self.canvas.bind("<ButtonPress-1>", self.start_crop)
            self.canvas.bind("<B1-Motion>", self.draw_crop)
            self.canvas.bind("<ButtonRelease-1>", self.end_crop)
    
    def start_crop(self, event):
        self.crop_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
    
    def draw_crop(self, event):
        if self.crop_start:
            x, y = self.crop_start
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
            self.crop_rect = self.canvas.create_rectangle(x, y, 
                                                          self.canvas.canvasx(event.x), 
                                                          self.canvas.canvasy(event.y), 
                                                          outline="red")
    
    def end_crop(self, event):
        if self.crop_start:
            x, y = self.crop_start
            end_x, end_y = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            if x != end_x and y != end_y:
                scale_x = self.original_image.width / self.current_image.width
                scale_y = self.original_image.height / self.current_image.height
                crop_area = (
                    int(min(x, end_x) * scale_x),
                    int(min(y, end_y) * scale_y),
                    int(max(x, end_x) * scale_x),
                    int(max(y, end_y) * scale_y)
                )
                self.original_image = self.original_image.crop(crop_area)
                self.initial_image = self.original_image.copy()
                self.reset_adjustments()
                self.update_image()
            self.crop_start = None
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
    
    def mouse_wheel(self, event):
        if self.current_image:
            if event.num == 5 or event.delta == -120:  # Zoom out
                self.apply_zoom(0.9)
            elif event.num == 4 or event.delta == 120:  # Zoom in
                self.apply_zoom(1.1)
    
    def apply_zoom(self, factor):
        if self.current_image:
            self.zoom_factor *= factor
            self.update_image()
    
    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)
    
    def pan_image(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
    
    def move_image(self, dx, dy):
        self.canvas.move(ALL, dx, dy)
    
    def original_size(self):
        if self.original_image:
            self.zoom_factor = 1.0
            self.update_image()
    
    def update_image(self, *args):
        if self.original_image:
            brightness = self.brightness_scale.get()
            contrast = self.contrast_scale.get()
            saturation = self.saturation_scale.get()
            
            # Aplicar ajustes
            adjusted_image = self.original_image.copy()
            adjusted_image = ImageEnhance.Brightness(adjusted_image).enhance(brightness)
            adjusted_image = ImageEnhance.Contrast(adjusted_image).enhance(contrast)
            adjusted_image = ImageEnhance.Color(adjusted_image).enhance(saturation)
            
            # Aplicar rotación
            if self.rotation != 0:
                adjusted_image = adjusted_image.rotate(self.rotation, expand=True)
            
            # Aplicar zoom
            new_width = int(adjusted_image.width * self.zoom_factor)
            new_height = int(adjusted_image.height * self.zoom_factor)
            self.current_image = adjusted_image.copy()
            self.current_image.thumbnail((new_width, new_height), Image.LANCZOS)
            
            self.display_image()
    
    def rotate_image(self, degrees):
        if self.original_image:
            self.rotation = (self.rotation + degrees) % 360
            self.update_image()
    
    def reset_adjustments(self):
        self.brightness_scale.set(1)
        self.contrast_scale.set(1)
        self.saturation_scale.set(1)
        self.rotation = 0
        if self.initial_image:
            self.original_image = self.initial_image.copy()
            self.update_image()

if __name__ == "__main__":
    root = ttk.Window("Photo-W v1.6")
    app = PhotoW(root)
    root.mainloop()