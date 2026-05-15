import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from skimage import color
import os
from model_lab import ColorizationCNN
import threading

class ColorizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Colorization")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1e1e2e")
        
        # Variables
        self.original_image = None
        self.colorized_image = None
        self.current_image_path = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.load_model()
        
        # Create GUI
        self.create_widgets()
        
    def load_model(self):
        """Load the trained colorization model"""
        try:
            self.model = ColorizationCNN().to(self.device)
            if os.path.exists("color_model_lab.pth"):
                self.model.load_state_dict(torch.load("color_model_lab.pth", map_location=self.device))
                self.model.eval()
                print("✓ Model loaded successfully!")
            else:
                messagebox.showerror("Error", "Model file 'color_model_lab.pth' not found!\nPlease train the model first.")
                self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title Bar
        title_frame = tk.Frame(self.root, bg="#2d2d44", height=80)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(
            title_frame,
            text="🎨 AI Image Colorization",
            font=("Helvetica", 28, "bold"),
            bg="#2d2d44",
            fg="#ffffff"
        )
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(
            title_frame,
            text="Transform black & white images into vibrant colors using AI",
            font=("Helvetica", 11),
            bg="#2d2d44",
            fg="#a0a0b0"
        )
        subtitle_label.pack()
        
        # Main Container
        main_frame = tk.Frame(self.root, bg="#1e1e2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left Panel - Original Image
        left_frame = tk.Frame(main_frame, bg="#2d2d44", relief=tk.RAISED, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        left_title = tk.Label(
            left_frame,
            text="📷 Original Image",
            font=("Helvetica", 14, "bold"),
            bg="#2d2d44",
            fg="#ffffff"
        )
        left_title.pack(pady=10)
        
        self.original_canvas = tk.Canvas(
            left_frame,
            bg="#1e1e2e",
            highlightthickness=0
        )
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Placeholder text for original
        self.original_placeholder = self.original_canvas.create_text(
            300, 250,
            text="No image loaded\n\nClick 'Upload Image' to start",
            font=("Helvetica", 12),
            fill="#6c6c7c",
            justify=tk.CENTER
        )
        
        # Right Panel - Colorized Image
        right_frame = tk.Frame(main_frame, bg="#2d2d44", relief=tk.RAISED, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        right_title = tk.Label(
            right_frame,
            text="🌈 Colorized Result",
            font=("Helvetica", 14, "bold"),
            bg="#2d2d44",
            fg="#ffffff"
        )
        right_title.pack(pady=10)
        
        self.colorized_canvas = tk.Canvas(
            right_frame,
            bg="#1e1e2e",
            highlightthickness=0
        )
        self.colorized_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Placeholder text for colorized
        self.colorized_placeholder = self.colorized_canvas.create_text(
            300, 250,
            text="Colorized image will appear here\n\nAfter processing",
            font=("Helvetica", 12),
            fill="#6c6c7c",
            justify=tk.CENTER
        )
        
        # Bottom Control Panel
        control_frame = tk.Frame(self.root, bg="#2d2d44", height=100)
        control_frame.pack(fill=tk.X, padx=20, pady=(10, 20))
        
        # Buttons Container
        button_container = tk.Frame(control_frame, bg="#2d2d44")
        button_container.pack(expand=True)
        
        # Upload Button
        self.upload_btn = tk.Button(
            button_container,
            text="📁 Upload Image",
            command=self.upload_image,
            font=("Helvetica", 12, "bold"),
            bg="#4a90e2",
            fg="white",
            activebackground="#357abd",
            activeforeground="white",
            relief=tk.FLAT,
            padx=30,
            pady=15,
            cursor="hand2"
        )
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        # Colorize Button
        self.colorize_btn = tk.Button(
            button_container,
            text="🎨 Colorize",
            command=self.colorize_image,
            font=("Helvetica", 12, "bold"),
            bg="#50c878",
            fg="white",
            activebackground="#3da65f",
            activeforeground="white",
            relief=tk.FLAT,
            padx=30,
            pady=15,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.colorize_btn.pack(side=tk.LEFT, padx=10)
        
        # Save Button
        self.save_btn = tk.Button(
            button_container,
            text="💾 Save Result",
            command=self.save_image,
            font=("Helvetica", 12, "bold"),
            bg="#b19cd9",
            fg="white",
            activebackground="#9b7ec4",
            activeforeground="white",
            relief=tk.FLAT,
            padx=30,
            pady=15,
            cursor="hand2",
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=10)
        
        # Clear Button
        self.clear_btn = tk.Button(
            button_container,
            text="🗑️ Clear",
            command=self.clear_images,
            font=("Helvetica", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            relief=tk.FLAT,
            padx=30,
            pady=15,
            cursor="hand2"
        )
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Status Bar
        self.status_label = tk.Label(
            self.root,
            text=f"Ready | Device: {self.device}",
            font=("Helvetica", 9),
            bg="#2d2d44",
            fg="#a0a0b0",
            anchor=tk.W,
            padx=20
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Progress Bar
        self.progress = ttk.Progressbar(
            control_frame,
            mode='indeterminate',
            length=300
        )
        
    def upload_image(self):
        """Upload and display an image"""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image
                self.original_image = Image.open(file_path).convert("RGB")
                self.current_image_path = file_path
                
                # Display original image
                self.display_image(self.original_image, self.original_canvas, self.original_placeholder)
                
                # Clear colorized canvas
                self.colorized_canvas.delete("all")
                self.colorized_placeholder = self.colorized_canvas.create_text(
                    300, 250,
                    text="Click 'Colorize' to process",
                    font=("Helvetica", 12),
                    fill="#6c6c7c"
                )
                
                # Enable colorize button
                self.colorize_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.DISABLED)
                
                filename = os.path.basename(file_path)
                self.status_label.config(text=f"Loaded: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image, canvas, placeholder_id=None):
        """Display image on canvas with proper scaling"""
        # Remove placeholder if exists
        if placeholder_id:
            canvas.delete(placeholder_id)
        
        # Get canvas dimensions
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Calculate scaling to fit canvas
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * 0.9
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized_image)
        
        # Display on canvas
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo,
            anchor=tk.CENTER
        )
        
        # Keep reference to prevent garbage collection
        canvas.image = photo
    
    def colorize_image(self):
        """Colorize the uploaded image using the model"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        # Disable buttons during processing
        self.colorize_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Processing... Please wait")
        
        # Show progress bar
        self.progress.pack(pady=10)
        self.progress.start(10)
        
        # Run colorization in separate thread
        thread = threading.Thread(target=self._colorize_thread)
        thread.start()
    
    def _colorize_thread(self):
        """Thread function for colorization"""
        try:
            # Convert to grayscale
            img_gray = self.original_image.convert('L')
            img_array = np.array(img_gray)
            
            # Convert to LAB
            img_lab = color.rgb2lab(np.array(self.original_image))
            L_channel = img_lab[:, :, 0]
            
            # Normalize L channel
            L_normalized = (L_channel - 50.0) / 50.0
            L_tensor = torch.FloatTensor(L_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                ab_predicted = self.model(L_tensor)
            
            # Convert back to image
            ab_numpy = ab_predicted.squeeze().cpu().numpy()
            ab_numpy = np.transpose(ab_numpy, (1, 2, 0))
            ab_numpy = ab_numpy * 110.0
            
            # Resize ab channels to match L channel
            from skimage.transform import resize
            ab_resized = resize(ab_numpy, (L_channel.shape[0], L_channel.shape[1]), anti_aliasing=True)
            
            # Combine L and ab
            lab_combined = np.zeros((L_channel.shape[0], L_channel.shape[1], 3))
            lab_combined[:, :, 0] = L_channel
            lab_combined[:, :, 1:] = ab_resized
            
            # Convert to RGB
            rgb_image = color.lab2rgb(lab_combined)
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
            self.colorized_image = Image.fromarray(rgb_image)
            
            # Update GUI in main thread
            self.root.after(0, self._update_colorized_display)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Colorization failed: {str(e)}"))
            self.root.after(0, self._reset_buttons)
    
    def _update_colorized_display(self):
        """Update GUI with colorized image"""
        self.progress.stop()
        self.progress.pack_forget()
        
        self.display_image(self.colorized_image, self.colorized_canvas, self.colorized_placeholder)
        
        self.save_btn.config(state=tk.NORMAL)
        self.colorize_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        self.status_label.config(text="✓ Colorization complete!")
    
    def _reset_buttons(self):
        """Reset button states"""
        self.progress.stop()
        self.progress.pack_forget()
        self.colorize_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Ready")
    
    def save_image(self):
        """Save the colorized image"""
        if self.colorized_image is None:
            messagebox.showwarning("Warning", "No colorized image to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.colorized_image.save(file_path, quality=95)
                messagebox.showinfo("Success", f"Image saved successfully!\n{file_path}")
                self.status_label.config(text=f"Saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def clear_images(self):
        """Clear all images"""
        self.original_canvas.delete("all")
        self.colorized_canvas.delete("all")
        
        self.original_placeholder = self.original_canvas.create_text(
            300, 250,
            text="No image loaded\n\nClick 'Upload Image' to start",
            font=("Helvetica", 12),
            fill="#6c6c7c",
            justify=tk.CENTER
        )
        
        self.colorized_placeholder = self.colorized_canvas.create_text(
            300, 250,
            text="Colorized image will appear here\n\nAfter processing",
            font=("Helvetica", 12),
            fill="#6c6c7c",
            justify=tk.CENTER
        )
        
        self.original_image = None
        self.colorized_image = None
        self.current_image_path = None
        
        self.colorize_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Ready")

def main():
    root = tk.Tk()
    app = ColorizationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()