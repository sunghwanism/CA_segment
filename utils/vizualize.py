import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from ipywidgets import interact, IntSlider
import pydicom
import numpy as np


def check_roi_alignment(image_hu, mask, slice_idx):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_hu, cmap='gray', vmin=-100, vmax=500)
    plt.title(f"Original Slice {slice_idx}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_hu, cmap='gray', vmin=-100, vmax=500)
    plt.imshow(mask, cmap='Reds', alpha=0.5) 
    plt.title(f"Red Mask Overlay (Score=0?)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    roi_pixels = image_hu[mask > 0]
    if roi_pixels.size > 0:
        print(f"📊 ROI Stats for Slice {slice_idx}:")
        print(f"   - Max HU: {roi_pixels.max():.2f} (Should be >= 130)")
        print(f"   - Min HU: {roi_pixels.min():.2f}")
        print(f"   - Mean HU: {roi_pixels.mean():.2f}")
    else:
        print("⚠️ Warning: Mask is empty (No pixels selected).")


def visualize_dicom_w_roi(volume, json_data):
    """
    volume: List of pydicom objects
    json_data: ROI dictionary assuming polygon coordinate structure:
               { "image_idx": [ {"points": [[x1,y1], [x2,y2]...], "class_name": "...", ...} ] }
    """
    
    z_to_roi_map = {}
    unique_classes = set()

    for roi_list in json_data.values():
        for item in roi_list:
            unique_classes.add(item.get('class_name', 'Unknown'))
            
    sorted_classes = sorted(list(unique_classes))

    cmap = cm.get_cmap('tab10', len(sorted_classes) if sorted_classes else 1)
    class_to_color = {name: cmap(i) for i, name in enumerate(sorted_classes)}

    for z, dcm in enumerate(volume):
        try:
            dicom_key = str(dcm.InstanceNumber)
            
            if dicom_key in json_data:
                z_to_roi_map[z] = json_data[dicom_key]
                    
        except AttributeError:
            continue
            
    max_slice = len(volume) - 1

    def view_slice(z):
        dcm = volume[z]
        img_arr = dcm.pixel_array
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_arr, cmap='gray')
        
        if z in z_to_roi_map:
            roi_list = z_to_roi_map[z]
            legend_labels = set()
            
            for roi_item in roi_list:
                points_data = roi_item.get('points', [])
                class_name = roi_item.get('class_name', 'Unknown')
                roi_color = class_to_color.get(class_name, 'red')

                if points_data and isinstance(points_data[0], list):
                    
                    coords = np.array(points_data)
                    
                    poly_patch = patches.Polygon(
                        coords,
                        closed=True,
                        facecolor=roi_color,
                        alpha=0.4,
                        edgecolor=roi_color,
                        linewidth=2,
                        label=class_name if class_name not in legend_labels else "_nolegend_"
                    )
                    ax.add_patch(poly_patch)
                    legend_labels.add(class_name)

            if legend_labels:
                ax.legend(loc='upper right', framealpha=1.0)
        
        inst_num = getattr(dcm, 'InstanceNumber', 'N/A')
        ax.set_title(f"Slice Index: {z} | InstanceNumber: {inst_num} | Objects: {len(z_to_roi_map.get(z, []))}")
        ax.axis('off')
        plt.show()

    interact(view_slice, 
             z=IntSlider(min=0, max=max_slice, step=1, value=0, description='Slice Idx:'))