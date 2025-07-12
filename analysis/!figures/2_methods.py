
import numpy as np
import time
import cv2; import math
import csv; import pathlib

num = 4
mcs = "1200"
run = "cancore2_scan-28"
cwd = "C:/Nexus/Academic/#OneDrive [IISc]/Coursework/Projects/2025-04 [Ramray Bhat]/Notes/Misc/"
for rep in range(1, num+1):
    src = cwd

    def image_analyser(image_path):
        # Read image in color (OpenCV loads as BGR by default)
        img = cv2.imread(str(image_path))
        # Apply a Gaussian blur to smooth edges
        img = cv2.GaussianBlur(img,(5,5),0)
        # Define color range corresponding to cells in BGR
        lower = np.array([160, 80, 80])
        upper = np.array([240, 160, 160])
        # Threshold the BGR image to get only cells
        mask = cv2.inRange(img, lower, upper)
        
        # Show thresholded image in the interactive window
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.show()
        
        # Morphological closing to connect nearby cells
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) # Set to (25,25) for dispersed modes
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Find contours and select the largest as the main cluster
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0, 0, 0, 0, "none"
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Show image with main cluster highlighted
        main_cluster_img = img.copy()
        cv2.drawContours(main_cluster_img, [largest_contour], 0, (0, 0, 255), 2)
        plt.figure()
        plt.imshow(cv2.cvtColor(main_cluster_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
        # Fit ellipse if enough points
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (xc, yc), (major, minor), rotation = ellipse
            major_axis = max(major, minor)
            eccentricity = math.sqrt(1 - (min(major, minor)/major_axis)**2)
            mean_diameter = math.sqrt(major * minor)
            
            # Show image with ellipse drawn
            ellipse_img = img.copy()
            cv2.ellipse(ellipse_img, ellipse, (0, 0, 255), 4)
            plt.figure()
            plt.imshow(cv2.cvtColor(ellipse_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        else: 
            major_axis = 0
            eccentricity = 0
            mean_diameter = 0
            rotation = 45
        # Count cells inside the ellipse
        mask_ellipse = np.zeros_like(mask)
        if len(largest_contour) >= 5:
            cv2.ellipse(mask_ellipse, ellipse, 255, -1)
            cells_in_ellipse = cv2.bitwise_and(mask, mask_ellipse)
        else: cells_in_ellipse = mask
        cell_area = np.count_nonzero(cells_in_ellipse)
        # Calculate density: fraction of pixels in the ellipse that are cells
        if major_axis > 0:
            ellipse_pixels = np.count_nonzero(mask_ellipse)
            cell_pixels = np.count_nonzero(cells_in_ellipse)
            density = cell_pixels / ellipse_pixels if ellipse_pixels > 0 else 0
        else: density = 0
        # Find connected objects (cells)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
        # Calculate convolution: total perimeter divided by expected perimeter
        if num_labels > 0:
            total_perimeter = 0; total_expected_perimeter = 0
            for label in range(1, num_labels):
                obj_mask = (labels == label).astype(np.uint8)
                contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                area = cv2.countNonZero(obj_mask)
                perimeter = cv2.arcLength(contours[0], True)
                total_expected_perimeter += 2*math.pi*math.sqrt(area/math.pi)
                total_perimeter += perimeter
            if total_expected_perimeter > 0:
                convolution = total_perimeter / total_expected_perimeter
            else: convolution = 0
        else: convolution = 0
        # Compute hypoxic fraction: hypoxic pixels / total mask pixels
        hypoxic_lower = np.array([160, 120, 120])
        hypoxic_mask = cv2.inRange(img, hypoxic_lower, upper)
        hypoxic_pixels = np.count_nonzero(cv2.bitwise_and(hypoxic_mask, mask))
        total_mask_pixels = np.count_nonzero(mask)
        hypoxic_fraction = hypoxic_pixels / total_mask_pixels if total_mask_pixels > 0 else 0
        # Compute density profile in concentric elliptical shells
        density_profile = []
        hypoxic_profile = []
        if len(largest_contour) >= 5 and major > 0 and minor > 0:
            # Calculate the geometric mean diameters
            mean_diameters = np.arange(25, 626, 25)
            # Calculate the scaling factors for each ellipse
            base_mean = math.sqrt(major * minor)
            for i in range(len(mean_diameters)):
                d = mean_diameters[i]
                scale = d / base_mean
                # Compute scaled axes
                scaled_major = major * scale
                scaled_minor = minor * scale
                # Draw outer ellipse
                mask_outer = np.zeros_like(mask)
                cv2.ellipse(mask_outer, ((xc, yc), (scaled_major, scaled_minor), rotation), 255, -1)
                if i == 0:
                    mask_inner = np.zeros_like(mask)
                else:
                    # Draw inner ellipse
                    d_inner = mean_diameters[i-1]
                    scale_inner = d_inner / base_mean
                    scaled_major_inner = major * scale_inner
                    scaled_minor_inner = minor * scale_inner
                    mask_inner = np.zeros_like(mask)
                    cv2.ellipse(mask_inner, ((xc, yc), (scaled_major_inner, scaled_minor_inner), rotation), 255, -1)
                # Shell mask is outer minus inner
                shell_mask = cv2.subtract(mask_outer, mask_inner)
                # Total density in shell
                cell_pixels_in_shell = np.count_nonzero(cv2.bitwise_and(mask, shell_mask))
                shell_area = np.count_nonzero(shell_mask)
                density_shell = cell_pixels_in_shell / shell_area if shell_area > 0 else 0
                density_profile.append(density_shell)
                # Hypoxic density in shell
                hypoxic_pixels_in_shell = np.count_nonzero(cv2.bitwise_and(hypoxic_mask, shell_mask))
                hypoxic_density_shell = hypoxic_pixels_in_shell / shell_area if shell_area > 0 else 0
                hypoxic_profile.append(hypoxic_density_shell)
        # Compute densities for specified mean diameter ranges
        ranges = [
            (0, max(125, mean_diameter/2)),
            (max(mean_diameter/2, mean_diameter-125), mean_diameter)
        ]
        densities = []
        for d_in, d_out in ranges:
            if len(largest_contour) >= 5 and major > 0 and minor > 0:
                scale_in = d_in / base_mean if base_mean > 0 else 0
                scale_out = d_out / base_mean if base_mean > 0 else 0
                # Draw outer ellipse
                mask_outer = np.zeros_like(mask)
                cv2.ellipse(mask_outer, ((xc, yc), (major * scale_out, minor * scale_out), rotation), 255, -1)
                # Draw inner ellipse
                mask_inner = np.zeros_like(mask)
                if d_in > 0:
                    cv2.ellipse(mask_inner, ((xc, yc), (major * scale_in, minor * scale_in), rotation), 255, -1)
                # Shell mask is outer minus inner
                shell_mask = cv2.subtract(mask_outer, mask_inner)
                cell_pixels_in_shell = np.count_nonzero(cv2.bitwise_and(mask, shell_mask))
                shell_area = np.count_nonzero(shell_mask)
                density_shell = cell_pixels_in_shell / shell_area if shell_area > 0 else 0
                densities.append(density_shell)
            else: densities.append(0)
        cen_density, mar_density = densities

        scale = 4.0*100/320 # Convert from pixels to microns
        # Return: number of clusters, total cell area, hypoxic fraction, major axis, eccentricity, rotation, mean diameter, density, convolution, density profile, centre density, margin density, hypoxic density profile
        return num_labels, (cell_area/scale**2), hypoxic_fraction, (major_axis/scale), eccentricity, rotation, (mean_diameter/scale), density, convolution, density_profile, cen_density, mar_density, hypoxic_profile

    src = pathlib.Path(src)
    images = sorted(src.glob("**/*.png"), key=lambda x: int(''.join(filter(str.isdigit, x.stem))))    
    for img in images:
        image_analyser(img)   
        time.sleep(2)