from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import FileResponse
from rest_framework.parsers import MultiPartParser, FormParser
import json
import os
from rest_framework import status
import shutil
import glob
from urllib.parse import unquote
import re
import numpy as np
import nibabel as nib
import imageio
import aim2numpy
from scipy.ndimage import gaussian_filter
from PIL import Image
import concurrent.futures
import io
import time
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import trimesh
import base64
import tempfile
import skimage.measure as measure
import skimage.transform as transform
import pygltflib
from scipy import ndimage
from scipy import signal
import gc
import sys
import psutil
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import spatial
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import skimage.morphology
from scipy.signal import convolve
from .ALPACA import ALPACA
from skimage import exposure
import os
# Set the number of threads for ITK to utilize
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(psutil.cpu_count(logical=True))
import ants
import shutil
from scipy.interpolate import RegularGridInterpolator
import open3d as o3d
from scipy.spatial import cKDTree

alpaca = ALPACA()

class SmartLandmarkingView(APIView):
    def post(self, request):
        print("Applying smart landmarking")
        directory = request.data['directory']
        reference_subject = request.data['reference_subject']
        target = request.data['target']

        # Send progress update before each scan is processed
        channel_layer = get_channel_layer()
        if channel_layer is not None:
            progress = 0
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': progress,
                    'scan_name': target,
                    'custom_message': f'Loading target image data...',
                    'total': 4,
                    'current': 0,
                }
            )

        covariance_threshold = request.data.get('covariance_threshold', 0.5)  # Default value if not provided
        max_landmark_region_radius = request.data.get('max_landmark_region_radius', 1)  # Default 1 mm
        variance_percentage = request.data.get('variance_percentage', 0.95)  # Default 95%
        
        #  ------------- find latest atlas edit -------------
        atlas_files = glob.glob(os.path.join(directory, "atlas", "atlas*.nii.gz"))
        atlas_edit_num = -1
        if(any("edit" in atlas_file for atlas_file in atlas_files)):
            for atlas_file in atlas_files:
                if "edit" in atlas_file:
                    atlas_edit_num = max(atlas_edit_num, int(atlas_file.split("_edit_")[1].split("_")[0]))
            atlas_latest_edit = os.path.basename(glob.glob(os.path.join(directory, "atlas", f"atlas_edit_{atlas_edit_num}*.nii.gz"))[0])
        else:
            atlas_latest_edit = "atlas.nii.gz"

        # Load the nifti data from edit
        nifti_path = os.path.join(directory, "atlas", atlas_latest_edit)
        nifti_img = nib.load(nifti_path)
        atlas_nifti_data = nifti_img.get_fdata().astype(np.int16)

        # Apply Gaussian smoothing
        atlas_nifti_data = gaussian_filter(atlas_nifti_data, sigma=1.0)

        # Load the json metadata
        json_path = os.path.join(directory, "atlas", "atlas.json")
        with open(json_path, 'r') as jf:
            json_metadata = json.load(jf)
            
        # Get threshold and voxel size from json metadata
        atlas_threshold = json_metadata['threshold']
        atlas_voxel_size = json_metadata['voxel_size']

        # Apply Marching Cubes to extract the mesh
        atlas_vertices, atlas_faces, _, _ = measure.marching_cubes(
            atlas_nifti_data, 
            level=atlas_threshold, 
            spacing=(atlas_voxel_size, atlas_voxel_size, atlas_voxel_size)
        )        

        # Invert face normals by reversing the order of vertices in each face
        atlas_faces = atlas_faces[:, ::-1]
        

        #  ------------- find latest reference edit -------------
        reference_files = glob.glob(os.path.join(directory, "extracted", reference_subject, "*.nii.gz"))
        reference_edit_num = -1
        if(any("edit" in reference_file for reference_file in reference_files)):
            for reference_file in reference_files:
                if "edit" in reference_file:
                    reference_edit_num = max(reference_edit_num, int(reference_file.split("_edit_")[1].split("_")[0]))
            reference_latest_edit = os.path.basename(glob.glob(os.path.join(directory, "extracted", reference_subject, f"{reference_subject}_edit_{reference_edit_num}*.nii.gz"))[0])
        else:
            reference_latest_edit = reference_subject + ".nii.gz"

        # Load the nifti data from edit
        nifti_path = os.path.join(directory, "extracted", reference_subject, reference_latest_edit)
        nifti_img = nib.load(nifti_path)
        reference_nifti_data = nifti_img.get_fdata().astype(np.int16)
        
        # Load the json metadata
        json_path = os.path.join(directory, "extracted", reference_subject, f"{reference_subject}.json")
        with open(json_path, 'r') as jf:
            json_metadata = json.load(jf)
            
        # Get threshold and voxel size from json metadata
        reference_threshold = json_metadata['threshold']
        reference_voxel_size = json_metadata['voxel_size']
        
        # Apply Gaussian smoothing
        volume = gaussian_filter(reference_nifti_data, sigma=1.0)
        
        # Apply Marching Cubes to extract the mesh
        vertices, faces, _, _ = measure.marching_cubes(
            volume, 
            level=reference_threshold, 
            spacing=(reference_voxel_size, reference_voxel_size, reference_voxel_size)
        )

        # Invert face normals by reversing the order of vertices in each face
        faces = faces[:, ::-1]

        # Send progress update before each scan is processed
        channel_layer = get_channel_layer()
        if channel_layer is not None:
            progress = 0
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': progress,
                    'scan_name': target,
                    'custom_message': f'Creating pseudo-landmarks...',
                    'total': 4,
                    'current': 1,
                }
            )
        
        # STEP 1: Create pseudo-landmarks from voxels that intersect the mesh surface
        pseudo_landmarks, pseudo_landmarks_indices = self.create_pseudo_landmarks_from_mesh_intersection(
            reference_nifti_data, 
            vertices, 
            faces, 
            reference_voxel_size,
            reference_threshold,
            directory
        )

        print(len(pseudo_landmarks), " pseudo-landmarks were created.")

        # Send progress update before each scan is processed
        channel_layer = get_channel_layer()
        if channel_layer is not None:
            progress = 1/4
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': progress,
                    'scan_name': target,
                    'custom_message': f'{len(pseudo_landmarks)} pseudo-landmarks were created. Filtering #1: keeping landmarks under {variance_percentage*100}% cumulative variance...',
                    'total': 4,
                    'current': 2,
                }
            )

        # STEP 2: For the reference, compute variance of deformation field vectors at each chosen landmark
        # This is where we implement the full population-based approach     
         
        # Filter the pseudo-landmarks to only keep the highest deformation field variance landmarks
        pseudo_landmarks, pseudo_landmarks_indices, pseudo_landmarks_fields, pseudo_landmarks_variances = self.filter_landmarks_by_variance(directory, pseudo_landmarks, pseudo_landmarks_indices, reference_subject, variance_percentage)

        print(len(pseudo_landmarks), " remaining pseudo landmarks after variance filtering.")

        # Plot the pseudo-landmarks
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(pseudo_landmarks[:, 0], pseudo_landmarks[:, 1], pseudo_landmarks[:, 2], s=1)
        plt.savefig(os.path.join(directory, "atlas", "pseudo_landmarks.png"))
        plt.close()

        print("Post filtering pseudo_landmarks.shape: ", pseudo_landmarks.shape)

        # Send progress update before each scan is processed
        channel_layer = get_channel_layer()
        if channel_layer is not None:
            progress = 2/4
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': progress,
                    'scan_name': target,
                    'custom_message': f'{len(pseudo_landmarks)} landmarks remaining after variance filtering. Filtering #2: keeping landmarks over {covariance_threshold} covariance inside a {max_landmark_region_radius} mm radius...',
                    'total': 4,
                    'current': 3,
                }
            )

        # STEP 3: Filter landmarks by variance and covariance
        final_landmarks = self.filter_landmarks_by_covariance_regions(
            pseudo_landmarks,
            pseudo_landmarks_fields,
            pseudo_landmarks_variances,            
            covariance_threshold,
            max_landmark_region_radius,
            reference_voxel_size,
            target
        )

        print(len(final_landmarks), " remaining landmarks after covariance filtering.")

        # Plot the final landmarks
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(final_landmarks[:, 0], final_landmarks[:, 1], final_landmarks[:, 2], s=1)
        plt.savefig(os.path.join(directory, "atlas", "final_landmarks.png"))
        plt.close()

        print("Final landmarks.shape: ", final_landmarks.shape)

        # Send progress update before each scan is processed
        channel_layer = get_channel_layer()
        if channel_layer is not None:
            progress = 3/4
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': progress,
                    'scan_name': target,
                    'custom_message': f'{len(final_landmarks)} landmarks remaining after covariance filtering. Displacing landmarks and saving...',
                    'total': 4,
                    'current': 4,
                }
            )

        # STEP 4: Save the landmarks as the reference subject latest edit landmarks
        if target == "reference":
            landmarks_path = os.path.join(directory, "extracted", reference_subject, reference_latest_edit.replace(".nii.gz", "_landmarks.json"))
            with open(landmarks_path, 'w') as jf:
                json.dump(final_landmarks.tolist(), jf, indent=4)

            # Distance analysis
            self.distance_analysis(directory, "extracted", reference_subject.replace(".nii.gz", ""), final_landmarks)

        elif target == "atlas":
            # Map the final_landmarks to the atlas

            # Load the atlas' average forward deformation field
            atlas_average_forward_deformation = ants.image_read(os.path.join(directory, "atlas", f"{reference_subject}_average_forward_transformation_deformation.nii.gz"))
            atlas_average_forward_deformation = atlas_average_forward_deformation.numpy()

            # Invert the deformation field only on the x and y vectors as observed in trial and error
            atlas_average_forward_deformation[:, :, :, 0] = -atlas_average_forward_deformation[:, :, :, 0]
            atlas_average_forward_deformation[:, :, :, 1] = -atlas_average_forward_deformation[:, :, :, 1]
            print(atlas_average_forward_deformation.shape)

            # Displace the landmarks by the closest field vector
            for i, landmark in enumerate(final_landmarks):
                final_landmarks[i] = self.displace_coordinate_by_closest_field_vector(landmark, atlas_average_forward_deformation, reference_voxel_size)  
            
            try:
                snapped_landmarks = self.snap_landmarks_to_mesh(final_landmarks, atlas_vertices, atlas_faces, atlas_voxel_size)
                final_landmarks = snapped_landmarks
            except Exception as e:
                print(f"Error snapping landmarks to mesh: {e}")

            # Perform distance analysis
            self.distance_analysis(directory, "atlas", atlas_latest_edit.replace(".nii.gz", ""), final_landmarks)

            # Save the landmarks
            landmarks_path = os.path.join(directory, "atlas", atlas_latest_edit.replace(".nii.gz", "_landmarks.json"))
            print(f"Saving landmarks to {landmarks_path}")
            with open(landmarks_path, 'w') as jf:
                json.dump(final_landmarks.tolist(), jf, indent=4)

        # Send progress update before each scan is processed
        channel_layer = get_channel_layer()
        if channel_layer is not None:
            progress = 4/4
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': progress,
                    'scan_name': target,
                    'custom_message': f'Landmarks successfully created.',
                    'total': 4,
                    'current': 4,
                }
            )


        return Response({'message': 'Smart landmarking applied successfully'}, status=status.HTTP_200_OK)


    def distance_analysis(self, directory, filename, edit, landmarks):

        # Get the sub path
        if "atlas" in filename:
            sub_path = "atlas"
        else:
            sub_path = "extracted/" + filename


        # Load the json metadata
        json_path = os.path.join(directory, sub_path, f"{filename}.json")
        with open(json_path, 'r') as jf:
            json_metadata = json.load(jf)

        # Get threshold from json metadata
        threshold = json_metadata['threshold']

        # Get number of landmarks
        num_landmarks = len(landmarks)

        # Load the nifti data from filename
        nifti_path = os.path.join(directory, sub_path, edit + ".nii.gz")
        nifti_img = nib.load(nifti_path)
        nifti_data = nifti_img.get_fdata().astype(np.int16)
        nifti_data = gaussian_filter(nifti_data, sigma=1.0)
        # Render the mesh with marchingcubes
        vertices, faces, _, _ = measure.marching_cubes(nifti_data, level=threshold, spacing=(json_metadata['voxel_size'], json_metadata['voxel_size'], json_metadata['voxel_size']))

        # Calculate the shortest distance between each of the landmarks and the mesh
        distances = alpaca.calculate_landmark_distances(landmarks, vertices, faces)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Number of outliers farther than 1 and 6 voxel distance equivalents
        outliers = np.sum(distances > json_metadata['voxel_size'])
        outliers_six = np.sum(distances > json_metadata['voxel_size'] * 6)

        # Save the distances to a json file
        distances_path = os.path.join(directory, sub_path, edit + "_landmark_distances.json")
        with open(distances_path, 'w') as jf:
            json.dump(distances.tolist(), jf, indent=4)  # Convert ndarray to list for JSON compatibility

        # Add metadata entry named landmarks, within it add the number of landmarks and the accuracy
        json_metadata['landmarks'] = {
            'num_landmarks': int(num_landmarks),
            'mean_distance': float(mean_distance),
            'std_distance': float(std_distance),
            'outliers': int(outliers),
            'outliers_six': int(outliers_six),
        }
        # Save the updated metadata to the json file
        with open(json_path, 'w') as jf:
            json.dump(json_metadata, jf, indent=4)

    
    def create_pseudo_landmarks_from_mesh_intersection(self, nifti_data, vertices, faces, voxel_size, threshold, directory):
        """
        Create pseudo-landmarks from voxels that intersect with the mesh surface.
        
        Args:
            nifti_data: 3D volume data
            vertices: Mesh vertices
            faces: Mesh faces
            voxel_size: Size of voxels
            threshold: Threshold for mesh surface
        Returns:
            List of pseudo-landmark coordinates
        """

        # Dilate the thresholded nifti data by 1 voxel
        #dilated_mask = skimage.morphology.binary_dilation(nifti_data > threshold)
        sobel = skimage.filters.sobel(nifti_data > threshold)

        # Make the center of each nifti data voxel that corresponds to a dilated mask voxel a landmark with coordinates at the center of the voxel
        landmarks = []
        indices = []
        for i in range(nifti_data.shape[0]):
            for j in range(nifti_data.shape[1]):
                for k in range(nifti_data.shape[2]):
                    if sobel[i, j, k]:
                        landmarks.append([(i + 0.5) * voxel_size, (j + 0.5) * voxel_size, (k + 0.5) * voxel_size])
                        indices.append([i, j, k])

        landmarks = np.array(landmarks)
        indices = np.array(indices)       

        # Snap landmarks to the mesh
        landmarks, indices = self.snap_landmarks_to_mesh(landmarks, vertices, faces, voxel_size, indices=indices)

        return landmarks, indices

    def filter_landmarks_by_variance(self, directory, landmarks, indices, reference_subject, variance_percentage):
        """
        Filter landmarks by variance of deformation field vectors
        
        Args:
            directory: Base directory
            landmarks: Landmarks to filter
            indices: Indices of the landmarks
            reference_subject: Reference subject
            variance_percentage: Percentage of variance to retain
        Returns:
            Filtered landmarks, deformation field vectors at each landmark
        """

        # For each folder in the extracted folder (except the reference), load the deformation field *_elastic_fwd.nii.gz
        extracted_folders = os.listdir(os.path.join(directory, "extracted"))
        extracted_folders = [os.path.basename(f) for f in extracted_folders 
                           if f != reference_subject 
                           and f != "project_settings.json" 
                           and os.path.isdir(os.path.join(directory, "extracted", f))]
        elastic_fwd_fields = np.zeros((len(extracted_folders)+1, len(landmarks), 3))

        for i, folder in enumerate(extracted_folders):
            elastic_fwd_files = glob.glob(os.path.join(directory, "extracted", folder, f"{folder}*elastic_fwd.nii.gz"))
            if not elastic_fwd_files or len(elastic_fwd_files) == 0:
                return Response({'message': 'No elastic inverse deformation field found in '+folder}, status=status.HTTP_400_BAD_REQUEST)
            if len(elastic_fwd_files) > 1:
                return Response({'message': 'More than one elastic inverse deformation field found in '+folder}, status=status.HTTP_400_BAD_REQUEST)
            elastic_fwd = ants.image_read(elastic_fwd_files[0])
            elastic_fwd = elastic_fwd.numpy()
            elastic_fwd[:, :, :, 0] = -elastic_fwd[:, :, :, 0]
            elastic_fwd[:, :, :, 1] = -elastic_fwd[:, :, :, 1]
            for j, index in enumerate(indices):
                elastic_fwd_fields[i, j, :] = elastic_fwd[index[0], index[1], index[2], :]

                            
        # Compute the variance of the deformation field vectors at each landmark
        field_variances = np.sum(np.var(elastic_fwd_fields, axis=0), axis=1)

        # Sort the landmarks by variance
        sorted_indices = np.argsort(field_variances)[::-1]
        sorted_landmarks = landmarks[sorted_indices]

        # Sort the field vectors by variance
        sorted_field_variances = field_variances[sorted_indices]

        # Sort the voxel indices by variance
        sorted_voxel_indices = indices[sorted_indices]

        # Sort the field vectors by variance
        sorted_field_vectors = elastic_fwd_fields[:, sorted_indices, :]

        # Select top 95% cumulative variation landmarks
        total_variance = np.sum(sorted_field_variances)
        cumulative_variance = np.cumsum(sorted_field_variances) / total_variance
        cutoff_idx = np.searchsorted(cumulative_variance, variance_percentage)
        filtered_landmarks = sorted_landmarks[:cutoff_idx+1]
        filtered_voxel_indices = sorted_voxel_indices[:cutoff_idx+1]        
        filtered_field_vectors = sorted_field_vectors[:, :cutoff_idx+1, :]
        filtered_field_variances = sorted_field_variances[:cutoff_idx+1]

        return filtered_landmarks, filtered_voxel_indices, filtered_field_vectors, filtered_field_variances

    def filter_landmarks_by_covariance_regions(self, landmarks, field_vectors, variances, covariance_threshold, max_landmark_region_radius, voxel_size, target):
        """
        Filter landmarks by covariance regions using KD-tree for efficient neighbor queries.

        Args:
            landmarks: Landmarks to filter
            field_vectors: Deformation field vectors at each landmark
            variances: Variance of each landmark
            covariance_threshold: Minimum covariance to consider landmarks related
            max_landmark_region_radius: Maximum radius of landmark region in mm
            voxel_size: Voxel size
        Returns:
            Filtered landmarks
        """
        landmarks = np.array(landmarks)
        field_vectors = np.array(field_vectors)
        variances = np.array(variances)

        n_landmarks = len(landmarks)
        remaining_mask = np.ones(n_landmarks, dtype=bool)
        final_landmarks = []

        # Build KD-tree once
        kd_tree = cKDTree(landmarks)

        n = 0
        while np.any(remaining_mask):
            # Find landmark with highest variance among remaining ones
            remaining_indices = np.where(remaining_mask)[0]
            remaining_variances = variances[remaining_mask]
            local_max_idx = np.argmax(remaining_variances)
            max_var_idx = remaining_indices[local_max_idx]

            current_landmark = landmarks[max_var_idx]
            current_field_vector = field_vectors[:, max_var_idx, :].reshape(-1)

            final_landmarks.append(current_landmark)

            # Query KD-tree for neighbors within radius
            region_indices = kd_tree.query_ball_point(current_landmark, r=max_landmark_region_radius)

            # Filter region_indices to only include remaining landmarks
            region_indices = [idx for idx in region_indices if remaining_mask[idx]]

            the_chosen_ones = [max_var_idx]  # Always include current landmark

            for idx in region_indices:
                if idx == max_var_idx:
                    continue
                vec2 = field_vectors[:, idx, :].reshape(-1)
                correlation = np.corrcoef(current_field_vector, vec2)[0, 1]
                if correlation >= covariance_threshold:
                    the_chosen_ones.append(idx)

            # Mark chosen landmarks as removed
            remaining_mask[the_chosen_ones] = False

            # Send progress update before each scan is processed
            channel_layer = get_channel_layer()
            if channel_layer is not None:
                progress = 2/4 + (((len(landmarks)-remaining_mask.sum())/len(landmarks))/4)
                async_to_sync(channel_layer.group_send)(
                    'progress_group',
                    {
                        'type': 'send_progress',
                        'progress': progress,
                        'scan_name': target,
                        'custom_message': f'Landmark # {n} identified.',
                        'total': 4,
                        'current': 4,
                    }
                )

            print("landmark # ", n, " set.")
            n += 1

        return np.array(final_landmarks)


    
    def displace_coordinate_by_closest_field_vector(self, coordinate, field_vector, voxel_size):
        """
        Displace a single coordinate by interpolating the displacement field at that location.
        
        Parameters:
            coordinate (array-like): Single coordinate (x, y, z) in physical space (mm)
            field_vector (numpy.ndarray): Displacement field with shape (X, Y, Z, 3), vectors in physical space (mm)
            voxel_size (float or array-like): Voxel size(s) in mm
            
        Returns:
            numpy.ndarray: The displaced coordinate in physical space (mm)
        """

        nearest_neighbor_mode = False

        # Convert coordinate to voxel space ONLY for looking up the right location in the field
        voxel_coord = np.array(coordinate) / np.array(voxel_size)
        
        # Try to get the 3 by 3 by 3 neighborhood of the coordinate
        if nearest_neighbor_mode:
            neighborhood_vectors = field_vector[np.floor(voxel_coord[0]).astype(int), np.floor(voxel_coord[1]).astype(int), np.floor(voxel_coord[2]).astype(int)]
        else:
            try:
                neighborhood_vectors = field_vector[np.floor(voxel_coord[0]-1).astype(int):np.floor(voxel_coord[0]+2).astype(int), np.floor(voxel_coord[1]-1).astype(int):np.floor(voxel_coord[1]+2).astype(int), np.floor(voxel_coord[2]-1).astype(int):np.floor(voxel_coord[2]+2).astype(int)]
                #print("neighborhood_vectors center: ", neighborhood_vectors[1, 1, 1, :])
            except:
                neighborhood_vectors = field_vector[np.floor(voxel_coord[0]).astype(int), np.floor(voxel_coord[1]).astype(int), np.floor(voxel_coord[2]).astype(int)]

        # If the neighborhood vectors are not 3 by 3 by 3, then we can just return the coordinate displaced by the closest vector
        if neighborhood_vectors.shape != (3, 3, 3, 3):
            displaced_coordinate = np.array(coordinate) + neighborhood_vectors            
            return displaced_coordinate

        # Generate the equivalent neighborhood containing voxel coordinates that correspond to the center of the voxels
        neighborhood_voxel_coords = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    neighborhood_voxel_coords[i, j, k, 0] = np.floor(voxel_coord[0]+(i-1)) + 0.5
                    neighborhood_voxel_coords[i, j, k, 1] = np.floor(voxel_coord[1]+(j-1)) + 0.5
                    neighborhood_voxel_coords[i, j, k, 2] = np.floor(voxel_coord[2]+(k-1)) + 0.5
            
        # Assuming that the neighborhood vectors correspond to the voxel coordinates, we can interpolate the displacement vector at the coordinate voxel_coord
        # Reshape neighborhood coordinates for interpolation
        x_coords = neighborhood_voxel_coords[:, :, :, 0]
        y_coords = neighborhood_voxel_coords[:, :, :, 1]
        z_coords = neighborhood_voxel_coords[:, :, :, 2]
        
        # Calculate the weights for interpolation based on distance
        x_weights = 1 - np.abs(voxel_coord[0] - x_coords)
        y_weights = 1 - np.abs(voxel_coord[1] - y_coords)
        z_weights = 1 - np.abs(voxel_coord[2] - z_coords)
        
        # Create the 3D weight matrix by element-wise multiplication
        weights = x_weights * y_weights * z_weights
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Apply weights to the displacement vectors
        interpolated_vector = np.sum(neighborhood_vectors * weights[..., np.newaxis], axis=(0, 1, 2))
        
        # Apply the interpolated displacement vector to the original coordinate
        displaced_coordinate = np.array(coordinate) + interpolated_vector   

        return displaced_coordinate

    
    def snap_landmarks_to_mesh(self, landmarks, vertices, faces, voxel_size, indices=None):
        print("Attempting: Snapping landmarks to closest mesh point")
        closest_points = alpaca.calculate_landmarks_closest_points(landmarks, vertices, faces)

        # Verify that shape of landmarks and closest points are the same
        if landmarks.shape != closest_points.shape:
            return Response({'message': 'Landmarks and closest points do not have the same shape'}, status=status.HTTP_400_BAD_REQUEST)

        # If the distance between any landmark and the closest point is greater than 1 voxel size, then skip this landmark altogether
        snapped_landmarks = []
        snapped_indices = []
        for i in range(len(landmarks)):
            if np.linalg.norm(landmarks[i] - closest_points[i]) > voxel_size:
                continue
            snapped_landmarks.append(closest_points[i])
            if indices is not None:
                snapped_indices.append(indices[i])

        snapped_landmarks = np.array(snapped_landmarks)
        if indices is not None:
            snapped_indices = np.array(snapped_indices)
            return snapped_landmarks, snapped_indices
        else:
            return snapped_landmarks

                

    def filter_landmarks(self, pseudo_landmarks, atlas_landmarks=None, covariance_threshold=0.7, max_landmarks_per_area=None):
        """
        Filter landmarks for a single subject when we don't have population variance data
        
        Args:
            pseudo_landmarks: Initial set of pseudo-landmarks
            atlas_landmarks: Optional atlas landmarks to guide selection
            covariance_threshold: Threshold for covariance/correlation
            max_landmarks_per_area: Maximum landmarks per local area
            
        Returns:
            Filtered landmarks
        """
        if len(pseudo_landmarks) == 0:
            return np.array([])
            
        # If we have atlas landmarks, use them as a guide
        if atlas_landmarks is not None:
            # Find correspondence between pseudo_landmarks and atlas_landmarks
            tree = cKDTree(pseudo_landmarks)
            _, indices = tree.query(atlas_landmarks)
            filtered_landmarks = pseudo_landmarks[indices]
            return filtered_landmarks
            
        # Without atlas landmarks, use spatial distribution
        # If max_landmarks_per_area not provided, use a default
        if max_landmarks_per_area is None:
            # Default: approximately landmarks every 1mm
            max_landmarks_per_area = 10
            
        # Use a similar approach to variance-based selection, but using spatial distance
        # as a proxy for correlation
        tree = cKDTree(pseudo_landmarks)
        
        final_landmarks = []
        remaining_indices = set(range(len(pseudo_landmarks)))
        
        # Simple distance-based greedy algorithm
        # Start with most central landmark
        center = np.mean(pseudo_landmarks, axis=0)
        distances_to_center = np.linalg.norm(pseudo_landmarks - center, axis=1)
        current_idx = np.argmin(distances_to_center)
        
        while remaining_indices:
            if current_idx not in remaining_indices:
                if not remaining_indices:
                    break
                current_idx = list(remaining_indices)[0]
                
            # Add current landmark
            final_landmarks.append(pseudo_landmarks[current_idx])
            
            # Find points in the neighborhood using covariance_threshold as a guide
            # Convert threshold to a distance measure (crude approximation)
            distance_threshold = -5 * np.log(covariance_threshold)
            neighbor_indices = tree.query_ball_point(pseudo_landmarks[current_idx], distance_threshold)
            
            # Limit by max_landmarks_per_area
            if len(neighbor_indices) > max_landmarks_per_area:
                # Keep only max_landmarks_per_area closest points
                distances = np.linalg.norm(
                    pseudo_landmarks[neighbor_indices] - pseudo_landmarks[current_idx], axis=1)
                closest_indices = np.argsort(distances)[:max_landmarks_per_area]
                neighbor_indices = [neighbor_indices[i] for i in closest_indices]
                
            # Remove from consideration
            for idx in neighbor_indices:
                if idx in remaining_indices:
                    remaining_indices.remove(idx)
                    
            # Find next landmark (farthest from all selected landmarks)
            if remaining_indices:
                remaining_landmarks = pseudo_landmarks[list(remaining_indices)]
                
                # For each remaining landmark, find min distance to any selected landmark
                min_distances = []
                for landmark in remaining_landmarks:
                    distances = [np.linalg.norm(landmark - fl) for fl in final_landmarks]
                    min_distances.append(min(distances))
                
                # Select the one with maximum minimum distance (farthest from all selected)
                next_local_idx = np.argmax(min_distances)
                next_idx = list(remaining_indices)[next_local_idx]
                current_idx = next_idx
        
        return np.array(final_landmarks)