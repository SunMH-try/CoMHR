# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2


class Renderer(object):
    def __init__(self, focal_length=600, center=[256, 256], img_w=512, img_h=512, faces=None,
                 same_mesh_color=True):
        # os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        self.camera_center = [center[0], center[1]]
        self.focal_length = focal_length
        self.faces = faces
        self.same_mesh_color = same_mesh_color
    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(0, 0, 0, 0)):
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=np.eye(4))

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # multiple person
        num_people = len(verts)
        # for every person in the scene
        for n in range(num_people):
            mesh = trimesh.Trimesh(verts[n], self.faces)
            mesh.apply_transform(rot)
            if self.same_mesh_color:
                mesh_color = colorsys.hsv_to_rgb(0, 0, 0.8)
            else:
                mesh_color = colorsys.hsv_to_rgb(float(n) / num_people, 0.5, 1.0)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
            scene.add(mesh, 'mesh')

        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[:, :, :3]
        if bg_img_rgb is None:
            return color_rgb
        else:
            mask = depth_map > 0
            bg_img_rgb[mask] = color_rgb[mask]
            return bg_img_rgb

    def render_side_view(self, verts, bg_img_rgb=None):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid

        # 使用独立背景
        if bg_img_rgb is not None:
            bg = bg_img_rgb.copy()
        else:
            bg = None
        side_view = self.render_front_view(pred_vert_arr_side, bg_img_rgb=bg)
        return side_view

    def render_top_view(self, verts, bg_img_rgb=None):
        centroid = verts.mean(axis=(0, 1))
        centroid[:2] = 0
        aroundx = cv2.Rodrigues(np.array([-np.radians(90.), 0, 0]))[0][np.newaxis, ...]
        pred_vert_arr_top = np.matmul((verts - centroid), aroundx) + centroid

        if bg_img_rgb is not None:
            bg = bg_img_rgb.copy()
        else:
            bg = None
        top_view = self.render_front_view(pred_vert_arr_top, bg_img_rgb=bg)
        return top_view


    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()

# Copyright (C) 2022. Huawei Technologies Co., Ltd.


#convertimage用
# import os
# import trimesh
# import pyrender
# import numpy as np
# import colorsys
# import cv2


# class Renderer(object):
#     def __init__(self, focal_length=600, center=[256, 256], img_w=512, img_h=512,
#                  faces=None, same_mesh_color=True,
#                  id_list=None, color_map=None):
#         self.renderer = pyrender.OffscreenRenderer(
#             viewport_width=img_w,
#             viewport_height=img_h,
#             point_size=1.0
#         )
#         self.camera_center = [center[0], center[1]]
#         self.focal_length = focal_length
#         self.faces = faces
#         self.same_mesh_color = same_mesh_color

#         # 🔥 新增
#         self.id_list = id_list
#         self.color_map = color_map

#     def _get_mesh_color(self, idx, num_people):
#         # 根据 ID 选择颜色
#         if self.id_list is not None:
#             pid = self.id_list[idx]
#         else:
#             pid = idx

#         # 优先使用 color_map
#         if self.color_map is not None and pid in self.color_map:
#             return self.color_map[pid]

#         # 否则自动颜色（稳定一致）
#         hue = float(pid % 50) / 50.0
#         return colorsys.hsv_to_rgb(hue, 0.6, 1.0)

#     def render_front_view(self, verts, bg_img_rgb=None, bg_color=(0, 0, 0, 0)):
#         scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)

#         camera = pyrender.camera.IntrinsicsCamera(
#             fx=self.focal_length, fy=self.focal_length,
#             cx=self.camera_center[0], cy=self.camera_center[1]
#         )
#         scene.add(camera, pose=np.eye(4))

#         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
#         lp = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
#         scene.add(light, pose=lp)
#         lp = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
#         scene.add(light, pose=lp)

#         rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
#         num_people = len(verts)

#         for n in range(num_people):
#             mesh = trimesh.Trimesh(verts[n], self.faces)
#             mesh.apply_transform(rot)

#             mesh_color = self._get_mesh_color(n, num_people)

#             material = pyrender.MetallicRoughnessMaterial(
#                 metallicFactor=0.2,
#                 alphaMode='OPAQUE',
#                 baseColorFactor=mesh_color
#             )
#             mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
#             scene.add(mesh, 'mesh')

#         color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
#         color_rgb = color_rgba[:, :, :3]

#         if bg_img_rgb is None:
#             return color_rgb

#         mask = depth_map > 0
#         bg_img_rgb[mask] = color_rgb[mask]
#         return bg_img_rgb

#     def render_side_view(self, verts, bg_img_rgb=None):
#         centroid = verts.mean(axis=(0, 1))
#         centroid[:2] = 0
#         around_y = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]
#         verts_side = np.matmul((verts - centroid), around_y) + centroid
#         return self.render_front_view(verts_side, bg_img_rgb.copy() if bg_img_rgb is not None else None)

#     def render_top_view(self, verts, bg_img_rgb=None):
#         centroid = verts.mean(axis=(0, 1))
#         centroid[:2] = 0
#         around_x = cv2.Rodrigues(np.array([-np.radians(90.), 0, 0]))[0][np.newaxis, ...]
#         verts_top = np.matmul((verts - centroid), around_x) + centroid
#         return self.render_front_view(verts_top, bg_img_rgb.copy() if bg_img_rgb is not None else None)

#     def delete(self):
#         self.renderer.delete()
