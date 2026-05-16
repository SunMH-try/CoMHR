import numpy as np
import colorsys
import trimesh

from aitviewer.viewer import Viewer
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.material import Material
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.skeletons import Skeletons

class Renderer(Viewer):
    """
    A headless renderer implementation using AITViewer that matches the interface of renderer_pyrd.py
    """
    samples = 4
    window_type = "headless"

    def __init__(self, focal_length=600, center=[256, 256], img_w=512, img_h=512, faces=None, intri=None, extri=None, floor=False, same_mesh_color=False, use_interaction_color=False):
        """
        Initialize the renderer
        :param focal_length: Camera focal length
        :param center: Camera center point [x, y]
        :param img_w: Image width
        :param img_h: Image height
        :param faces: Mesh faces
        :param same_mesh_color: Whether to use same color for all meshes
        """
        from aitviewer.configuration import Configuration
        Configuration.instance._conf['auto_set_floor'] = floor

        super().__init__(size=(img_w, img_h))

        self.focal_length = focal_length
        self.camera_center = center
        self.faces = faces
        self.same_mesh_color = same_mesh_color
        self.use_interaction_color = use_interaction_color
        self.color = [(215 / 255, 160 / 255, 110 / 255), 
                      ( 190/ 255, 200 / 255,  122/ 255),
                      (96 / 255, 153 / 255, 246 / 255),
                    ( 135/ 255, 144 / 255,  207/ 255),]


        # self.color = [(170 * 1 / 255, 170 * 1 / 255, 220 * 1 / 255), (210 / 255, 166 / 255, 143 / 255)] 
        # self.material = Material(
        #     diffuse=0.5,
        #     ambient=0.35,
        #     specular=0.5,
        #     color=(0.9, 0.9, 0.9, 1.0)
        # )
        self.material = Material(
            diffuse=0.5,
            ambient=0.42,
            specular=0.5,
            color=(0.5, 0.5, 0.5, 1.0)
        )
        self.pc_color = (0.9, 0.9, 0.9, 1.0)
        self.skeleton_color = (1.0, 230 / 255, 1 / 255, 1.0)
        self.skeletons = {
            'coco17':[[0,1],[1,3],[0,2],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],[14,16],[11,12]],
            'halpe':[[0,1],[1,3],[0,2],[2,4],[5,18],[6,18],[18,17],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],[14,16],[11,19],[19,12],[18,19],[15,24],[15,20],[20,22],[16,25],[16,21],[21,23]],
            'lsp':[[0,1],[1,2],[2,3],[5,4],[4,3],[3,9],[9,8],[8,2],[6,7],[7,8],[9,10],[10,11]],
            'smpl':[[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[9,13],[9,14],[12,15],[13,16],[14,17],[16,18],[17,19],[18,20],[19,21],[20,22],[21,23]],
            }
        self.skeleton_joint_nums = {
            'coco17':17,
            'halpe':26,
            'lsp':14,
            'smpl':24,
        }
        
        if intri is not None:
            K = np.array(intri, dtype=np.float64)
        else:
            K = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype=np.float64)
        
        if extri is not None:
            Rt = np.array(extri, dtype=np.float64)
        else:
            Rt = np.array(
                [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]], dtype=np.float64)

        rot_z_180 = trimesh.transformations.rotation_matrix(np.radians(180), [0, 0, 1])
        Rt[:3, :3] = np.dot(rot_z_180[:3, :3], Rt[:3, :3])

        self.img_w = img_w
        self.img_h = img_h
        self.far = 1000
        self.near = 1

        self.camera = OpenCVCamera(
            K=K,
            Rt=Rt,
            cols=self.img_w,
            rows=self.img_h,
            near=self.near,
            far=self.far
        )
        self.scene.camera = self.camera

    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(255, 255, 255, 0), verts_color=None, transparent_background=False):
        """
        Render front view of the mesh
        :param verts: Mesh vertices
        :param bg_img_rgb: Background image
        :param bg_color: Background color
        :param verts_color: Vertex colors for each mesh
        :return: Rendered image
        """
        # Clear existing meshes and billboards
        nodes_to_remove = []
        for node in self.scene.nodes:
            if isinstance(node, Meshes) or isinstance(node, Billboard):
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            self.scene.remove(node)

        # Add background billboard if bg_img_rgb is not None
        if bg_img_rgb is not None:
            # Use Billboard.from_camera_and_distance for automatic scaling and positioning
            background_billboard = Billboard.from_camera_and_distance(
                camera=self.camera,
                distance=self.camera.far * 0.95,  # Place it just before the far plane
                cols=self.img_w,
                rows=self.img_h,
                textures=[bg_img_rgb], # textures expects a list of numpy arrays or PIL Images
            )
            self.scene.add(background_billboard)

        # Add meshes
        num_people = len(verts)
        # No rotation applied to verts in this renderer, handled by camera setup if needed.

        for n in range(num_people):
            vertices = verts[n]

            rot_z_180 = trimesh.transformations.rotation_matrix(np.radians(180), [0, 0, 1])
            vertices = np.matmul(vertices, rot_z_180[:3, :3]) + rot_z_180[:3, 3]

            # Set mesh color
            if self.use_interaction_color:
                mesh_color = list(self.color[n%len(self.color)]) + [1.0]  # Add alpha channel
            elif self.same_mesh_color:
                mesh_color = [0.7, 0.7, 0.7, 1.0] 
            else:
                mesh_color = list(self.color[n%len(self.color)]) + [1.0]
                #mesh_color = list(colorsys.hsv_to_rgb(float(n) / num_people, 0.5, 1.0)) + [1.0]  # Add alpha channel

            # Create mesh
            mesh = Meshes(
                vertices=vertices[None],  # Add batch dimension
                faces=self.faces,
                is_selectable=False,
                gui_affine=False,
                color=mesh_color,
                name=f"Mesh_{n}"
            )

            mesh.material.diffuse = self.material.diffuse
            mesh.material.ambient = self.material.ambient
            mesh.material.specular = self.material.specular

            # Set vertex colors if provided
            if verts_color is not None:
                mesh.vertex_colors = verts_color[n]

            self.scene.add(mesh)

        # Render frame
        self._init_scene()
        self.render(0, 0, export=True, transparent_background=transparent_background)
        color_rgba = self.get_current_frame_as_image(alpha=True)
        color_rgba = np.array(color_rgba)

        if transparent_background:
            return color_rgba
        elif bg_img_rgb is not None or not transparent_background:
            return color_rgba[:, :, :3]

    def render_side_view(self, verts, bg_img_rgb=None, verts_color=None, transparent_background=False):
        centroid = verts.mean(axis=(0, 1))
        centroid[:2] = 0
        
        # Rotate 90 degrees around Y axis
        aroundy = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        
        verts_rotated = np.matmul((verts - centroid), aroundy) + centroid
        # verts_rotated += np.array([-1, 0, 4.5])

        return self.render_front_view(
            verts_rotated,
            bg_img_rgb=bg_img_rgb,
            verts_color=verts_color,
            transparent_background=transparent_background
        )


    def render_back_view(self, verts, bg_img_rgb=None, verts_color=None, transparent_background=False):
        centroid = verts.mean(axis=(0, 1))
        centroid[:2] = 0
        
        # Rotate 180 degrees around Y axis
        aroundy = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        
        verts_rotated = np.matmul((verts - centroid), aroundy) + centroid

        return self.render_front_view(
            verts_rotated,
            bg_img_rgb=bg_img_rgb,
            verts_color=verts_color,
            transparent_background=transparent_background
        )


    def render_backside_view(self, verts, bg_img_rgb=None, verts_color=None, transparent_background=False):
        centroid = verts.mean(axis=(0, 1))
        centroid[:2] = 0
        
        # First 180 degrees around Y
        aroundy = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        verts_rotated = np.matmul((verts - centroid), aroundy) + centroid
        
        # Then further 90 degrees
        centroid = verts_rotated.mean(axis=(0, 1))
        centroid[:2] = 0
        aroundy = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        verts_rotated = np.matmul((verts_rotated - centroid), aroundy) + centroid

        return self.render_front_view(
            verts_rotated,
            bg_img_rgb=bg_img_rgb,
            verts_color=verts_color,
            transparent_background=transparent_background
        )


    def render_top_view(self, verts, bg_img_rgb=None, verts_color=None, transparent_background=False):
        centroid = verts.mean(axis=(0, 1))
        centroid[:2] = 0
        
        # Rotate 90 degrees around X axis
        aroundx = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ])
        
        verts_rotated = np.matmul((verts - centroid), aroundx) + centroid
        # verts_rotated += np.array([-50, 10, 200.0])

        return self.render_front_view(
            verts_rotated,
            bg_img_rgb=bg_img_rgb,
            verts_color=verts_color,
            transparent_background=transparent_background
        )

    def render_pc(self, verts, verts_color=None, transparent_background=False):
        """
        Render point cloud of the mesh
        :param verts: Mesh vertices
        :param verts_color: Vertex colors for each mesh
        :return: Rendered image
        """
        num_people = len(verts)
        if verts_color is None:
            verts_color = [np.tile(self.pc_color, (verts[n].shape[0], 1)) for n in range(num_people)]
            verts_color = np.array(verts_color)
        
        if self.use_interaction_color:
            verts_color = [np.tile(self.color[n], (verts[n].shape[0], 1)) for n in range(num_people)]
            verts_color = np.array(verts_color)
            verts_color = np.concatenate([verts_color, np.ones((*verts_color.shape[:-1], 1))], axis=-1)
        
        for n in range(num_people):
            vertices = verts[n]
            rot_z_180 = trimesh.transformations.rotation_matrix(np.radians(180), [0, 0, 1])
            vertices = np.matmul(vertices, rot_z_180[:3, :3]) + rot_z_180[:3, 3]
            ptc = PointClouds(vertices.reshape(1, -1, 3), colors=verts_color[n].reshape(1, -1, 4))
            self.scene.add(ptc)

        self._init_scene()
        self.render(0, 0, export=True, transparent_background=transparent_background)
        color_rgba = self.get_current_frame_as_image(alpha=True)
        color_rgba = np.array(color_rgba)

        if transparent_background:
            return color_rgba
        else:
            return color_rgba[:, :, :3]

    def render_skeleton(self, joints, format='coco17', transparent_background=False):

        skeleton = self.skeletons[format]
        assert joints.shape[1] == self.skeleton_joint_nums[format], 'Skeleton format and joints shape mismatch'
        assert len(joints.shape) == 3, 'joints must be 3D, [num_agents, num_joints, 3]'
        
        num_agents = joints.shape[0]
        for n in range(num_agents):
            joints_n = joints[n].reshape(1, -1, 3) # add frame dimension
            skeleton_node = Skeletons(joints_n, 
                                     skeleton, 
                                     gui_affine=False,
                                     color=self.skeleton_color,
                                     name=f"Skeleton_{n}")
            self.scene.add(skeleton_node)
        
        self._init_scene()
        self.render(0, 0, export=True, transparent_background=transparent_background)
        color_rgba = self.get_current_frame_as_image(alpha=True)
        color_rgba = np.array(color_rgba)
        
        if transparent_background:
            return color_rgba
        else:
            return color_rgba[:, :, :3]
        
        return color_rgba

    def delete(self):
        """
        Clean up resources and release all contexts.
        The `window.close()` and `window.destroy()` methods typically handle the release
        of the associated ModernGL context (`ctx`).
        """
        if hasattr(self, 'window'):
            self.window.close()
            self.window.destroy()
        
        self.ctx.release()
        del self.ctx