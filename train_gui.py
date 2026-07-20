7  #
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from pickle import FALSE
import time
import math
import random
import bisect
import torch
from random import randint
from event_utils.event_loader import EventLoader
from event_utils.event_accumulator import accumulate_events
from event_utils.event_loss import compute_event_loss
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training_report
import math
from utils.gui_utils import orbit_camera, OrbitCamera
import numpy as np
import dearpygui.dearpygui as dpg


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3], dtype=torch.float32).cuda()


class GUI:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations

        self.tb_writer = prepare_output_and_logger(dataset)
        self.gaussians = GaussianModel(dataset.sh_degree)
        self.deform = DeformModel(is_blender=dataset.is_blender, is_6dof=dataset.is_6dof)
        self.deform.train_setting(opt)

        self.scene = Scene(dataset, self.gaussians)
        self.gaussians.training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.train_time_ms = 0.0  
        self.iteration = 1

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_iteration = 0
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

        # For UI
        self.visualization_mode = 'RGB'

        self.gui = args.gui # enable gui
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)

        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.training = False

        self.USE_EVENTS = True
        self.N_SUB = 3
        self.LAMBDA_EVENT = 0.05
        self.CONTRAST_THRESHOLD = 0.2
        self.EVENT_START_ITER = 3000
        self.EVENT_DETACH_COLOR = True
        self.LAMBDA_TLI = 1e-5
        self.TLI_DELTA_T = 0.05
        self.TLI_K_CONN = 10
        self.USE_MARKOV = self.dataset.is_blender
        self.MARKOV_ALPHA_POS = 0.5      
        self.MARKOV_WINDOW_SIZE = 2      
        self.MARKOV_START_ITER = 3000
        self.LAMBDA_MARKOV_SMOOTH = 1e-2
        self.EVENT_OBS_WINDOW = 0.01    

        if self.USE_EVENTS:
            dense_json = os.path.join(dataset.source_path, "move_cam_event/transforms_train.json")
            move_cam_ev = os.path.join(dataset.source_path, "move_cam_event/events/events.npz")
            static_ev   = os.path.join(dataset.source_path, "events/events.npz")

            if os.path.exists(dense_json) and os.path.exists(move_cam_ev):
                TOTAL_DURATION_NS = int(__import__('numpy').load(move_cam_ev)['t'].max())
                self.event_loader = EventLoader(npz_path=move_cam_ev, total_duration_ns=TOTAL_DURATION_NS)
                ref_cam = self.scene.getTrainCameras()[0]
                self.dense_pose_list = self._load_dense_poses(
                    dense_json, ref_width=ref_cam.image_width, ref_height=ref_cam.image_height,
                )
                self.dense_times = [p['fid'] for p in self.dense_pose_list]
                self.use_dense_pose = True
                print("[Event] move_cam_event + dense_pose")
            elif os.path.exists(static_ev):
                data = __import__('numpy').load(static_ev)
                TOTAL_DURATION_NS = int(data['t'].max())
                self.event_loader = EventLoader(npz_path=static_ev, total_duration_ns=TOTAL_DURATION_NS)
                self.dense_pose_list = None
                self.use_dense_pose = False
                print("[Event] static_cam")
            else:
                print("[Event] no event data")
                self.USE_EVENTS = False

        if hasattr(self.deform.deform, 'use_markov'):
            self.deform.deform.use_markov = self.USE_MARKOV
            self.deform.deform.markov_alpha_pos = self.MARKOV_ALPHA_POS
            self.deform.deform.markov_window_size = self.MARKOV_WINDOW_SIZE

        self.train_camera_dict = {}
        for cam in self.scene.getTrainCameras():
            t = round(cam.fid.item(), 6)
            self.train_camera_dict[t] = cam
        self.train_times_sorted = sorted(self.train_camera_dict.keys())

        # static_cam 
        if self.USE_EVENTS and not getattr(self, 'use_dense_pose', True):
            self.static_cam = self.train_camera_dict[self.train_times_sorted[0]]

        self.event_obs_cache = {}
        if self.USE_MARKOV and self.USE_EVENTS:
            for t_key, cam in self.train_camera_dict.items():
                t_lo = max(0.0, t_key - self.EVENT_OBS_WINDOW)
                t_hi = min(1.0, t_key + self.EVENT_OBS_WINDOW)
                ev = self.event_loader.get_events_between(t_lo, t_hi)
                self.event_obs_cache[t_key] = accumulate_events(
                    ev, cam.image_height, cam.image_width, device='cuda')
            mb = sum(v.element_size() * v.numel() for v in self.event_obs_cache.values()) / 1e6
            print(f"[Markov] event_obs 预累积完成：{len(self.event_obs_cache)} 帧，GPU 占用 ≈ {mb:.1f} MB")

        if self.USE_MARKOV and self.USE_EVENTS and getattr(self, 'use_dense_pose', False):
            self.event_proj_by_time = {}
            for p in self.dense_pose_list:
                mc = MiniCam(p['c2w'], p['width'], p['height'], p['fovy'], p['fovx'],
                             0.01, 100.0, p['fid'])
                self.event_proj_by_time[round(p['fid'], 6)] = mc.full_proj_transform
            ref = self.scene.getTrainCameras()[0]
            ref_h, ref_w = ref.image_height, ref.image_width

            def _event_provider(t, _self=self, _h=ref_h, _w=ref_w):
                # 
                i = bisect.bisect_left(_self.dense_times, t)
                cands = [j for j in (i - 1, i) if 0 <= j < len(_self.dense_times)]
                t_d = min(cands, key=lambda j: abs(_self.dense_times[j] - t))
                proj = _self.event_proj_by_time[round(_self.dense_times[t_d], 6)]
                # 
                obs = _self.event_obs_cache.get(round(t, 6))
                if obs is None:
                    t_lo = max(0.0, t - _self.EVENT_OBS_WINDOW)
                    t_hi = min(1.0, t + _self.EVENT_OBS_WINDOW)
                    obs = accumulate_events(_self.event_loader.get_events_between(t_lo, t_hi),
                                            _h, _w, device='cuda')
                return obs, proj
            self._event_provider = _event_provider
            if hasattr(self.deform.deform, 'event_provider'):
                self.deform.deform.event_provider = _event_provider
            print(f"[EventDeform]")

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()
        
    def _load_dense_poses(self, json_path, ref_width, ref_height):
        import json
        with open(json_path) as f:
            data = json.load(f)
        angle_x = data['camera_angle_x']
        focal = ref_width / (2 * math.tan(angle_x / 2))
        fovx = angle_x
        fovy = 2 * math.atan(ref_height / (2 * focal))
        pose_list = []
        for fr in data['frames']:
            pose_list.append({
                'c2w':   np.array(fr['transform_matrix'], dtype=np.float32),
                'fid':   fr.get('time', 0.0),
                'fovx':  fovx,
                'fovy':  fovy,
                'width': ref_width,
                'height': ref_height,
            })
        pose_list.sort(key=lambda x: x['fid'])
        print(f"[DensePose]")
        return pose_list

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Visualization: ")

                    def callback_vismode(sender, app_data, user_data):
                        self.visualization_mode = user_data
                        if user_data == 'Node':
                            self.node_vis_fea = True if not hasattr(self, 'node_vis_fea') else not self.node_vis_fea
                            print("Visualize node features" if self.node_vis_fea else "Visualize node importance")
                            if self.node_vis_fea or True:
                                from motion import visualize_featuremap
                                if True:  #self.renderer.gaussians.motion_model.soft_edge:
                                    if hasattr(self.renderer.gaussians.motion_model, 'nodes_fea'):
                                        node_rgb = visualize_featuremap(self.renderer.gaussians.motion_model.nodes_fea.detach().cpu().numpy())
                                        self.node_rgb = torch.from_numpy(node_rgb).cuda()
                                    else:
                                        self.node_rgb = None
                                else:
                                    self.node_rgb = None
                            else:
                                node_imp = self.renderer.gaussians.motion_model.cal_node_importance(x=self.renderer.gaussians.get_xyz)
                                node_imp = (node_imp - node_imp.min()) / (node_imp.max() - node_imp.min())
                                node_rgb = torch.zeros([node_imp.shape[0], 3], dtype=torch.float32).cuda()
                                node_rgb[..., 0] = node_imp
                                node_rgb[..., -1] = 1 - node_imp
                                self.node_rgb = node_rgb

                    dpg.add_button(
                        label="RGB",
                        tag="_button_vis_rgb",
                        callback=callback_vismode,
                        user_data='RGB',
                    )
                    dpg.bind_item_theme("_button_vis_rgb", theme_button)

                    dpg.add_button(
                        label="UV_COOR",
                        tag="_button_vis_uv",
                        callback=callback_vismode,
                        user_data='UV_COOR',
                    )
                    dpg.bind_item_theme("_button_vis_uv", theme_button)
                    dpg.add_button(
                        label="MotionMask",
                        tag="_button_vis_motion_mask",
                        callback=callback_vismode,
                        user_data='MotionMask',
                    )
                    dpg.bind_item_theme("_button_vis_motion_mask", theme_button)

                    dpg.add_button(
                        label="Node",
                        tag="_button_vis_node",
                        callback=callback_vismode,
                        user_data='Node',
                    )
                    dpg.bind_item_theme("_button_vis_node", theme_button)

                    def callback_use_const_var(sender, app_data):
                        self.use_const_var = not self.use_const_var
                    dpg.add_button(
                        label="Const Var",
                        tag="_button_const_var",
                        callback=callback_use_const_var
                    )
                    dpg.bind_item_theme("_button_const_var", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Scale Const: ")
                    def callback_vis_scale_const(sender):
                        self.vis_scale_const = 10 ** dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Log vis_scale_const (For debugging)",
                        default_value=-3,
                        max_value=-.5,
                        min_value=-5,
                        callback=callback_vis_scale_const,
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Temporal Speed: ")
                    self.video_speed = 1.
                    def callback_speed_control(sender):
                        self.video_speed = dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="Play speed",
                        default_value=1.,
                        max_value=2.,
                        min_value=0.0,
                        callback=callback_speed_control,
                    )
                
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_button(
                        label="pcl",
                        tag="_button_save_pcl",
                        callback=callback_save,
                        user_data='pcl',
                    )
                    dpg.bind_item_theme("_button_save_pcl", theme_button)

                    def call_back_save_train(sender, app_data, user_data):
                        self.render_all_train_data()
                    dpg.add_button(
                        label="save_train",
                        tag="_button_save_train",
                        callback=call_back_save_train,
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            # self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_psnr")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("render", "depth"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True
                
        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Deformable-Gaussian",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=5000):
        if iters > 0:
            for i in tqdm.trange(iters):
                self.train_step()
    

    def train_step(self):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.do_shs_python, self.pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                            0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, self.dataset.source_path)
                if do_training and ((self.iteration < int(self.opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        self.iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()

        total_frame = len(self.viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        # print(f"iter {self.iteration}, cam fid: {viewpoint_cam.fid.item()}, image: {viewpoint_cam.image_name}")

        markov_active = False  
        if self.iteration < self.opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = self.gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
            markov_active = self.USE_MARKOV and self.iteration >= self.MARKOV_START_ITER
            if hasattr(self.deform.deform, 'use_markov'):
                self.deform.deform.use_markov = markov_active

            event_obs, event_proj = None, None
            if markov_active and self.USE_EVENTS:
                cur_t = round(float(fid.item()), 6)
                event_obs = self.event_obs_cache.get(cur_t)
                if event_obs is None:
                    t_lo = max(0.0, cur_t - self.EVENT_OBS_WINDOW)
                    t_hi = min(1.0, cur_t + self.EVENT_OBS_WINDOW)
                    events_win = self.event_loader.get_events_between(t_lo, t_hi)
                    event_obs = accumulate_events(
                        events_win, viewpoint_cam.image_height, viewpoint_cam.image_width, device='cuda')
                event_proj = viewpoint_cam.full_proj_transform
            d_xyz, d_rotation, d_scaling = self.deform.step(
                self.gaussians.get_xyz.detach(), time_input + ast_noise,
                event_obs=event_obs, event_proj=event_proj)

        # Render
        render_pkg_re = render(viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling, self.dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        # depth = render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # ===== TLI（Temporal Local Isometry）=====
        if self.LAMBDA_TLI > 0 and self.iteration >= self.opt.warm_up:
            loss = loss + self.LAMBDA_TLI * self.deform.tli_loss(
                time_input, delta_t=self.TLI_DELTA_T, K_conn=self.TLI_K_CONN)

        # ===== Markov Smooth=====
        if self.LAMBDA_MARKOV_SMOOTH > 0 and markov_active:
            loss = loss + self.LAMBDA_MARKOV_SMOOTH * self.deform.markov_smooth_loss()

        # ===== Event loss =====
        if self.USE_EVENTS and self.iteration > self.EVENT_START_ITER and self.iteration % 5 == 0:
            t_cur = round(fid.item(), 6)
            idx = self.train_times_sorted.index(t_cur) if t_cur in self.train_times_sorted else -1
            t_next = None
            if idx >= 0:
                if idx < len(self.train_times_sorted) - 1:
                    t_next = self.train_times_sorted[idx + 1]
                elif idx > 0:
                    t_next = self.train_times_sorted[idx - 1]

            if t_next is not None:
                t_a = min(t_cur, t_next)
                t_b = max(t_cur, t_next)
                N_gs = self.gaussians.get_xyz.shape[0]

                if self.use_dense_pose:
                    DENSE_STEP = 10 if self.dataset.is_blender else 1
                    lo = bisect.bisect_left(self.dense_times, t_a)
                    hi = bisect.bisect_right(self.dense_times, t_b) - 1
                    if hi - lo >= DENSE_STEP:
                        start = random.randint(lo, hi - DENSE_STEP)
                        p_a = self.dense_pose_list[start]
                        p_b = self.dense_pose_list[start + DENSE_STEP]

                        cam_a = MiniCam(p_a['c2w'], p_a['width'], p_a['height'],
                                        p_a['fovy'], p_a['fovx'], 0.01, 100.0, p_a['fid'])
                        cam_b = MiniCam(p_b['c2w'], p_b['width'], p_b['height'],
                                        p_b['fovy'], p_b['fovx'], 0.01, 100.0, p_b['fid'])

                        seg_events = self.event_loader.get_events_between(p_a['fid'], p_b['fid'])
                        emap = accumulate_events(seg_events, p_a['height'], p_a['width'], device='cuda')

                        fid_a = torch.tensor([[p_a['fid']]], device='cuda').expand(N_gs, -1)
                        fid_b = torch.tensor([[p_b['fid']]], device='cuda').expand(N_gs, -1)
                        if self.iteration >= self.opt.warm_up:
                            d_a = self.deform.step(self.gaussians.get_xyz.detach(), fid_a)
                            d_b = self.deform.step(self.gaussians.get_xyz.detach(), fid_b)
                        else:
                            d_a = d_b = (0.0, 0.0, 0.0)

                        pkg_a = render(cam_a, self.gaussians, self.pipe, self.background,
                                       *d_a, self.dataset.is_6dof,
                                       detach_color=self.EVENT_DETACH_COLOR)
                        pkg_b = render(cam_b, self.gaussians, self.pipe, self.background,
                                       *d_b, self.dataset.is_6dof,
                                       detach_color=self.EVENT_DETACH_COLOR)
                        if pkg_a["visibility_filter"].sum() == 0 or pkg_b["visibility_filter"].sum() == 0:
                            pass  # 跳过，避免梯度 shape 错误
                        else:
                            event_loss_val = compute_event_loss(
                                pkg_a["render"], pkg_b["render"], emap, self.CONTRAST_THRESHOLD)
                            loss = loss + self.LAMBDA_EVENT * event_loss_val
                else:
                    # ── static_cam ──
                    from event_utils.event_accumulator import get_subsegment_event_maps
                    H_img = self.static_cam.image_height
                    W_img = self.static_cam.image_width
                    event_maps, sub_timestamps = get_subsegment_event_maps(
                        t_a, t_b, self.N_SUB, self.event_loader, H_img, W_img, device='cuda')
                    sub_loss = torch.tensor(0.0, device='cuda')
                    valid_count = 0
                    for emap, (tau_s, tau_e) in zip(event_maps, sub_timestamps):
                        if self.iteration >= self.opt.warm_up:
                            fid_s = torch.tensor([[tau_s]], device='cuda').expand(N_gs, -1)
                            fid_e = torch.tensor([[tau_e]], device='cuda').expand(N_gs, -1)
                            d_s = self.deform.step(self.gaussians.get_xyz.detach(), fid_s)
                            d_e = self.deform.step(self.gaussians.get_xyz.detach(), fid_e)
                        else:
                            d_s = d_e = (0.0, 0.0, 0.0)
                        pkg_s = render(self.static_cam, self.gaussians, self.pipe, self.background,
                                       *d_s, self.dataset.is_6dof,
                                       detach_color=self.EVENT_DETACH_COLOR)
                        pkg_e = render(self.static_cam, self.gaussians, self.pipe, self.background,
                                       *d_e, self.dataset.is_6dof,
                                       detach_color=self.EVENT_DETACH_COLOR)
                        if pkg_s["visibility_filter"].sum() == 0 or pkg_e["visibility_filter"].sum() == 0:
                            continue
                        sub_loss = sub_loss + compute_event_loss(
                            pkg_s["render"], pkg_e["render"], emap, self.CONTRAST_THRESHOLD)
                        valid_count += 1
                    if valid_count > 0:
                        loss = loss + self.LAMBDA_EVENT * (sub_loss / valid_count)

        loss.backward()

        self.iter_end.record()

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            # Keep track of max radii in image-space for pruning
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            iter_ms = self.iter_start.elapsed_time(self.iter_end)
            self.train_time_ms += iter_ms
            if self.iteration == self.opt.iterations:
                print(f"\n[纯训练时间] {self.train_time_ms/1000:.1f}s "
                      f"({self.train_time_ms/1000/60:.2f}min) — 累加每步 GPU 时间, 不含 test/eval")
            cur_psnr = training_report(self.tb_writer, self.iteration, Ll1, loss, l1_loss, iter_ms, self.testing_iterations, self.scene, render, (self.pipe, self.background), self.deform, self.dataset.load2gpu_on_the_fly, self.dataset.is_6dof)
            if self.iteration in self.testing_iterations:
                if cur_psnr.item() > self.best_psnr:
                    self.best_psnr = cur_psnr.item()
                    self.best_iteration = self.iteration

            if self.iteration in self.saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(args.model_path, self.iteration)

            # Densification
            if self.iteration < self.opt.densify_until_iter:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.05, self.scene.cameras_extent, size_threshold)

                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
                self.deform.update_learning_rate(self.iteration)

        if self.gui:
            dpg.set_value(
                "_log_train_psnr",
                "Best PSNR = {} in Iteration {}".format(self.best_psnr, self.best_iteration)
            )
        else:
            print("Best PSNR = {} in Iteration {}".format(self.best_psnr, self.best_iteration))
        self.iteration += 1

        if self.gui:
            dpg.set_value(
                "_log_train_log",
                f"step = {self.iteration: 5d} loss = {loss.item():.4f}",
            )
    
    @torch.no_grad()
    def test_step(self):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if not hasattr(self, 't0'):
            self.t0 = time.time()
            self.fps_of_fid = 10

        cur_cam = MiniCam(
            self.cam.pose,
            self.W,
            self.H,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
            fid=torch.remainder(torch.tensor((time.time()-self.t0) * self.fps_of_fid).float().cuda() / len(self.scene.getTrainCameras()), 1.)
        )
        fid = cur_cam.fid

        if self.iteration < self.opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = self.gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            d_xyz, d_rotation, d_scaling = self.deform.step(self.gaussians.get_xyz.detach(), time_input)
        
        out = render(viewpoint_camera=cur_cam, pc=self.gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, is_6dof=self.dataset.is_6dof)

        buffer_image = out[self.mode]  # [3, H, W]

        if self.mode in ['depth', 'alpha']:
            buffer_image = buffer_image.repeat(3, 1, 1)
            if self.mode == 'depth':
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        self.need_update = True

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS FID: {fid.item()})")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    # no gui mode
    def train(self, iters=5000):
        if iters > 0:
            for i in tqdm.trange(iters):
                self.train_step()        

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--gui', action='store_false', help="start a GUI")
    parser.add_argument('--W', type=int, default=346, help="GUI width")
    parser.add_argument('--H', type=int, default=260, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gui = GUI(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),testing_iterations=args.test_iterations, saving_iterations=args.save_iterations)

    if args.gui:
        gui.render()
    else:
        gui.train(args.iterations)
    
    # All done
    print("\nTraining complete.")
