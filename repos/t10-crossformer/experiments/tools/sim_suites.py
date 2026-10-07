"""SimplerEnv visual-matching suites, copied from the official scripts at SimplerEnv@eddb569
(scripts/octo_bridge.sh, *_pick_coke_can_visual_matching.sh, *_move_near_visual_matching.sh,
*_drawer_visual_matching.sh). Each job is the argument list of one simpler_env/main_inference.py call.

Episodes per suite: bridge 4x24 = 96, coke_can 3 orientations x 25 positions x 4 URDFs = 300,
move_near 60 x 4 URDFs = 240, drawer 6 tasks x 9 poses x 4 URDFs = 216 (ray-traced, slow).
"""

URDFS = ["None", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2",
         "recolor_cabinet_visual_matching_1"]


def _rot(yaw):
    return ["--robot-init-rot-quat-center", "0", "0", "0", "1",
            "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", str(yaw), str(yaw), "1"]


def _xy(x, y):
    return ["--robot-init-x", str(x), str(x), "1", "--robot-init-y", str(y), str(y), "1"]


def bridge(asset_dir):
    jobs = []
    common = ["--policy-setup", "widowx_bridge", "--control-freq", "5", "--sim-freq", "500",
              "--obj-variation-mode", "episode", "--obj-episode-range", "0", "24"] + _rot(0)
    for env in ["StackGreenCubeOnYellowCubeBakedTexInScene-v0", "PutCarrotOnPlateInScene-v0",
                "PutSpoonOnTableClothInScene-v0"]:
        jobs.append((env, ["--robot", "widowx", "--env-name", env, "--scene-name", "bridge_table_1_v1",
                           "--max-episode-steps", "60",
                           "--rgb-overlay-path", f"{asset_dir}/real_inpainting/bridge_real_eval_1.png"]
                     + _xy(0.147, 0.028) + common))
    env = "PutEggplantInBasketScene-v0"
    jobs.append((env, ["--robot", "widowx_sink_camera_setup", "--env-name", env, "--scene-name", "bridge_table_1_v2",
                       "--max-episode-steps", "120",
                       "--rgb-overlay-path", f"{asset_dir}/real_inpainting/bridge_sink.png"]
                 + _xy(0.127, 0.06) + common))
    return jobs


def coke_can(asset_dir):
    jobs = []
    for urdf in URDFS:
        for opt, tag in [("lr_switch=True", "horizontal"), ("upright=True", "standing"),
                         ("laid_vertically=True", "vertical")]:
            jobs.append((f"pick_coke_can_{tag}", [
                "--robot", "google_robot_static", "--policy-setup", "google_robot",
                "--control-freq", "3", "--sim-freq", "501", "--max-episode-steps", "80",
                "--env-name", "GraspSingleOpenedCokeCanInScene-v0", "--scene-name", "google_pick_coke_can_1_v4",
                "--rgb-overlay-path", f"{asset_dir}/real_inpainting/google_coke_can_real_eval_1.png",
                "--obj-init-x", "-0.35", "-0.12", "5", "--obj-init-y", "-0.02", "0.42", "5",
                "--additional-env-build-kwargs", opt, f"urdf_version={urdf}"] + _xy(0.35, 0.20) + _rot(0)))
    return jobs


def move_near(asset_dir):
    jobs = []
    for urdf in URDFS:
        jobs.append(("move_near", [
            "--robot", "google_robot_static", "--policy-setup", "google_robot",
            "--control-freq", "3", "--sim-freq", "513", "--max-episode-steps", "80",
            "--env-name", "MoveNearGoogleBakedTexInScene-v0", "--scene-name", "google_pick_coke_can_1_v4",
            "--rgb-overlay-path", f"{asset_dir}/real_inpainting/google_move_near_real_eval_1.png",
            "--obj-variation-mode", "episode", "--obj-episode-range", "0", "60",
            "--additional-env-build-kwargs", f"urdf_version={urdf}",
            "--additional-env-save-tags", "baked_except_bpb_orange"] + _xy(0.35, 0.21) + _rot(-0.09)))
    return jobs


DRAWER_POSES = {  # name: (x, y, yaw)
    "a0": (0.644, -0.179, -0.03), "a1": (0.765, -0.182, -0.02), "a2": (0.889, -0.203, -0.06),
    "b0": (0.652, 0.009, 0), "b1": (0.752, 0.009, 0), "b2": (0.851, 0.035, 0),
    "c0": (0.665, 0.224, 0), "c1": (0.765, 0.222, -0.025), "c2": (0.865, 0.222, -0.025),
}


def drawer(asset_dir):
    jobs = []
    envs = ["OpenTopDrawerCustomInScene-v0", "OpenMiddleDrawerCustomInScene-v0", "OpenBottomDrawerCustomInScene-v0",
            "CloseTopDrawerCustomInScene-v0", "CloseMiddleDrawerCustomInScene-v0", "CloseBottomDrawerCustomInScene-v0"]
    for urdf in URDFS:
        for env in envs:
            for pose, (x, y, yaw) in DRAWER_POSES.items():
                jobs.append(("open_drawer" if env.startswith("Open") else "close_drawer", [
                    "--robot", "google_robot_static", "--policy-setup", "google_robot",
                    "--control-freq", "3", "--sim-freq", "513", "--max-episode-steps", "113",
                    "--env-name", env, "--scene-name", "dummy_drawer",
                    "--obj-init-x-range", "0", "0", "1", "--obj-init-y-range", "0", "0", "1",
                    "--rgb-overlay-path", f"{asset_dir}/real_inpainting/open_drawer_{pose}.png",
                    "--enable-raytracing", "--additional-env-build-kwargs", "station_name=mk_station_recolor",
                    "light_mode=simple", "disable_bad_material=True", f"urdf_version={urdf}"]
                    + _xy(x, y) + _rot(yaw)))
    return jobs


SUITES = {"bridge": bridge, "coke_can": coke_can, "move_near": move_near, "drawer": drawer}
EMBODIMENT = {"bridge": "widowx", "coke_can": "google_robot", "move_near": "google_robot", "drawer": "google_robot"}
