import os
import habitat

os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"

config_path = "run_r2r/iter_train.yaml"
config = habitat.get_config(config_path)

# Override dataset paths if necessary (modify to your real paths)
# config.defrost()
# config.DATASET.DATA_PATH = "/datasets/bevbert/vln_ce/v1"
# config.DATASET.SCENES_DIR = "/datasets/bevbert/matterport3d/v1/scans"
# config.freeze()

print("Creating VLN-CE environment...")
env = habitat.Env(config)
print("Environment created successfully!")
env.close()