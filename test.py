import isaacgym
import isaacgymenvs
import torch
import torch
import torchvision.transforms as T
from PIL import Image
import os


num_envs = 4
envs = isaacgymenvs.make(
    seed=0,
    task="PushToy",
    num_envs=num_envs,
    sim_device="cuda:0",
    rl_device="cuda:0",
    graphics_device_id=0,
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()


# for i in range(100):
while True:
    l_dp = []
    l_quat = []

    actions = torch.zeros((num_envs,) + envs.action_space.shape, device = 'cuda:0')
    actions[:, :2] = envs.states["obj_pos"][:, :2] - envs.states['eef_pos'][:, :2] # Desired position = object position (2D)
    actions[:, 2] = 1.0
    obs_dict, _ ,_, _ = envs.step(actions)

    # depth_tensor = obs_dict['obs'][:, 7:].clone().reshape(num_envs, 1, 128, 128)
    # # 저장 폴더
    # os.makedirs("depth_images", exist_ok=True)

    # # 정규화 및 PIL 변환 함수
    # to_pil = T.ToPILImage()

    # for j in range(depth_tensor.shape[0]):
    #     # depth는 일반적으로 float이므로, [0, 1]로 normalize 필요
    #     img = depth_tensor[j]
    #     img_min, img_max = img.min(), img.max()
    #     img_norm = (img - img_min) / (img_max - img_min + 1e-8)  # avoid divide by zero

    #     # PIL 이미지로 변환 및 저장
    #     pil_img = to_pil(img_norm)
    #     pil_img.save(f"depth_images/depth_{i}_{j}.png")
        


