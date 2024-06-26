import gym
import numpy as np

from typing import Any, Dict, Tuple
from common.autoencoder import load_ae


class AutoencoderWrapper(gym.Wrapper):
    """
    Wrapper to encode input image using pre-trained AutoEncoder

    :param env: Gym environment
    :param ae_path: absolute path to the pretrained AutoEncoder
    """

    def __init__(self, env: gym.Env, ae_path: str):
        super().__init__(env)
        self.autoencoder = load_ae(ae_path)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.autoencoder.z_size + 1, ),
            dtype=np.float32,
        )

        # self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        # Convert to BGR
        encoded_image = self.autoencoder.encode_from_raw_image(obs[:, :, ::-1])
        new_obs = np.concatenate([encoded_image.flatten(), [0.0]])
        return new_obs

    def step(
            self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, infos = self.env.step(action)
        # Encode with the pre-trained AutoEncoder
        encoded_image = self.autoencoder.encode_from_raw_image(obs[:, :, ::-1])
        # reconstructed_image = self.autoencoder.decode(encoded_image)[0]
        # cv2.imshow("Original", obs[:, :, ::-1])
        # cv2.imshow("Reconstruction", reconstructed_image)
        # # stop if escape is pressed
        # k = cv2.waitKey(0) & 0xFF
        # if k == 27:
        #     pass
        speed = infos["speed"]
        new_obs = np.concatenate([encoded_image.flatten(), [speed]])

        return new_obs, reward, done, infos


class HistoryWrapper(gym.Wrapper):
    """
    Stack past observations and actions to give an history to the agent.

    :param env: (gym.Env)
    :param horizon: (int) Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 5):
        assert isinstance(env.observation_space, gym.spaces.Box)

        wrapped_obs_space = env.observation_space
        wrapped_action_space = env.action_space

        # TODO: double check, it seems wrong when we have different low and highs
        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)

        low_action = np.repeat(wrapped_action_space.low, horizon, axis=-1)
        high_action = np.repeat(wrapped_action_space.high, horizon, axis=-1)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space = gym.spaces.Box(low=low,
                                               high=high,
                                               dtype=wrapped_obs_space.dtype)

        super(HistoryWrapper, self).__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self):
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs = self.env.reset()
        self.obs_history[..., -obs.shape[-1]:] = obs
        return self._create_obs_from_history()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history,
                                   shift=-last_ax_size,
                                   axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs

        self.action_history = np.roll(self.action_history,
                                      shift=-action.shape[-1],
                                      axis=-1)
        self.action_history[..., -action.shape[-1]:] = action
        return self._create_obs_from_history(), reward, done, info
