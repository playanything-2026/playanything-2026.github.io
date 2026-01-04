from typing import Dict, Any
from copy import deepcopy
import os
from PIL import Image

from cradle import constants
from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.memory import LocalMemory
from cradle.utils.image_utils import draw_axis, draw_mask_panel, draw_color_band, draw_grids, crop_grow_image
from cradle.config import Config

logger = Logger()
memory = LocalMemory()
config = Config()

class AugmentProvider(BaseProvider):

    def __init__(self,
                 draw_axis: bool = False,
                 draw_grid: bool = False,
                 draw_mask_panel: bool = False,
                 draw_color_band: bool = False,
                 axis_config: Dict[str, Any] = None,
                 grid_config: Dict[str, Any] = None,
                 mask_panel_config: Dict[str, Any] = None,
                 color_band_config: Dict[str, Any] = None,
                 ):

        super(AugmentProvider, self).__init__()

        self.draw_axis = draw_axis
        self.draw_grid = draw_grid
        self.draw_mask_panel = draw_mask_panel
        self.draw_color_band = draw_color_band
        self.axis_config = axis_config
        self.grid_config = grid_config
        self.mask_panel_config = mask_panel_config
        self.color_band_config = color_band_config


    def run(self, *args, image, **kwargs):

        if self.draw_mask_panel:
            image = draw_mask_panel(image, **self.mask_panel_config)

        if self.draw_axis:
            axis_config_copy = self.axis_config.copy()
            axis_config_copy.pop('crop_region', None)

            image = draw_axis(image, **axis_config_copy)

        if self.draw_grid:
            grid_config_copy = self.grid_config.copy()
            grid_config_copy.pop('crop_region', None)

            image = draw_grids(image, **grid_config_copy)

        if self.draw_color_band:
            image = draw_color_band(image, **self.color_band_config)

        return image


    @BaseProvider.write
    def __call__(self,
                 *args,
                 **kwargs) -> Dict[str, Any]:

        logger.write(f"Draw axis on the screen shot.")

        params = deepcopy(memory.working_area)
        logger.write(f"Draw Axis: {self.draw_axis}")

        screenshot_path = params.get(constants.IMAGES_MEM_BUCKET, None)

        # 如果使用 game_region，则截图已经是正确区域，无需再裁剪
        # 只有在没有 game_region 时才使用 crop_region_local 进行裁剪
        if not getattr(config, 'game_region', None):
            crop_region = config.crop_region_local
            if crop_region:
                try:
                    screenshot_path = crop_grow_image(
                        screenshot_path,
                        custom_crop_region=crop_region,
                        overwrite_flag=False
                    )
                    logger.debug(f"Image cropped to region {crop_region}. New path: {screenshot_path}")
                except Exception as e:
                    logger.error(f"Error during image cropping: {e}. Proceeding with original image.")
        else:
            logger.debug(f"Using game_region, skipping crop_region_local cropping")

        augmented_screenshot_path = screenshot_path.replace(".jpg", "_augmented.jpg")  # 确保基于裁剪后的路径

        if not os.path.exists(screenshot_path):
            logger.error(f"screenshot_path {screenshot_path} not exists")
        else:

            if os.path.exists(augmented_screenshot_path):
                image = Image.open(augmented_screenshot_path)
            else:
                # 图像加载使用裁剪后的路径 (或原路径)
                image = Image.open(screenshot_path)

            if image is None:
                # 图像加载失败，记录错误并返回
                logger.write(f"Image Size after loading: {image.size}")
                logger.error(
                    f"Image object is None after loading from path {screenshot_path}. Cannot run augmentations.")
                return {constants.AUGMENTED_IMAGES_MEM_BUCKET: None}

            logger.write(f"Image Size before run(): {image.size}")
            image = self.run(image=image)  # 只有当 image 有效时才调用
            logger.write(f"Image Size after run(): {image.size}")
            image.save(augmented_screenshot_path)

        res_params = {
            constants.AUGMENTED_IMAGES_MEM_BUCKET: augmented_screenshot_path
        }

        memory.update_info_history(res_params)

        del params

        return res_params
