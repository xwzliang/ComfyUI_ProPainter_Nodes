import torch
from comfy import model_management

from .propainter_inference import (
    ProPainterConfig,
    feature_propagation,
    process_inpainting,
)
from .utils.image_utils import (
    ImageConfig,
    ImageOutpaintConfig,
    convert_image_to_frames,
    handle_output,
    prepare_frames_and_masks,
    extrapolation,
    prepare_frames_and_masks_for_outpaint,
)
from .utils.model_utils import initialize_models


def check_inputs(frames: torch.Tensor, masks: torch.Tensor) -> Exception | None:
    if frames.size(dim=0) <= 1:
        raise Exception(f"""Image length must be greater than 1, but got:
                        Image length: ({frames.size(dim=0)})""")
    if frames.size(dim=0) != masks.size(dim=0) and masks.size(dim=0) != 1:
        raise Exception(f"""Image and Mask must have the same length or Mask have length 1, but got:
                        Image length: {frames.size(dim=0)}
                        Mask length: {masks.size(dim=0)}""")

    if frames.size(dim=1) != masks.size(dim=1) or frames.size(dim=2) != masks.size(
        dim=2
    ):
        raise Exception(f"""Image and Mask must have the same dimensions, but got:
                        Image: ({frames.size(dim=1)}, {frames.size(dim=2)})
                        Mask: ({masks.size(dim=1)}, {masks.size(dim=2)})""")


class ProPainterInpaint:
    """ComfyUI Node for performing inpainting on video frames using ProPainter."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # --video
                "mask": ("MASK",),  # --mask
                "width": ("INT", {"default": 640, "min": 0, "max": 2560}),  # --width
                "height": ("INT", {"default": 360, "min": 0, "max": 2560}),  # --height
                "mask_dilates": (
                    "INT",
                    {"default": 5, "min": 0, "max": 100},
                ),  # --mask_dilates
                "flow_mask_dilates": (
                    "INT",
                    {"default": 8, "min": 0, "max": 100},
                ),  # arg dont exist on original code
                "ref_stride": (
                    "INT",
                    {"default": 10, "min": 1, "max": 100},
                ),  # --ref_stride
                "neighbor_length": (
                    "INT",
                    {"default": 10, "min": 2, "max": 300},
                ),  # --neighbor_length
                "subvideo_length": (
                    "INT",
                    {"default": 80, "min": 1, "max": 300},
                ),  # --subvideo_length
                "raft_iter": (
                    "INT",
                    {"default": 20, "min": 1, "max": 100},
                ),  # --raft_iter
                "fp16": (["enable", "disable"],),  # --fp16
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "MASK",
    )
    RETURN_NAMES = (
        "IMAGE",
        "FLOW_MASK",
        "MASK_DILATE",
    )
    FUNCTION = "propainter_inpainting"
    CATEGORY = "ProPainter"

    def propainter_inpainting(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        width: int,
        height: int,
        mask_dilates: int,
        flow_mask_dilates: int,
        ref_stride: int,
        neighbor_length: int,
        subvideo_length: int,
        raft_iter: int,
        fp16: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform inpainting on images input using the ProPainter model inference."""
        check_inputs(image, mask)
        device = model_management.get_torch_device()
        # TODO: Check if this convertion from Torch to PIL is really necessary.
        frames = convert_image_to_frames(image)
        # video_length = image.size(dim=0)
        video_length = 2
        input_size = frames[0].size

        image_config = ImageConfig(
            width, height, mask_dilates, flow_mask_dilates, input_size, video_length
        )
        inpaint_config = ProPainterConfig(
            ref_stride,
            neighbor_length,
            subvideo_length,
            raft_iter,
            fp16,
            video_length,
            device,
            image_config.process_size,
        )
        models = initialize_models(device, inpaint_config.fp16)

        def process_frame_pair(frames):
            image_config = ImageConfig(
                width, height, mask_dilates, flow_mask_dilates, input_size, len(frames)
            )
            inpaint_config = ProPainterConfig(
                ref_stride,
                neighbor_length,
                subvideo_length,
                raft_iter,
                fp16,
                len(frames),
                device,
                image_config.process_size,
            )
            frames_tensor, flow_masks_tensor, masks_dilated_tensor, original_frames = (
                prepare_frames_and_masks(frames, mask, image_config, device)
            )

            print(f"\nProcessing  {inpaint_config.video_length} frames...")

            updated_frames, updated_masks, pred_flows_bi = process_inpainting(
                models,
                frames_tensor,
                flow_masks_tensor,
                masks_dilated_tensor,
                inpaint_config,
            )

            composed_frames = feature_propagation(
                models.inpaint_model,
                updated_frames,
                updated_masks,
                masks_dilated_tensor,
                pred_flows_bi,
                original_frames,
                inpaint_config,
            )
            return (composed_frames, flow_masks_tensor, masks_dilated_tensor)

        processed_frames = []
        # Check if there's an odd number of frames
        if len(frames) % 2 == 1:
            # Process all but the last 3 frames in pairs
            for i in range(0, len(frames) - 3, 2):
                frame_pair = frames[i:i+2]
                processed_pair, flow_masks_tensor, masks_dilated_tensor = process_frame_pair(frame_pair)
                processed_frames.extend(processed_pair)

            # Now process the last 3 frames together (or in whatever way makes sense)
            last_three = frames[-3:]
            processed_triple, flow_masks_tensor, masks_dilated_tensor = process_frame_pair(last_three)
            processed_frames.extend(processed_triple)

        else:
            # If even, just process all frames in pairs
            for i in range(0, len(frames), 2):
                frame_pair = frames[i:i+2]
                processed_pair, flow_masks_tensor, masks_dilated_tensor = process_frame_pair(frame_pair)
                processed_frames.extend(processed_pair)

        return handle_output(processed_frames, flow_masks_tensor, masks_dilated_tensor)


class ProPainterOutpaint:
    """ComfyUI Node for performing outpainting on video frames using ProPainter."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # --video
                "width": ("INT", {"default": 640, "min": 0, "max": 2560}),  # --width
                "height": ("INT", {"default": 360, "min": 0, "max": 2560}),  # --height
                "width_scale": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                    },
                ),
                "height_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                    },
                ),
                "mask_dilates": (
                    "INT",
                    {"default": 5, "min": 0, "max": 100},
                ),  # --mask_dilates
                "flow_mask_dilates": (
                    "INT",
                    {"default": 8, "min": 0, "max": 100},
                ),  # arg dont exist on original code
                "ref_stride": (
                    "INT",
                    {"default": 10, "min": 1, "max": 100},
                ),  # --ref_stride
                "neighbor_length": (
                    "INT",
                    {"default": 10, "min": 2, "max": 300},
                ),  # --neighbor_length
                "subvideo_length": (
                    "INT",
                    {"default": 80, "min": 1, "max": 300},
                ),  # --subvideo_length
                "raft_iter": (
                    "INT",
                    {"default": 20, "min": 1, "max": 100},
                ),  # --raft_iter
                "fp16": (["enable", "disable"],),  # --fp16
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "IMAGE",
        "OUTPAINT_MASK",
        "output_width",
        "output_height",
    )
    FUNCTION = "propainter_outpainting"
    CATEGORY = "ProPainter"

    def propainter_outpainting(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        width_scale: float,
        height_scale: float,
        mask_dilates: int,
        flow_mask_dilates: int,
        ref_stride: int,
        neighbor_length: int,
        subvideo_length: int,
        raft_iter: int,
        fp16: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Perform inpainting on images input using the ProPainter model inference."""
        device = model_management.get_torch_device()
        # TODO: Check if this convertion from Torch to PIL is really necessary.
        frames = convert_image_to_frames(image)
        video_length = image.size(dim=0)
        input_size = frames[0].size

        image_config = ImageOutpaintConfig(
            width,
            height,
            mask_dilates,
            flow_mask_dilates,
            input_size,
            video_length,
            width_scale,
            height_scale,
        )

        outpaint_config = ProPainterConfig(
            ref_stride,
            neighbor_length,
            subvideo_length,
            raft_iter,
            fp16,
            video_length,
            device,
            image_config.outpaint_size,
        )

        paded_frames, paded_flow_masks, paded_masks_dilated = extrapolation(
            frames, image_config
        )

        frames_tensor, flow_masks_tensor, masks_dilated_tensor, original_frames = (
            prepare_frames_and_masks_for_outpaint(
                paded_frames, paded_flow_masks, paded_masks_dilated, device
            )
        )

        models = initialize_models(device, outpaint_config.fp16)
        print(f"\nProcessing  {outpaint_config.video_length} frames...")

        updated_frames, updated_masks, pred_flows_bi = process_inpainting(
            models,
            frames_tensor,
            flow_masks_tensor,
            masks_dilated_tensor,
            outpaint_config,
        )

        composed_frames = feature_propagation(
            models.inpaint_model,
            updated_frames,
            updated_masks,
            masks_dilated_tensor,
            pred_flows_bi,
            original_frames,
            outpaint_config,
        )

        output_frames, output_masks, _ = handle_output(
            composed_frames, flow_masks_tensor, masks_dilated_tensor
        )
        output_width, output_height = outpaint_config.process_size
        return output_frames, output_masks, output_width, output_height


NODE_CLASS_MAPPINGS = {
    "ProPainterInpaint": ProPainterInpaint,
    "ProPainterOutpaint": ProPainterOutpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProPainterInpaint": "ProPainter Inpainting",
    "ProPainterOutpaint": "ProPainter Outpainting",
}
