"""Serve GALA actions over ZMQ."""
from dataclasses import dataclass
import tyro
from gala.checkpoint import resolve_checkpoint, load_config
from gala.experiment.data_config import load_data_config
from gala.model.policy import GALAPolicy
from gala.eval.robot import RobotInferenceServer

@dataclass
class Args:
    model_path: str
    server: bool = False
    port: int = 5555
    host: str = '127.0.0.1'
    data_config: str = 'fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only'
    denoising_steps: int = 4
    infer_sample_shift: float = 1.0


def load_policy(args):
    path = resolve_checkpoint(args.model_path)
    config = load_config(path)
    bridge = config.bridge_cfg
    data = load_data_config(
        args.data_config, eagle_path=config.backbone_cfg['eagle_path'],
        use_bridge=bridge['use_bridge'],
        ignore_lang_prefix=getattr(config, 'ignore_lang_prefix', False),
        num_bridge_tokens=bridge['num_bridge_tokens'],
    )
    transform = data.transform()
    return GALAPolicy(
        model_path=path, embodiment_tag='gr1',
        modality_config=data.modality_config(), modality_transform=transform,
        denoising_steps=args.denoising_steps, infer_sample_shift=args.infer_sample_shift,
        tokenizer_len=len(transform.transforms[-1].eagle_processor.tokenizer),
    )

if __name__ == '__main__':
    args = tyro.cli(Args)
    if not args.server:
        raise ValueError('Pass --server to start the inference service.')
    policy = load_policy(args)
    RobotInferenceServer(policy, host=args.host, port=args.port).run()
