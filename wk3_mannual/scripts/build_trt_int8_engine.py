from pathlib import Path
import torch
import tensorrt as trt
from utils import preprocess


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir: Path, cache_file: Path):
        super().__init__()
        self.images = list(image_dir.glob('*.jpg'))
        self.cache_file = cache_file
        self.current_index = 0
        self.device_input = torch.zeros((1,3,640,640), dtype=torch.float32).cuda()

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        if self.current_index >= len(self.images):
            return None
        img = preprocess(self.images[self.current_index])
        self.device_input.copy_(torch.from_numpy(img))
        self.current_index += 1
        return [self.device_input.data_ptr()]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.write_bytes(bytes(cache))


def main():
    project_root = Path(__file__).resolve().parent.parent
    onnx_path = project_root / 'models' / 'yolov5su.onnx'
    engine_path = project_root / 'engines' / 'yolov5su_int8.engine'
    calib_images = Path('/mnt/d/intern_dataset/VisDrone2019-DET-test-dev/images')
    cache_file = project_root / 'engines' / 'calib.cache'
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = Int8Calibrator(calib_images, cache_file)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('Failed to build engine')
    
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    print(f'INT8 engine saved to {engine_path}')

if __name__ == '__main__':
    main()