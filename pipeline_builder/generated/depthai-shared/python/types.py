# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = global_properties_from_dict(json.loads(json_string))
#     result = color_camera_properties_from_dict(json.loads(json_string))
#     result = my_consumer_properties_from_dict(json.loads(json_string))
#     result = my_producer_properties_from_dict(json.loads(json_string))
#     result = neural_network_properties_from_dict(json.loads(json_string))
#     result = video_encoder_properties_from_dict(json.loads(json_string))
#     result = x_link_in_properties_from_dict(json.loads(json_string))
#     result = x_link_out_properties_from_dict(json.loads(json_string))
#     result = processor_type_from_dict(json.loads(json_string))
#     result = pipeline_schema_from_dict(json.loads(json_string))

from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional, List, TypeVar, Type, Callable, cast


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


class ColorOrderInsidePixel(Enum):
    """For 24 bit color these can be either RGB or BGR"""
    BGR = "bgr"
    RGB = "rgb"


class CameraSensorResolution(Enum):
    """Select the camera sensor resolution"""
    THE_1080_P = "1080p"
    THE_4_K = "4k"


@dataclass
class ColorCameraProperties:
    """Specify ColorCamera options such as camera ID, ..."""
    """Which color camera the node will use"""
    cam_id: int
    """For 24 bit color these can be either RGB or BGR"""
    color_order: ColorOrderInsidePixel
    """Are colors interleaved (R1G1B1, R2G2B2, ...) or planar (R1R2..., G1G2..., B1B2)"""
    interleaved: bool
    """Preview frame output height"""
    preview_height: float
    """Preview frame output width"""
    preview_width: float
    """Select the camera sensor resolution"""
    resolution: CameraSensorResolution

    @staticmethod
    def from_dict(obj: Any) -> 'ColorCameraProperties':
        assert isinstance(obj, dict)
        cam_id = from_int(obj.get("camId"))
        color_order = ColorOrderInsidePixel(obj.get("colorOrder"))
        interleaved = from_bool(obj.get("interleaved"))
        preview_height = from_float(obj.get("previewHeight"))
        preview_width = from_float(obj.get("previewWidth"))
        resolution = CameraSensorResolution(obj.get("resolution"))
        return ColorCameraProperties(cam_id, color_order, interleaved, preview_height, preview_width, resolution)

    def to_dict(self) -> dict:
        result: dict = {}
        result["camId"] = from_int(self.cam_id)
        result["colorOrder"] = to_enum(ColorOrderInsidePixel, self.color_order)
        result["interleaved"] = from_bool(self.interleaved)
        result["previewHeight"] = to_float(self.preview_height)
        result["previewWidth"] = to_float(self.preview_width)
        result["resolution"] = to_enum(CameraSensorResolution, self.resolution)
        return result


class ProcessorType(Enum):
    """On which processor the node will be placed
    
    Enum specifying processor
    """
    LOS = "LOS"
    LRT = "LRT"


@dataclass
class MyConsumerProperties:
    """Specify message and processor placement of MyConsumer node"""
    """On which processor the node will be placed"""
    processor_placement: ProcessorType

    @staticmethod
    def from_dict(obj: Any) -> 'MyConsumerProperties':
        assert isinstance(obj, dict)
        processor_placement = ProcessorType(obj.get("processorPlacement"))
        return MyConsumerProperties(processor_placement)

    def to_dict(self) -> dict:
        result: dict = {}
        result["processorPlacement"] = to_enum(ProcessorType, self.processor_placement)
        return result


@dataclass
class MyProducerProperties:
    """Specify message and processor placement of MyProducer node"""
    """On which processor the node will be placed"""
    processor_placement: ProcessorType
    message: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'MyProducerProperties':
        assert isinstance(obj, dict)
        processor_placement = ProcessorType(obj.get("processorPlacement"))
        message = from_union([from_str, from_none], obj.get("message"))
        return MyProducerProperties(processor_placement, message)

    def to_dict(self) -> dict:
        result: dict = {}
        result["processorPlacement"] = to_enum(ProcessorType, self.processor_placement)
        result["message"] = from_union([from_str, from_none], self.message)
        return result


@dataclass
class NeuralNetworkProperties:
    """Specify NeuralNetwork options such as blob path, ..."""
    """Uri which points to blob"""
    blob_uri: str
    """Blob binary size in bytes"""
    blob_size: Optional[int] = None
    """Number of available output tensors in pool"""
    num_frames: Optional[int] = None

    @staticmethod
    def from_dict(obj: Any) -> 'NeuralNetworkProperties':
        assert isinstance(obj, dict)
        blob_uri = from_str(obj.get("blobUri"))
        blob_size = from_union([from_int, from_none], obj.get("blobSize"))
        num_frames = from_union([from_int, from_none], obj.get("numFrames"))
        return NeuralNetworkProperties(blob_uri, blob_size, num_frames)

    def to_dict(self) -> dict:
        result: dict = {}
        result["blobUri"] = from_str(self.blob_uri)
        result["blobSize"] = from_union([from_int, from_none], self.blob_size)
        result["numFrames"] = from_union([from_int, from_none], self.num_frames)
        return result


class VideoEncoderProfile(Enum):
    """Encoding profile, H264, H265 or MJPEG"""
    H264_BASELINE = "H264_BASELINE"
    H264_HIGH = "H264_HIGH"
    H264_MAIN = "H264_MAIN"
    H265_MAIN = "H265_MAIN"
    MJPEG = "MJPEG"


class RateControlMode(Enum):
    """Rate control mode specifies if constant or variable bitrate should be used (H264 / H265)"""
    CBR = "CBR"
    VBR = "VBR"


@dataclass
class VideoEncoderProperties:
    """Specify VideoEncoder options such as profile, bitrate, ..."""
    """Input and compressed output frame height"""
    height: int
    """Encoding profile, H264, H265 or MJPEG"""
    profile: VideoEncoderProfile
    """Input and compressed output frame width"""
    width: int
    """Specifies prefered bitrate of compressed output bitstream"""
    bitrate: Optional[int] = None
    """Every x number of frames a keyframe will be inserted"""
    keyframe_frequency: Optional[int] = None
    """Specifies maximum bitrate of compressed output bitstream"""
    max_bitrate: Optional[int] = None
    """Specifies number of B frames to be inserted"""
    num_b_frames: Optional[int] = None
    """This options specifies how many frames are available in this nodes pool (can help if
    receiver node is slow at consuming
    """
    num_frames_pool: Optional[int] = None
    """Value between 0-100% (approximates quality)"""
    quality: Optional[int] = None
    """Rate control mode specifies if constant or variable bitrate should be used (H264 / H265)"""
    rate_ctrl_mode: Optional[RateControlMode] = None

    @staticmethod
    def from_dict(obj: Any) -> 'VideoEncoderProperties':
        assert isinstance(obj, dict)
        height = from_int(obj.get("height"))
        profile = VideoEncoderProfile(obj.get("profile"))
        width = from_int(obj.get("width"))
        bitrate = from_union([from_int, from_none], obj.get("bitrate"))
        keyframe_frequency = from_union([from_int, from_none], obj.get("keyframeFrequency"))
        max_bitrate = from_union([from_int, from_none], obj.get("maxBitrate"))
        num_b_frames = from_union([from_int, from_none], obj.get("numBFrames"))
        num_frames_pool = from_union([from_int, from_none], obj.get("numFramesPool"))
        quality = from_union([from_int, from_none], obj.get("quality"))
        rate_ctrl_mode = from_union([RateControlMode, from_none], obj.get("rateCtrlMode"))
        return VideoEncoderProperties(height, profile, width, bitrate, keyframe_frequency, max_bitrate, num_b_frames, num_frames_pool, quality, rate_ctrl_mode)

    def to_dict(self) -> dict:
        result: dict = {}
        result["height"] = from_int(self.height)
        result["profile"] = to_enum(VideoEncoderProfile, self.profile)
        result["width"] = from_int(self.width)
        result["bitrate"] = from_union([from_int, from_none], self.bitrate)
        result["keyframeFrequency"] = from_union([from_int, from_none], self.keyframe_frequency)
        result["maxBitrate"] = from_union([from_int, from_none], self.max_bitrate)
        result["numBFrames"] = from_union([from_int, from_none], self.num_b_frames)
        result["numFramesPool"] = from_union([from_int, from_none], self.num_frames_pool)
        result["quality"] = from_union([from_int, from_none], self.quality)
        result["rateCtrlMode"] = from_union([lambda x: to_enum(RateControlMode, x), from_none], self.rate_ctrl_mode)
        return result


@dataclass
class XLinkInProperties:
    """Properties for XLinkIn which define stream name"""
    stream_name: str

    @staticmethod
    def from_dict(obj: Any) -> 'XLinkInProperties':
        assert isinstance(obj, dict)
        stream_name = from_str(obj.get("streamName"))
        return XLinkInProperties(stream_name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["streamName"] = from_str(self.stream_name)
        return result


@dataclass
class XLinkOutProperties:
    """Properties for XLinkOut which define stream name"""
    """Set a limit to how many packets will be sent further to host"""
    max_fps_limit: float
    stream_name: str

    @staticmethod
    def from_dict(obj: Any) -> 'XLinkOutProperties':
        assert isinstance(obj, dict)
        max_fps_limit = from_float(obj.get("maxFpsLimit"))
        stream_name = from_str(obj.get("streamName"))
        return XLinkOutProperties(max_fps_limit, stream_name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["maxFpsLimit"] = to_float(self.max_fps_limit)
        result["streamName"] = from_str(self.stream_name)
        return result


@dataclass
class NodeConnectionSchema:
    """Specifies a connection between nodes IOs"""
    node1_id: int
    node1_output: str
    node2_id: int
    node2_input: str

    @staticmethod
    def from_dict(obj: Any) -> 'NodeConnectionSchema':
        assert isinstance(obj, dict)
        node1_id = from_int(obj.get("node1Id"))
        node1_output = from_str(obj.get("node1Output"))
        node2_id = from_int(obj.get("node2Id"))
        node2_input = from_str(obj.get("node2Input"))
        return NodeConnectionSchema(node1_id, node1_output, node2_id, node2_input)

    def to_dict(self) -> dict:
        result: dict = {}
        result["node1Id"] = from_int(self.node1_id)
        result["node1Output"] = from_str(self.node1_output)
        result["node2Id"] = from_int(self.node2_id)
        result["node2Input"] = from_str(self.node2_input)
        return result


@dataclass
class GlobalProperties:
    """Specify properties which apply for whole pipeline"""
    """Set frequency of Leon OS - Incresing can improve performance, at the cost of higher power
    draw
    """
    leon_os_frequency_khz: float
    """Set frequency of Leon RT - Incresing can improve performance, at the cost of higher power
    draw
    """
    leon_rt_frequency_khz: float
    pipeline_name: Optional[str] = None
    pipeline_version: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'GlobalProperties':
        assert isinstance(obj, dict)
        leon_os_frequency_khz = from_float(obj.get("leonOsFrequencyKhz"))
        leon_rt_frequency_khz = from_float(obj.get("leonRtFrequencyKhz"))
        pipeline_name = from_union([from_str, from_none], obj.get("pipelineName"))
        pipeline_version = from_union([from_str, from_none], obj.get("pipelineVersion"))
        return GlobalProperties(leon_os_frequency_khz, leon_rt_frequency_khz, pipeline_name, pipeline_version)

    def to_dict(self) -> dict:
        result: dict = {}
        result["leonOsFrequencyKhz"] = to_float(self.leon_os_frequency_khz)
        result["leonRtFrequencyKhz"] = to_float(self.leon_rt_frequency_khz)
        result["pipelineName"] = from_union([from_str, from_none], self.pipeline_name)
        result["pipelineVersion"] = from_union([from_str, from_none], self.pipeline_version)
        return result


@dataclass
class NodeObjInfo:
    id: int
    name: str
    properties: Any

    @staticmethod
    def from_dict(obj: Any) -> 'NodeObjInfo':
        assert isinstance(obj, dict)
        id = from_int(obj.get("id"))
        name = from_str(obj.get("name"))
        properties = obj.get("properties")
        return NodeObjInfo(id, name, properties)

    def to_dict(self) -> dict:
        result: dict = {}
        result["id"] = from_int(self.id)
        result["name"] = from_str(self.name)
        result["properties"] = self.properties
        return result


@dataclass
class PipelineSchema:
    """Specifies whole pipeline, nodes, properties and connections between nodes IOs"""
    connections: List[NodeConnectionSchema]
    global_properties: GlobalProperties
    nodes: List[NodeObjInfo]

    @staticmethod
    def from_dict(obj: Any) -> 'PipelineSchema':
        assert isinstance(obj, dict)
        connections = from_list(NodeConnectionSchema.from_dict, obj.get("connections"))
        global_properties = GlobalProperties.from_dict(obj.get("globalProperties"))
        nodes = from_list(NodeObjInfo.from_dict, obj.get("nodes"))
        return PipelineSchema(connections, global_properties, nodes)

    def to_dict(self) -> dict:
        result: dict = {}
        result["connections"] = from_list(lambda x: to_class(NodeConnectionSchema, x), self.connections)
        result["globalProperties"] = to_class(GlobalProperties, self.global_properties)
        result["nodes"] = from_list(lambda x: to_class(NodeObjInfo, x), self.nodes)
        return result


def global_properties_from_dict(s: Any) -> GlobalProperties:
    return GlobalProperties.from_dict(s)


def global_properties_to_dict(x: GlobalProperties) -> Any:
    return to_class(GlobalProperties, x)


def color_camera_properties_from_dict(s: Any) -> ColorCameraProperties:
    return ColorCameraProperties.from_dict(s)


def color_camera_properties_to_dict(x: ColorCameraProperties) -> Any:
    return to_class(ColorCameraProperties, x)


def my_consumer_properties_from_dict(s: Any) -> MyConsumerProperties:
    return MyConsumerProperties.from_dict(s)


def my_consumer_properties_to_dict(x: MyConsumerProperties) -> Any:
    return to_class(MyConsumerProperties, x)


def my_producer_properties_from_dict(s: Any) -> MyProducerProperties:
    return MyProducerProperties.from_dict(s)


def my_producer_properties_to_dict(x: MyProducerProperties) -> Any:
    return to_class(MyProducerProperties, x)


def neural_network_properties_from_dict(s: Any) -> NeuralNetworkProperties:
    return NeuralNetworkProperties.from_dict(s)


def neural_network_properties_to_dict(x: NeuralNetworkProperties) -> Any:
    return to_class(NeuralNetworkProperties, x)


def video_encoder_properties_from_dict(s: Any) -> VideoEncoderProperties:
    return VideoEncoderProperties.from_dict(s)


def video_encoder_properties_to_dict(x: VideoEncoderProperties) -> Any:
    return to_class(VideoEncoderProperties, x)


def x_link_in_properties_from_dict(s: Any) -> XLinkInProperties:
    return XLinkInProperties.from_dict(s)


def x_link_in_properties_to_dict(x: XLinkInProperties) -> Any:
    return to_class(XLinkInProperties, x)


def x_link_out_properties_from_dict(s: Any) -> XLinkOutProperties:
    return XLinkOutProperties.from_dict(s)


def x_link_out_properties_to_dict(x: XLinkOutProperties) -> Any:
    return to_class(XLinkOutProperties, x)


def processor_type_from_dict(s: Any) -> ProcessorType:
    return ProcessorType(s)


def processor_type_to_dict(x: ProcessorType) -> Any:
    return to_enum(ProcessorType, x)


def pipeline_schema_from_dict(s: Any) -> PipelineSchema:
    return PipelineSchema.from_dict(s)


def pipeline_schema_to_dict(x: PipelineSchema) -> Any:
    return to_class(PipelineSchema, x)
