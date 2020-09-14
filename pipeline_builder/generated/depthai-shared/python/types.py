# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = common_from_dict(json.loads(json_string))
#     result = global_from_dict(json.loads(json_string))
#     result = schema_from_dict(json.loads(json_string))

from typing import Any, List, Optional, TypeVar, Callable, Type, cast


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
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


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


class Child:
    datatype: str

    def __init__(self, datatype: str) -> None:
        self.datatype = datatype

    @staticmethod
    def from_dict(obj: Any) -> 'Child':
        assert isinstance(obj, dict)
        datatype = from_str(obj.get("datatype"))
        return Child(datatype)

    def to_dict(self) -> dict:
        result: dict = {}
        result["datatype"] = from_str(self.datatype)
        return result


class Hierarchy:
    datatype: str
    children: List[Child]

    def __init__(self, datatype: str, children: List[Child]) -> None:
        self.datatype = datatype
        self.children = children

    @staticmethod
    def from_dict(obj: Any) -> 'Hierarchy':
        assert isinstance(obj, dict)
        datatype = from_str(obj.get("datatype"))
        children = from_list(Child.from_dict, obj.get("children"))
        return Hierarchy(datatype, children)

    def to_dict(self) -> dict:
        result: dict = {}
        result["datatype"] = from_str(self.datatype)
        result["children"] = from_list(lambda x: to_class(Child, x), self.children)
        return result


class LeonFrequencyKhz:
    title: str
    description: str
    type: str
    minimum: int
    maximum: int
    default: int

    def __init__(self, title: str, description: str, type: str, minimum: int, maximum: int, default: int) -> None:
        self.title = title
        self.description = description
        self.type = type
        self.minimum = minimum
        self.maximum = maximum
        self.default = default

    @staticmethod
    def from_dict(obj: Any) -> 'LeonFrequencyKhz':
        assert isinstance(obj, dict)
        title = from_str(obj.get("title"))
        description = from_str(obj.get("description"))
        type = from_str(obj.get("type"))
        minimum = from_int(obj.get("minimum"))
        maximum = from_int(obj.get("maximum"))
        default = from_int(obj.get("default"))
        return LeonFrequencyKhz(title, description, type, minimum, maximum, default)

    def to_dict(self) -> dict:
        result: dict = {}
        result["title"] = from_str(self.title)
        result["description"] = from_str(self.description)
        result["type"] = from_str(self.type)
        result["minimum"] = from_int(self.minimum)
        result["maximum"] = from_int(self.maximum)
        result["default"] = from_int(self.default)
        return result


class PipelineName:
    type: str

    def __init__(self, type: str) -> None:
        self.type = type

    @staticmethod
    def from_dict(obj: Any) -> 'PipelineName':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        return PipelineName(type)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        return result


class CommonProperties:
    pipeline_version: PipelineName
    pipeline_name: PipelineName
    leon_os_frequency_khz: LeonFrequencyKhz
    leon_rt_frequency_khz: LeonFrequencyKhz

    def __init__(self, pipeline_version: PipelineName, pipeline_name: PipelineName, leon_os_frequency_khz: LeonFrequencyKhz, leon_rt_frequency_khz: LeonFrequencyKhz) -> None:
        self.pipeline_version = pipeline_version
        self.pipeline_name = pipeline_name
        self.leon_os_frequency_khz = leon_os_frequency_khz
        self.leon_rt_frequency_khz = leon_rt_frequency_khz

    @staticmethod
    def from_dict(obj: Any) -> 'CommonProperties':
        assert isinstance(obj, dict)
        pipeline_version = PipelineName.from_dict(obj.get("pipelineVersion"))
        pipeline_name = PipelineName.from_dict(obj.get("pipelineName"))
        leon_os_frequency_khz = LeonFrequencyKhz.from_dict(obj.get("leonOsFrequencyKhz"))
        leon_rt_frequency_khz = LeonFrequencyKhz.from_dict(obj.get("leonRtFrequencyKhz"))
        return CommonProperties(pipeline_version, pipeline_name, leon_os_frequency_khz, leon_rt_frequency_khz)

    def to_dict(self) -> dict:
        result: dict = {}
        result["pipelineVersion"] = to_class(PipelineName, self.pipeline_version)
        result["pipelineName"] = to_class(PipelineName, self.pipeline_name)
        result["leonOsFrequencyKhz"] = to_class(LeonFrequencyKhz, self.leon_os_frequency_khz)
        result["leonRtFrequencyKhz"] = to_class(LeonFrequencyKhz, self.leon_rt_frequency_khz)
        return result


class Global:
    id: Optional[str]
    schema: Optional[str]
    title: Optional[str]
    description: Optional[str]
    type: Optional[str]
    enum: Optional[List[str]]
    hierarchies: Optional[List[Hierarchy]]
    required: Optional[List[str]]
    properties: Optional[CommonProperties]

    def __init__(self, id: Optional[str], schema: Optional[str], title: Optional[str], description: Optional[str], type: Optional[str], enum: Optional[List[str]], hierarchies: Optional[List[Hierarchy]], required: Optional[List[str]], properties: Optional[CommonProperties]) -> None:
        self.id = id
        self.schema = schema
        self.title = title
        self.description = description
        self.type = type
        self.enum = enum
        self.hierarchies = hierarchies
        self.required = required
        self.properties = properties

    @staticmethod
    def from_dict(obj: Any) -> 'Global':
        assert isinstance(obj, dict)
        id = from_union([from_str, from_none], obj.get("$id"))
        schema = from_union([from_str, from_none], obj.get("$schema"))
        title = from_union([from_str, from_none], obj.get("title"))
        description = from_union([from_str, from_none], obj.get("description"))
        type = from_union([from_str, from_none], obj.get("type"))
        enum = from_union([lambda x: from_list(from_str, x), from_none], obj.get("enum"))
        hierarchies = from_union([lambda x: from_list(Hierarchy.from_dict, x), from_none], obj.get("hierarchies"))
        required = from_union([lambda x: from_list(from_str, x), from_none], obj.get("required"))
        properties = from_union([CommonProperties.from_dict, from_none], obj.get("properties"))
        return Global(id, schema, title, description, type, enum, hierarchies, required, properties)

    def to_dict(self) -> dict:
        result: dict = {}
        result["$id"] = from_union([from_str, from_none], self.id)
        result["$schema"] = from_union([from_str, from_none], self.schema)
        result["title"] = from_union([from_str, from_none], self.title)
        result["description"] = from_union([from_str, from_none], self.description)
        result["type"] = from_union([from_str, from_none], self.type)
        result["enum"] = from_union([lambda x: from_list(from_str, x), from_none], self.enum)
        result["hierarchies"] = from_union([lambda x: from_list(lambda x: to_class(Hierarchy, x), x), from_none], self.hierarchies)
        result["required"] = from_union([lambda x: from_list(from_str, x), from_none], self.required)
        result["properties"] = from_union([lambda x: to_class(CommonProperties, x), from_none], self.properties)
        return result


class GlobalProperties:
    ref: str

    def __init__(self, ref: str) -> None:
        self.ref = ref

    @staticmethod
    def from_dict(obj: Any) -> 'GlobalProperties':
        assert isinstance(obj, dict)
        ref = from_str(obj.get("$ref"))
        return GlobalProperties(ref)

    def to_dict(self) -> dict:
        result: dict = {}
        result["$ref"] = from_str(self.ref)
        return result


class Connections:
    type: str
    items: GlobalProperties

    def __init__(self, type: str, items: GlobalProperties) -> None:
        self.type = type
        self.items = items

    @staticmethod
    def from_dict(obj: Any) -> 'Connections':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        items = GlobalProperties.from_dict(obj.get("items"))
        return Connections(type, items)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["items"] = to_class(GlobalProperties, self.items)
        return result


class DatatypeEntryProperties:
    datatype: GlobalProperties
    children: Connections

    def __init__(self, datatype: GlobalProperties, children: Connections) -> None:
        self.datatype = datatype
        self.children = children

    @staticmethod
    def from_dict(obj: Any) -> 'DatatypeEntryProperties':
        assert isinstance(obj, dict)
        datatype = GlobalProperties.from_dict(obj.get("datatype"))
        children = Connections.from_dict(obj.get("children"))
        return DatatypeEntryProperties(datatype, children)

    def to_dict(self) -> dict:
        result: dict = {}
        result["datatype"] = to_class(GlobalProperties, self.datatype)
        result["children"] = to_class(Connections, self.children)
        return result


class DatatypeEntry:
    type: str
    additional_properties: bool
    required: List[str]
    properties: DatatypeEntryProperties

    def __init__(self, type: str, additional_properties: bool, required: List[str], properties: DatatypeEntryProperties) -> None:
        self.type = type
        self.additional_properties = additional_properties
        self.required = required
        self.properties = properties

    @staticmethod
    def from_dict(obj: Any) -> 'DatatypeEntry':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        additional_properties = from_bool(obj.get("additionalProperties"))
        required = from_list(from_str, obj.get("required"))
        properties = DatatypeEntryProperties.from_dict(obj.get("properties"))
        return DatatypeEntry(type, additional_properties, required, properties)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["additionalProperties"] = from_bool(self.additional_properties)
        result["required"] = from_list(from_str, self.required)
        result["properties"] = to_class(DatatypeEntryProperties, self.properties)
        return result


class Descendants:
    type: str
    default: bool

    def __init__(self, type: str, default: bool) -> None:
        self.type = type
        self.default = default

    @staticmethod
    def from_dict(obj: Any) -> 'Descendants':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        default = from_bool(obj.get("default"))
        return Descendants(type, default)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["default"] = from_bool(self.default)
        return result


class ItemsProperties:
    datatype: GlobalProperties
    descendants: Descendants

    def __init__(self, datatype: GlobalProperties, descendants: Descendants) -> None:
        self.datatype = datatype
        self.descendants = descendants

    @staticmethod
    def from_dict(obj: Any) -> 'ItemsProperties':
        assert isinstance(obj, dict)
        datatype = GlobalProperties.from_dict(obj.get("datatype"))
        descendants = Descendants.from_dict(obj.get("descendants"))
        return ItemsProperties(datatype, descendants)

    def to_dict(self) -> dict:
        result: dict = {}
        result["datatype"] = to_class(GlobalProperties, self.datatype)
        result["descendants"] = to_class(Descendants, self.descendants)
        return result


class Items:
    type: str
    required: List[str]
    properties: ItemsProperties

    def __init__(self, type: str, required: List[str], properties: ItemsProperties) -> None:
        self.type = type
        self.required = required
        self.properties = properties

    @staticmethod
    def from_dict(obj: Any) -> 'Items':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        required = from_list(from_str, obj.get("required"))
        properties = ItemsProperties.from_dict(obj.get("properties"))
        return Items(type, required, properties)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["required"] = from_list(from_str, self.required)
        result["properties"] = to_class(ItemsProperties, self.properties)
        return result


class Datatypes:
    type: str
    items: Items

    def __init__(self, type: str, items: Items) -> None:
        self.type = type
        self.items = items

    @staticmethod
    def from_dict(obj: Any) -> 'Datatypes':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        items = Items.from_dict(obj.get("items"))
        return Datatypes(type, items)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["items"] = to_class(Items, self.items)
        return result


class TypeClass:
    type: str
    enum: List[str]

    def __init__(self, type: str, enum: List[str]) -> None:
        self.type = type
        self.enum = enum

    @staticmethod
    def from_dict(obj: Any) -> 'TypeClass':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        enum = from_list(from_str, obj.get("enum"))
        return TypeClass(type, enum)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["enum"] = from_list(from_str, self.enum)
        return result


class InputProperties:
    name: PipelineName
    type: TypeClass
    accepted_datatypes: Datatypes

    def __init__(self, name: PipelineName, type: TypeClass, accepted_datatypes: Datatypes) -> None:
        self.name = name
        self.type = type
        self.accepted_datatypes = accepted_datatypes

    @staticmethod
    def from_dict(obj: Any) -> 'InputProperties':
        assert isinstance(obj, dict)
        name = PipelineName.from_dict(obj.get("name"))
        type = TypeClass.from_dict(obj.get("type"))
        accepted_datatypes = Datatypes.from_dict(obj.get("acceptedDatatypes"))
        return InputProperties(name, type, accepted_datatypes)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = to_class(PipelineName, self.name)
        result["type"] = to_class(TypeClass, self.type)
        result["acceptedDatatypes"] = to_class(Datatypes, self.accepted_datatypes)
        return result


class Input:
    type: str
    required: List[str]
    properties: InputProperties

    def __init__(self, type: str, required: List[str], properties: InputProperties) -> None:
        self.type = type
        self.required = required
        self.properties = properties

    @staticmethod
    def from_dict(obj: Any) -> 'Input':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        required = from_list(from_str, obj.get("required"))
        properties = InputProperties.from_dict(obj.get("properties"))
        return Input(type, required, properties)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["required"] = from_list(from_str, self.required)
        result["properties"] = to_class(InputProperties, self.properties)
        return result


class PropertiesProperties:
    pass

    def __init__(self, ) -> None:
        pass

    @staticmethod
    def from_dict(obj: Any) -> 'PropertiesProperties':
        assert isinstance(obj, dict)
        return PropertiesProperties()

    def to_dict(self) -> dict:
        result: dict = {}
        return result


class NodeObjInfoProperties:
    name: PipelineName
    id: GlobalProperties
    properties: PropertiesProperties

    def __init__(self, name: PipelineName, id: GlobalProperties, properties: PropertiesProperties) -> None:
        self.name = name
        self.id = id
        self.properties = properties

    @staticmethod
    def from_dict(obj: Any) -> 'NodeObjInfoProperties':
        assert isinstance(obj, dict)
        name = PipelineName.from_dict(obj.get("name"))
        id = GlobalProperties.from_dict(obj.get("id"))
        properties = PropertiesProperties.from_dict(obj.get("properties"))
        return NodeObjInfoProperties(name, id, properties)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = to_class(PipelineName, self.name)
        result["id"] = to_class(GlobalProperties, self.id)
        result["properties"] = to_class(PropertiesProperties, self.properties)
        return result


class NodeObjInfo:
    type: str
    required: List[str]
    properties: NodeObjInfoProperties

    def __init__(self, type: str, required: List[str], properties: NodeObjInfoProperties) -> None:
        self.type = type
        self.required = required
        self.properties = properties

    @staticmethod
    def from_dict(obj: Any) -> 'NodeObjInfo':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        required = from_list(from_str, obj.get("required"))
        properties = NodeObjInfoProperties.from_dict(obj.get("properties"))
        return NodeObjInfo(type, required, properties)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["required"] = from_list(from_str, self.required)
        result["properties"] = to_class(NodeObjInfoProperties, self.properties)
        return result


class OutputProperties:
    name: PipelineName
    type: TypeClass
    possible_datatypes: Datatypes

    def __init__(self, name: PipelineName, type: TypeClass, possible_datatypes: Datatypes) -> None:
        self.name = name
        self.type = type
        self.possible_datatypes = possible_datatypes

    @staticmethod
    def from_dict(obj: Any) -> 'OutputProperties':
        assert isinstance(obj, dict)
        name = PipelineName.from_dict(obj.get("name"))
        type = TypeClass.from_dict(obj.get("type"))
        possible_datatypes = Datatypes.from_dict(obj.get("possibleDatatypes"))
        return OutputProperties(name, type, possible_datatypes)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = to_class(PipelineName, self.name)
        result["type"] = to_class(TypeClass, self.type)
        result["possibleDatatypes"] = to_class(Datatypes, self.possible_datatypes)
        return result


class Output:
    type: str
    required: List[str]
    properties: OutputProperties

    def __init__(self, type: str, required: List[str], properties: OutputProperties) -> None:
        self.type = type
        self.required = required
        self.properties = properties

    @staticmethod
    def from_dict(obj: Any) -> 'Output':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        required = from_list(from_str, obj.get("required"))
        properties = OutputProperties.from_dict(obj.get("properties"))
        return Output(type, required, properties)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["required"] = from_list(from_str, self.required)
        result["properties"] = to_class(OutputProperties, self.properties)
        return result


class Definitions:
    datatype_entry: Optional[DatatypeEntry]
    output: Optional[Output]
    input: Optional[Input]
    node_obj_info: Optional[NodeObjInfo]

    def __init__(self, datatype_entry: Optional[DatatypeEntry], output: Optional[Output], input: Optional[Input], node_obj_info: Optional[NodeObjInfo]) -> None:
        self.datatype_entry = datatype_entry
        self.output = output
        self.input = input
        self.node_obj_info = node_obj_info

    @staticmethod
    def from_dict(obj: Any) -> 'Definitions':
        assert isinstance(obj, dict)
        datatype_entry = from_union([DatatypeEntry.from_dict, from_none], obj.get("DatatypeEntry"))
        output = from_union([Output.from_dict, from_none], obj.get("Output"))
        input = from_union([Input.from_dict, from_none], obj.get("Input"))
        node_obj_info = from_union([NodeObjInfo.from_dict, from_none], obj.get("NodeObjInfo"))
        return Definitions(datatype_entry, output, input, node_obj_info)

    def to_dict(self) -> dict:
        result: dict = {}
        result["DatatypeEntry"] = from_union([lambda x: to_class(DatatypeEntry, x), from_none], self.datatype_entry)
        result["Output"] = from_union([lambda x: to_class(Output, x), from_none], self.output)
        result["Input"] = from_union([lambda x: to_class(Input, x), from_none], self.input)
        result["NodeObjInfo"] = from_union([lambda x: to_class(NodeObjInfo, x), from_none], self.node_obj_info)
        return result


class Node1Output:
    type: str
    min_length: int

    def __init__(self, type: str, min_length: int) -> None:
        self.type = type
        self.min_length = min_length

    @staticmethod
    def from_dict(obj: Any) -> 'Node1Output':
        assert isinstance(obj, dict)
        type = from_str(obj.get("type"))
        min_length = from_int(obj.get("minLength"))
        return Node1Output(type, min_length)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = from_str(self.type)
        result["minLength"] = from_int(self.min_length)
        return result


class SchemaProperties:
    hierarchies: Optional[Connections]
    name: Optional[PipelineName]
    description: Optional[PipelineName]
    category: Optional[PipelineName]
    properties: Optional[Node1Output]
    outputs: Optional[Connections]
    inputs: Optional[Connections]
    node1_id: Optional[GlobalProperties]
    node2_id: Optional[GlobalProperties]
    node1_output: Optional[Node1Output]
    node2_input: Optional[Node1Output]
    global_properties: Optional[GlobalProperties]
    nodes: Optional[Connections]
    connections: Optional[Connections]

    def __init__(self, hierarchies: Optional[Connections], name: Optional[PipelineName], description: Optional[PipelineName], category: Optional[PipelineName], properties: Optional[Node1Output], outputs: Optional[Connections], inputs: Optional[Connections], node1_id: Optional[GlobalProperties], node2_id: Optional[GlobalProperties], node1_output: Optional[Node1Output], node2_input: Optional[Node1Output], global_properties: Optional[GlobalProperties], nodes: Optional[Connections], connections: Optional[Connections]) -> None:
        self.hierarchies = hierarchies
        self.name = name
        self.description = description
        self.category = category
        self.properties = properties
        self.outputs = outputs
        self.inputs = inputs
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.node1_output = node1_output
        self.node2_input = node2_input
        self.global_properties = global_properties
        self.nodes = nodes
        self.connections = connections

    @staticmethod
    def from_dict(obj: Any) -> 'SchemaProperties':
        assert isinstance(obj, dict)
        hierarchies = from_union([Connections.from_dict, from_none], obj.get("hierarchies"))
        name = from_union([PipelineName.from_dict, from_none], obj.get("name"))
        description = from_union([PipelineName.from_dict, from_none], obj.get("description"))
        category = from_union([PipelineName.from_dict, from_none], obj.get("category"))
        properties = from_union([Node1Output.from_dict, from_none], obj.get("properties"))
        outputs = from_union([Connections.from_dict, from_none], obj.get("outputs"))
        inputs = from_union([Connections.from_dict, from_none], obj.get("inputs"))
        node1_id = from_union([GlobalProperties.from_dict, from_none], obj.get("node1Id"))
        node2_id = from_union([GlobalProperties.from_dict, from_none], obj.get("node2Id"))
        node1_output = from_union([Node1Output.from_dict, from_none], obj.get("node1Output"))
        node2_input = from_union([Node1Output.from_dict, from_none], obj.get("node2Input"))
        global_properties = from_union([GlobalProperties.from_dict, from_none], obj.get("globalProperties"))
        nodes = from_union([Connections.from_dict, from_none], obj.get("nodes"))
        connections = from_union([Connections.from_dict, from_none], obj.get("connections"))
        return SchemaProperties(hierarchies, name, description, category, properties, outputs, inputs, node1_id, node2_id, node1_output, node2_input, global_properties, nodes, connections)

    def to_dict(self) -> dict:
        result: dict = {}
        result["hierarchies"] = from_union([lambda x: to_class(Connections, x), from_none], self.hierarchies)
        result["name"] = from_union([lambda x: to_class(PipelineName, x), from_none], self.name)
        result["description"] = from_union([lambda x: to_class(PipelineName, x), from_none], self.description)
        result["category"] = from_union([lambda x: to_class(PipelineName, x), from_none], self.category)
        result["properties"] = from_union([lambda x: to_class(Node1Output, x), from_none], self.properties)
        result["outputs"] = from_union([lambda x: to_class(Connections, x), from_none], self.outputs)
        result["inputs"] = from_union([lambda x: to_class(Connections, x), from_none], self.inputs)
        result["node1Id"] = from_union([lambda x: to_class(GlobalProperties, x), from_none], self.node1_id)
        result["node2Id"] = from_union([lambda x: to_class(GlobalProperties, x), from_none], self.node2_id)
        result["node1Output"] = from_union([lambda x: to_class(Node1Output, x), from_none], self.node1_output)
        result["node2Input"] = from_union([lambda x: to_class(Node1Output, x), from_none], self.node2_input)
        result["globalProperties"] = from_union([lambda x: to_class(GlobalProperties, x), from_none], self.global_properties)
        result["nodes"] = from_union([lambda x: to_class(Connections, x), from_none], self.nodes)
        result["connections"] = from_union([lambda x: to_class(Connections, x), from_none], self.connections)
        return result


class Schema:
    id: str
    schema: str
    title: str
    description: str
    type: str
    required: Optional[List[str]]
    definitions: Optional[Definitions]
    properties: Optional[SchemaProperties]
    minimum: Optional[int]

    def __init__(self, id: str, schema: str, title: str, description: str, type: str, required: Optional[List[str]], definitions: Optional[Definitions], properties: Optional[SchemaProperties], minimum: Optional[int]) -> None:
        self.id = id
        self.schema = schema
        self.title = title
        self.description = description
        self.type = type
        self.required = required
        self.definitions = definitions
        self.properties = properties
        self.minimum = minimum

    @staticmethod
    def from_dict(obj: Any) -> 'Schema':
        assert isinstance(obj, dict)
        id = from_str(obj.get("$id"))
        schema = from_str(obj.get("$schema"))
        title = from_str(obj.get("title"))
        description = from_str(obj.get("description"))
        type = from_str(obj.get("type"))
        required = from_union([lambda x: from_list(from_str, x), from_none], obj.get("required"))
        definitions = from_union([Definitions.from_dict, from_none], obj.get("definitions"))
        properties = from_union([SchemaProperties.from_dict, from_none], obj.get("properties"))
        minimum = from_union([from_int, from_none], obj.get("minimum"))
        return Schema(id, schema, title, description, type, required, definitions, properties, minimum)

    def to_dict(self) -> dict:
        result: dict = {}
        result["$id"] = from_str(self.id)
        result["$schema"] = from_str(self.schema)
        result["title"] = from_str(self.title)
        result["description"] = from_str(self.description)
        result["type"] = from_str(self.type)
        result["required"] = from_union([lambda x: from_list(from_str, x), from_none], self.required)
        result["definitions"] = from_union([lambda x: to_class(Definitions, x), from_none], self.definitions)
        result["properties"] = from_union([lambda x: to_class(SchemaProperties, x), from_none], self.properties)
        result["minimum"] = from_union([from_int, from_none], self.minimum)
        return result


def common_from_dict(s: Any) -> Global:
    return Global.from_dict(s)


def common_to_dict(x: Global) -> Any:
    return to_class(Global, x)


def global_from_dict(s: Any) -> Global:
    return Global.from_dict(s)


def global_to_dict(x: Global) -> Any:
    return to_class(Global, x)


def schema_from_dict(s: Any) -> Schema:
    return Schema.from_dict(s)


def schema_to_dict(x: Schema) -> Any:
    return to_class(Schema, x)
