PARAM_TYPE_MAP = {str: "string", int: "integer", bool: "boolean", float: "number"}


class ToolParam:
    def __init__(self, name, desc, dtype, required=True, enum=None):
        if dtype not in PARAM_TYPE_MAP:
            raise ValueError(f"Tool parameter type has to be one of: {PARAM_TYPE_MAP.keys()}")
        self.name = name
        self.desc = desc
        self.dtype = PARAM_TYPE_MAP[dtype]
        self.required = required
        self.enum = enum
        self.openai_fmt = self._parse_to_openai_fmt()

    def to_openai(self):
        return self.openai_fmt

    def _parse_to_openai_fmt(self):
        properties_dict = {
            self.name: {
                "type": self.dtype,
                "description": self.desc,
            }
        }
        if self.enum is not None:
            properties_dict["enum"] = [str(x) for x in self.enum]
        return properties_dict
