from abc import abstractmethod, ABC
from pepechain.tool_param import ToolParam


class ToolBase(ABC):
    def __init__(self, name: str, desc: str, params: tuple[ToolParam, ...] | None = None):
        self.name = name
        self.desc = desc
        self.params = params
        self.openai_fmt = self._parse_to_openai_fmt()

    def __call__(self, **kwargs):
        return str(self._exec(**kwargs))

    def to_openai(self):
        return self.openai_fmt

    @abstractmethod
    def _exec(self, **kwargs):
        """
        Executes the tool with kwargs filled by LLM.
        This method has to take precisely the same names of args as are defined in "params"
        constructor arg.
        :param kwargs: filled by LLM, can be also no kwargs, to be defined in child class
        :return: any output, to be defined in child class
        """
        pass

    def _parse_to_openai_fmt(self):
        func_spec = {
            "name": self.name,
            "description": self.desc,
        }
        if self.params is not None:
            props = {}
            [props.update(par.openai_fmt) for par in self.params]
            func_spec["parameters"] = {
                "type": "object",
                "properties": props,
                "required": [x.name for x in self.params if x.required is True],
            }
        return {"type": "function", "function": func_spec}
