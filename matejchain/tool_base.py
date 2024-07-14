from abc import abstractmethod, ABC
from tool_param import ToolParam
from functools import cached_property


class ToolBase(ABC):
    def __init__(self, name: str, desc: str, params: list[ToolParam] | None = None):
        self.name = name
        self.desc = desc
        self.params = params

    @cached_property
    def to_openai(self):
        func_spec = {
            "name": self.name,
            "description": self.desc,
        }
        if self.params is not None:
            props = {}
            [props.update(par.to_openai()) for par in self.params]
            func_spec["parameters"] = {
                "type": "object",
                "properties": props,
                "required": [x.name for x in self.params if x.required is True],
            }
        return {"type": "function", "function": func_spec}

    @abstractmethod
    def exec(self, **kwargs):
        """
        Executes the tool with kwargs filled by LLM.
        This method has to take precisely the same names of args as are defined in "params"
        constructor arg.
        :param kwargs: filled by LLM, can be also no kwargs, to be defined in child class
        :return: any output, to be defined in child class
        """
        return
