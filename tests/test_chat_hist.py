from matejchain.chat_hist import ChatHist
from matejchain.msg import SysMsg, UsrMsg, AssMsg
import pytest


@pytest.mark.parametrize(
    "sys_prompt", [SysMsg("You are useful assistant"), "You are useful assistant", None]
)
def test_chat_history(sys_prompt):
    # init
    h = ChatHist(4, sys_prompt)

    if sys_prompt is not None:
        expected_list = [
            SysMsg("You are useful assistant"),
            AssMsg("mmm"),
            UsrMsg("hehe2"),
            AssMsg("Hehehe"),
        ]
        assert len(h) == 1

    else:
        expected_list = [UsrMsg("Hehe"), AssMsg("mmm"), UsrMsg("hehe2"), AssMsg("Hehehe")]
        assert len(h) == 0

    # add many msgs
    h.add(UsrMsg("hey!"))
    h = h + AssMsg("Hello there!")
    h += UsrMsg("Hehe")
    h.add(AssMsg("mmm"))
    h = h + UsrMsg("hehe2")
    h += AssMsg("Hehehe")
    assert len(h) == 4

    # check content
    assert h.to_list() == expected_list
    assert h.openai_fmt() == [x.openai_fmt for x in expected_list]
