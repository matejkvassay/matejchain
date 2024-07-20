from matejchain.message import UsrMsg, SysMsg, ToolMsg, AssMsg


def test_user_message():
    msg = UsrMsg("Hello there!")
    openai_fmt = {"role": "user", "content": "Hello there!"}
    assert msg.is_usr_msg()
    assert not msg.is_tool_msg()
    assert not msg.is_ass_msg()
    assert not msg.is_sys_msg()
    assert msg.openai_param == openai_fmt
    assert msg.to_openai() == openai_fmt


def test_system_message():
    msg = SysMsg("Hello there!")
    openai_fmt = {"role": "system", "content": "Hello there!"}
    assert not msg.is_usr_msg()
    assert not msg.is_tool_msg()
    assert not msg.is_ass_msg()
    assert msg.is_sys_msg()
    assert msg.openai_param == openai_fmt
    assert msg.to_openai() == openai_fmt


def test_assistant_message():
    msg = AssMsg("Hello there!")
    openai_fmt = {"role": "assistant", "content": "Hello there!"}
    assert not msg.is_usr_msg()
    assert not msg.is_tool_msg()
    assert msg.is_ass_msg()
    assert not msg.is_sys_msg()
    assert msg.openai_param == openai_fmt
    assert msg.to_openai() == openai_fmt


def test_tool_message():
    msg = ToolMsg("Hello there!")
    openai_fmt = {"role": "tool", "content": "Hello there!"}
    assert not msg.is_usr_msg()
    assert msg.is_tool_msg()
    assert not msg.is_ass_msg()
    assert not msg.is_sys_msg()
    assert msg.openai_param == openai_fmt
    assert msg.to_openai() == openai_fmt
