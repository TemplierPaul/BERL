from berl import *
import berl

def test_logger():
    l = Logger()

    assert len(l._data) == 0

    l.add("add one")
    l.add_list(["add two", "add three"])

    for i in ["add one", "add two", "add three"]:
        assert i in l._data.keys()
        assert l._data[i] == []

    assert isinstance(l.__repr__(), str)

    l.log("test", 42)
    assert l._data["test"]==[42]
    l("test", 17)
    assert l._data["test"]==[42, 17]

    assert l.last("test") == 17
    
    d = l.export()
    assert isinstance(d, dict)
    assert d["test"] == 17
    assert "add one" not in d.keys()

    d = {
        "test":31,
        "test 2": 32
    }
    l.log_dict(d)
    assert l._data["test"]==[42, 17,  31]
    assert l._data["test 2"]==[32]