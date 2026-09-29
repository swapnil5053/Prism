import uuid

from prism.api.workspace import sign, verify

SECRET = "s" * 40


def test_round_trip() -> None:
    ws = uuid.uuid4()
    assert verify(sign(ws, SECRET), SECRET) == ws


def test_rejects_other_secret() -> None:
    assert verify(sign(uuid.uuid4(), SECRET), "t" * 40) is None


def test_rejects_swapped_id() -> None:
    token = sign(uuid.uuid4(), SECRET)
    _, mac = token.split(".")
    assert verify(f"{uuid.uuid4().hex}.{mac}", SECRET) is None


def test_rejects_garbage() -> None:
    for bad in [None, "", "abc", "a.b.c", "not-a-uuid.sig", f"{uuid.uuid4().hex}."]:
        assert verify(bad, SECRET) is None
