from mlib import human_readable_payload


def test_human_readable_payload():
    result = human_readable_payload(0.2, 0.5)
    assert "No Moroso" == result["Type_user"]

    result = human_readable_payload(0.7, 0.5)
    assert "Moroso" == result["Type_user"]
