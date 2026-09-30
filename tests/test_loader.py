from src.data.loader import parse_tskv


def test_parse_tskv(tmp_path):
    p = tmp_path / "mini.tskv"
    p.write_text(
        "tskv\taddress=A 1\tname_ru=Кафе\trubrics=Кафе;Бар\trating=5.000000\ttext=Отлично\n"
        "tskv\taddress=B 2\tname_ru=Аптека\trubrics=Аптека\trating=1.000000\ttext=Плохо=совсем\n",
        encoding="utf-8",
    )
    df = parse_tskv(p)
    assert list(df.columns) == ["address", "name_ru", "rubrics", "rating", "text"]
    assert len(df) == 2
    assert df.loc[0, "name_ru"] == "Кафе"
    assert df.loc[1, "text"] == "Плохо=совсем"  # '=' внутри значения не ломает разбор