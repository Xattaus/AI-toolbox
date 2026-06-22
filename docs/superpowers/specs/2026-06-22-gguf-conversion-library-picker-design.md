# GGUF-konversion kirjastovalitsin — suunnitteludokumentti

**Päivä:** 2026-06-22
**Tila:** Hyväksytty (käyttäjä: "korjaa tuo")
**Laajuus:** Vain GGUF-konversion mallinvalinta (`_select_model_for_conversion`)

## Ongelma

GGUF-konversion mallinvalinta (`cli/gguf_tools_cmd.py:_select_model_for_conversion`)
tarjoaa vain "Syötä polku manuaalisesti" + "Valitse latauksista". Se ei näytä
kirjastossa olevia muunnettavia (SafeTensors/PyTorch) malleja, joten käyttäjän on
naputeltava polku käsin. Picker on käytössä kahdessa kohdassa: `_convert_local_model`
(rivi 188) ja `_convert_and_quantize` (rivi 299).

Muut mallinvalintaflowt (abliteration, training, merger, benchmark, ollama) näyttävät
jo kirjaston. `_convert_from_library` (rivi 199) näyttää jo kirjaston `get_convertible_models()`-
metodilla — mutta `_select_model_for_conversion` ei.

## Päätökset

- Picker näyttää **vain kirjasto + lataukset** (ei manuaalipolkua) — käyttäjän valinta.
- Vain tämä yksi flow korjataan; toimivia abliteration/training-flowja ei kosketa
  (ei duplikaation poistoa tässä erässä). Datasetit eivät kuulu tähän.

## Ratkaisu

### Testattava pure-apuri (`cli/selection.py`)

```
build_conversion_choices(convertible_models, downloaded) -> list[dict]
```
- Rakentaa numeroidun valintalistan: kirjaston muunnettavat mallit (source="library")
  + ladatut HF-mallit (source="download").
- Palauttaa dictit `{path: Path, name: str, size_bytes: int, source: str}`.
- Pure-funktio (ei I/O:ta) → yksikkötestattavissa mockatuilla syötteillä.

### `_select_model_for_conversion` uudelleenkirjoitus

- Kerää muunnettavat: `library.get_convertible_models()` + `get_all_merged()`-suodatus
  (samoin kuin `_convert_from_library` tekee), dedup polun mukaan.
- `downloaded = downloader.list_downloaded()`.
- `items = build_conversion_choices(convertible, downloaded)`.
- Tyhjä → varoitus "lataa malli ensin", paluu None.
- Rich-taulukko (kirjasto/HF-sektiot, #, nimi, koko, lähde) + numerovalinta `0 = peruuta`.
- **Ei** manuaalipolku- eikä erillistä "valitse latauksista" -haaraa.

## Virheenkäsittely

- Tyhjä lista → selkeä viesti, ei kaadu.
- Valittu polku jonka `config.json` puuttuu: olemassa oleva tarkistus
  `_convert_local_model`-kohdassa (rivi 192) säilyy.

## Testaus

`tests/cli/test_selection.py`:
- `build_conversion_choices`: kirjasto + lataukset → oikea järjestys, source-tagit,
  poludat; tyhjät syötteet → tyhjä lista; vain-kirjasto / vain-lataukset.
Interaktiivinen taulukko+syöttö jää ohueksi kuoreksi, varmennetaan manuaalisesti ajossa.

## Hyväksyntäkriteerit

- GGUF Tools → Local Model → GGUF (ja Konvertoi & Kvantisoi) näyttää kirjaston
  muunnettavat mallit listana ilman polun naputtelua.
- Tyhjä kirjasto+lataukset → ohjeistava viesti, ei kaatumista.
