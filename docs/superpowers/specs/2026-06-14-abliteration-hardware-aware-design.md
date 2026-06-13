# Laitteistotietoinen abliterointi — suunnitteludokumentti

**Päivä:** 2026-06-14
**Tila:** Hyväksytty (käyttäjä)
**Laajuus:** Vain abliteration-moduuli (Full Abliteration -flow)

## Ongelma

Full Abliteration -flow kaatuu suurilla malleilla Windowsin virheeseen
`os error 1455` (`ERROR_COMMITMENT_LIMIT`, "Sivutustiedosto on liian pieni").
Syy: malli + valitut lisäasetukset (esim. auto-tune lataa mallin uudelleen,
capability preservation, batch_size 8, 8.8B-malli fp16) vaativat enemmän
*commit-muistia* (RAM + sivutustiedosto) kuin järjestelmässä on varattavissa.
Kaatuminen tapahtuu vasta latausvaiheessa, kun käyttäjä on jo syöttänyt kaikki
asetukset.

Nykyinen `Abliterator.estimate_requirements()` (abliterator.py:3159) tuottaa vain
karkean arvion mallin koosta. Se **ei**:
- lue koneen todellista RAM/VRAM/sivutustiedostoa,
- huomioi lisäasetusten (auto-tune, batch, capability/direction) muistipiikkiä,
- vertaa arviota saatavilla olevaan muistiin ennen ajoa.

## Tavoite

Tunnistaa käyttäjän laitteisto, suositella sen perusteella järkevät oletukset,
ja varoittaa **ennen** mallin latausta jos konfiguraatio ylittäisi muistin —
poistaen 1455-virheen juurisyyn.

## Hyväksytyt päätökset

1. **Laajuus:** Suosittele oletukset **ja** varoita ennakkoon (pre-flight).
2. **Ylityksen sattuessa:** Varoita + tarjoa kevennetty "turvallinen" profiili,
   jonka voi hyväksyä yhdellä valinnalla. Käyttäjä saa silti ohittaa ja jatkaa.
3. **Sivutustiedosto:** Tarjoa **automaattinen säätö** (admin + uudelleenkäynnistys)
   eksplisiittisellä suostumuksella; fallback manuaaliohjeisiin jos UAC perutaan.
4. **Sijoitus:** Koodi abliteration-moduulissa (`abliteration/hardware.py`),
   integroitu vain Full Abliteration -flowhun.

## Arkkitehtuuri

### Uusi tiedosto: `src/ai_toolbox/abliteration/hardware.py`

Puhtaita funktioita + dataclasseja, ei Rich-tulostusta eikä questionarya
(CLI-kerros hoitaa käyttäjävuorovaikutuksen). Ei uusia riippuvuuksia —
`psutil` ja `torch` ovat jo käytössä.

#### Dataclassit

```python
@dataclass
class HardwareProfile:
    total_ram_gb: float
    available_ram_gb: float
    pagefile_total_gb: float        # psutil.swap_memory().total
    pagefile_free_gb: float
    cuda_available: bool
    gpu_name: Optional[str]
    vram_total_gb: Optional[float]
    vram_free_gb: Optional[float]
    # Käytettävissä oleva commit-budjetti = available_ram + pagefile_free
    @property
    def commit_budget_gb(self) -> float: ...

@dataclass
class MemoryEstimate:
    peak_commit_gb: float           # RAM + pagefile -tarve (1455:n mittari)
    peak_vram_gb: float
    breakdown: dict[str, float]     # erien erittely (selitettävyys)

@dataclass
class RecommendedSettings:
    offload_mode: str               # auto | sequential_cpu | sequential_disk
    batch_size: int
    enable_auto_tune: bool
    notes: list[str]                # miksi näin (näytetään käyttäjälle)

@dataclass
class PreflightResult:
    status: str                     # "ok" | "warn" | "fail"
    bottleneck: Optional[str]       # "ram" | "pagefile" | "vram" | None
    shortfall_gb: float
    safe_profile: Optional[RecommendedSettings]
    recommended_pagefile_gb: Optional[int]   # asetettu jos bottleneck=pagefile
    message: str
```

#### Funktiot

- `detect_hardware() -> HardwareProfile`
  - RAM: `psutil.virtual_memory()`
  - Pagefile: `psutil.swap_memory()` (Windowsilla = page file)
  - VRAM/GPU: `torch.cuda.is_available()`, `torch.cuda.mem_get_info()`,
    `torch.cuda.get_device_properties(0)`
  - **Kaikki kyselyt try/except-suojattu.** Puuttuva kirjasto/kysely →
    osittainen profiili (kentät None / 0.0). Ei koskaan nosta poikkeusta.

- `estimate_cost(model_info: dict, config: AbliterationConfig) -> MemoryEstimate`
  - Pohjana fp16-painot (`params_b * 2 GB`).
  - Lisät: auto-tune → ×2 mallilataus; batch_size → aktivaatiokomponentti;
    capability preservation / direction selection → kiinteä overhead-lisä;
    offload-tila vaikuttaa siihen kuinka paljon valuu pagefileen vs VRAMiin.
  - `breakdown` säilyttää erät, jotta paneeli voi selittää mistä piikki tulee.

- `recommend_config(profile, model_info) -> RecommendedSettings`
  - VRAM riittää koko malliin → offload `auto`, batch 2–4.
  - VRAM rajallinen → `sequential_cpu`, batch 1–2.
  - Pieni RAM/pagefile → auto-tune oletuksena pois (poistaa ×2-piikin).
  - `notes` kertoo perustelut käyttäjälle.

- `check_preflight(profile, estimate) -> PreflightResult`
  - `peak_commit_gb` vs `commit_budget_gb` → ram/pagefile-pullonkaula.
  - `peak_vram_gb` vs `vram_free_gb` → vram-pullonkaula.
  - Rajat: ok (mahtuu marginaalilla), warn (mahtuu mutta tiukka),
    fail (ei mahdu). Marginaali esim. 10 %.
  - fail/warn → täyttää `safe_profile`-kentän (`recommend_config` tiukemmilla
    rajoilla) ja `recommended_pagefile_gb`-kentän jos bottleneck=pagefile.

#### Pagefile-apurit (Windows-only)

- `get_pagefile_settings() -> dict` — nykyinen alku/maksimikoko (WMI
  `Win32_PageFileSetting` / `Win32_PageFileUsage` CIM-kyselyllä).
- `recommend_pagefile_gb(estimate) -> tuple[int, int]` — (alku, maksimi) GB,
  riittävä `peak_commit_gb`:lle marginaalilla.
- `apply_pagefile_setting(initial_gb, max_gb) -> bool` — rakentaa elevoidun
  PowerShell-komennon ja ajaa `Start-Process powershell -Verb RunAs` (UAC).
  Asettaa `AutomaticManagedPagefile=$false` + `Win32_PageFileSetting`
  alku/maksimi. **Ei käynnistä konetta uudelleen** — palauttaa True ja kehottaa
  käynnistämään. Vahti: vain `platform.system() == "Windows"`; UAC peruttu →
  False → CLI näyttää manuaaliohjeet.

### Integraatio: `src/ai_toolbox/cli/abliteration_cmd.py`

CLI vain orkestroi ja tulostaa; logiikka tulee `hardware.py`:stä.

1. **Flown alussa** (mallin valinnan jälkeen): `detect_hardware()` kerran →
   "🖥️ Laitteisto havaittu" -paneeli (GPU-nimi, VRAM total/free, RAM
   total/available, pagefile total/free). Brändin oranssi tyyli, `console`
   core/ui:sta.

2. **Promptien oletukset:** olemassa olevat strength/offload/batch/auto-tune
   -kysymykset säilyvät. `recommend_config(profile, model_info)` tuottaa niiden
   **default-arvot** (esim. `questionary ... default=rec.batch_size`).
   `rec.notes` näytetään lyhyenä dim-vihjeenä. Ei uusia kysymyksiä.

3. **Ennen "Aloitetaanko abliteration?"** (nykyinen summary ~rivi 1206):
   - `estimate = estimate_cost(model_info, config)`
   - `pf = check_preflight(profile, estimate)`
   - `status == "ok"` → jatka normaalisti.
   - `warn`/`fail` → varoituspaneeli (`pf.message`, `bottleneck`, `shortfall_gb`,
     `estimate.breakdown`) + questionary-valikko:
     - **Käytä turvallista profiilia** → sovella `pf.safe_profile`, päivitä
       config, aja preflight uudelleen.
     - **Säädä pagefile automaattisesti** (näkyy vain jos bottleneck=pagefile)
       → vahvistus → `apply_pagefile_setting(...)` → uudelleenkäynnistysmuistutus.
     - **Jatka silti** → etene ohittaen varoitus.
     - **Peruuta** → palaa valikkoon.

### Data flow

```
mallin valinta
  → detect_hardware() ─────────────► HardwareProfile (paneeli)
  → recommend_config() ────────────► promptien defaultit
  → käyttäjä syöttää asetukset → AbliterationConfig
  → estimate_cost(config) ─────────► MemoryEstimate
  → check_preflight() ─────────────► PreflightResult
        ok → aja
        warn/fail → valikko (turvallinen profiili / pagefile / jatka / peruuta)
```

## Virheenkäsittely

- **Detection ei koskaan blokkaa.** Jos `detect_hardware()` epäonnistuu osittain,
  paneeli näyttää "ei saatavilla" puuttuville kentille ja preflight ohitetaan
  siltä osin kuin dataa puuttuu (degrade to current behavior).
- **Auto-säätö ei käynnistä konetta** — vain kehottaa uudelleenkäynnistykseen.
- **UAC peruttu / ei admin** → fallback manuaaliohjeet (nykyinen page file -opas).
- **Ei-Windows** → pagefile-osio piilotetaan; swap luetaan silti psutililla
  (informatiivinen), mutta auto-säätöä ei tarjota.

## Testaus

Yksikkötestit (`tests/abliteration/test_hardware.py`), mockattu `HardwareProfile`
— ei oikeaa rautaa eikä mallilatauksia:

- `estimate_cost`: tunnettu model_info + config → odotettu `peak_commit_gb`
  (mm. auto-tune ×2 -kerroin, batch-skaalaus, capability-overhead).
- `recommend_config`: useita profiileja → odotetut offload/batch/auto-tune.
- `check_preflight`: ok/warn/fail-rajat ja oikea `bottleneck`, `safe_profile`
  täytetty fail/warn-tilassa.
- Pagefile: `recommend_pagefile_gb` arvot ja `apply`-komennon PowerShell-merkkijonon
  rakennus (komento koostetaan oikein; itse `Start-Process`-kutsua ei ajeta).

`detect_hardware()` ja `apply_pagefile_setting()` (systeemikutsut) jätetään
yksikkötestien ulkopuolelle, mutta ne on Windows-vahdittu ja try/except-suojattu.

## Hyväksyntäkriteerit

- Annettu 1455-laukaissut konfiguraatio (8.8B + auto-tune + capability + batch 8)
  tuottaa `status="fail"` ennen latausta, oikealla bottleneckilla.
- Turvallinen profiili pudottaa konfiguraation `status="ok"`-tilaan.
- Pagefile-bottleneckissä tarjotaan oikea suosituskoko ja toimiva auto-säätö.
- Detection-virhe ei kaada flowta missään tilanteessa.
- Olemassa olevat abliteration-testit menevät edelleen läpi.
