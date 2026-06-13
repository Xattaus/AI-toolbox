# Abliteration ja Päättelykyvyn Palauttaminen

## Sisällysluettelo
1. [Ongelman kuvaus](#ongelman-kuvaus)
2. [Ratkaisustrategiat](#ratkaisustrategiat)
3. [Toteutussuunnitelma](#toteutussuunnitelma)
4. [Projektin konfiguraatio](#projektin-konfiguraatio)

---

## Ongelman kuvaus

### Abliteration ja sen sivuvaikutukset

Abliteration poistaa mallin kieltäytymismekanismin (refusal direction), mutta naiiivi toteutus voi aiheuttaa:

1. **Päättelykyvyn heikkeneminen** - Erityisesti matemaattinen päättely (GSM8K) on herkkä
2. **Koherenssin lasku** - Vastaukset voivat olla sekavia tai epäloogisia
3. **"Lobotomia-efekti"** - Liian aggressiivinen abliteration tuhoaa myös hyödyllisiä ominaisuuksia

> "Mathematical reasoning showed highest sensitivity to abliteration... any perturbation to intermediate representations can cascade."
> — [Comparative Analysis of LLM Abliteration Methods](https://arxiv.org/pdf/2512.13655)

### Poro-2-8B Abliteration tulokset

| Mittari | Tulos |
|---------|-------|
| Voimakkuus | 0.65 (dynaaminen avg 0.38) |
| Reasoning score | 60% (tavoite: 75%) |
| Tervehdys | ✓ |
| Transitiivinen päättely | ✓ |
| Matematiikka | ✗ |
| Logiikkapulma | ✗ |

---

## Ratkaisustrategiat

### Strategia 1: Model Merging (SUOSITELTU - Poro + DeepSeek-R1)

**Valittu malli:** [huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated](https://huggingface.co/huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated)

**Miksi tämä malli:**
- ✅ **Vahva päättely** - Distilloitu DeepSeek-R1:stä (SOTA reasoning)
- ✅ **Jo abliteroitu** - Ei kieltäytymiskonflikkteja mergessä
- ✅ **Sama arkkitehtuuri** - Llama-pohjainen → suoraan yhteensopiva Poron kanssa
- ✅ **Chain-of-thought** - Osaa ajatella askel askeleelta

**Merge-strategia:**
```
Poro-2-8B-abliterated (suomi, kulttuuri)
         +
DeepSeek-R1-Distill-Llama-8B-abliterated (päättely, matematiikka)
         =
Poro-R1-Merged (suomi + päättely)
```

**Suositellut merge-suhteet:**

| Suhde | Painotus | Käyttötapaus |
|-------|----------|--------------|
| 70/30 | 70% Poro, 30% DeepSeek | Suomi-painotteinen, parannettu päättely |
| 50/50 | Tasapaino | Hyvä yleismalli |
| 30/70 | 30% Poro, 70% DeepSeek | Päättely-painotteinen, säilytetty suomi |

### Strategia 2: DPO Healing (Vaihtoehto)

**Miten toimii:** Abliteroitu malli "parannetaan" DPO (Direct Preference Optimization) hienosäädöllä.

**Esimerkki:** [NeuralDaredevil-8B-abliterated](https://huggingface.co/mlabonne/NeuralDaredevil-8B-abliterated)
- Abliteroitu Daredevil-8B
- Parannettu DPO:lla käyttäen `orpo-dpo-mix-40k` datasettia
- Tulokset: 69.1% MMLU, 71.8% GSM8K, 85.05% HellaSwag

**Tarvittavat resurssit:**
- Dataset: [mlabonne/orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)
- Työkalu: Axolotl tai TRL
- GPU: 24GB+ VRAM (tai LoRA pienemmällä)

### Strategia 3: Parempi Abliteration (Tulevaisuus)

**Norm-Preserving Biprojected Abliteration:**
- Säilyttää normin → parempi reasoning
- [HuggingFace Blog](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)

**Gabliteration (Layer-Dependent Weights):**
- Eri voimakkuus eri kerroksille
- Vahva keskellä, heikkenee reunoille

---

## Toteutussuunnitelma

### Vaihe 1: Lataa DeepSeek-R1-Distill-Llama-8B-abliterated

```bash
# Käyttäen AI Toolbox TUI:ta
python -m ai_toolbox
# Model Hub -> Lataa HuggingFacesta -> huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated

# TAI suoraan huggingface-cli:llä
huggingface-cli download huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated
```

### Vaihe 2: SLERP Merge

**Konfiguraatiotiedosto: `merge_poro_deepseek.yaml`**

```yaml
# merge_poro_deepseek.yaml
# SLERP merge: Poro (suomi) + DeepSeek-R1 (päättely)

models:
  - model: ./models/abliterated/Poro-2-8B-abliterated-v2
    parameters:
      weight: 1.0
  - model: huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated
    parameters:
      weight: 1.0

merge_method: slerp
base_model: ./models/abliterated/Poro-2-8B-abliterated-v2

parameters:
  t: 0.3  # 0.0 = 100% Poro, 1.0 = 100% DeepSeek
          # 0.3 = 70% Poro, 30% DeepSeek (suositeltu)

dtype: bfloat16
```

**Suoritus:**
```bash
# Asenna mergekit
pip install mergekit

# Suorita merge
mergekit-yaml merge_poro_deepseek.yaml ./models/merged/Poro-R1-Merged --cuda
```

### Vaihe 3: TIES-DARE Merge (Vaihtoehto)

TIES-DARE säilyttää paremmin molempien mallien erikoisominaisuudet:

```yaml
# merge_poro_deepseek_ties.yaml
models:
  - model: ./models/abliterated/Poro-2-8B-abliterated-v2
    parameters:
      weight: 0.7
      density: 0.7  # DARE: pudota 30% parametreista
  - model: huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated
    parameters:
      weight: 0.3
      density: 0.7

merge_method: dare_ties
base_model: meta-llama/Meta-Llama-3.1-8B  # Yhteinen base

parameters:
  int8_mask: true
  normalize: true

dtype: bfloat16
```

### Vaihe 4: Testaus

```python
# test_merged_model.py
test_prompts = [
    # Suomen kieli
    ("Moikka! Kerro itsestäsi.", "suomi"),

    # Transitiivinen päättely
    ("Matti on Pekan isä. Pekka on Jonin isä. Kuka on Jonin isoisä?",
     "matti" tai "isoisä"),

    # Matematiikka
    ("Milla on 5 vuotta vanhempi kuin Liisa. Liisa on 8. Kuinka vanha Milla on?",
     "13"),

    # Logiikka
    ("Isäni ainoan lapsen nimi on Ville. Kuka minä olen?",
     "ville"),

    # Chain-of-thought
    ("Ajattele askel askeleelta: Jos junalla kestää 2 tuntia kulkea 120 km, mikä on junan nopeus?",
     "60"),
]
```

---

## Projektin Konfiguraatio

### Kriittiset tiedostot

```
src/ai_toolbox/abliteration/
├── abliterator.py          # Pää-abliteraattori
│   ├── AbliterationConfig  # Kaikki asetukset
│   └── Abliterator         # Pääluokka

src/ai_toolbox/merging/      # TODO: Lisää merge-tuki
├── merger.py               # Model merger
└── configs/                # Merge-konfiguraatiot
```

### AbliterationConfig - Tärkeimmät asetukset

```python
@dataclass
class AbliterationConfig:
    # PERUS
    model_path: str                    # Mallin polku
    output_name: str                   # Tulosteen nimi
    strength: float = 0.5             # Voimakkuus (0.0-1.0)
    method: str = "gradient"          # mean_diff, pca, gradient

    # OFFLOAD MODE - KRIITTINEN NOPEUDELLE!
    offload_mode: str = "auto"
    # Vaihtoehdot:
    # - "gpu_only"       : Koko malli GPU:lle (16GB+ VRAM)
    # - "sequential_cpu" : CPU-tallennus, GPU-laskenta (8GB riittää, NOPEA)
    # - "auto"           : Transformers päättää (8GB riittää, HIDAS)

    # AUTO-TUNE
    use_auto_tune: bool = True
    auto_tune_max_iterations: int = 5
    auto_tune_test_prompts: int = 4

    # REASONING VALIDATION
    use_reasoning_validation: bool = True
    reasoning_min_score: float = 0.6
    reasoning_strength_reduction: float = 0.15
    reasoning_min_strength: float = 0.15
    reasoning_max_retries: int = 5

    # SMART LAYERS
    use_smart_layers: bool = True
    layer_signal_threshold: float = 0.5

    # DYNAMIC STRENGTH
    use_dynamic_strength: bool = True

    # LINEAR PROBE
    use_linear_probe: bool = True
    probe_accuracy_threshold: float = 0.85
```

### Offload Mode vertailu

| Mode | VRAM | Nopeus | Selitys |
|------|------|--------|---------|
| `gpu_only` | 16GB+ | ⚡⚡⚡ | Koko malli GPU:lla |
| `sequential_cpu` | 8GB | ⚡⚡ | Kerros kerrallaan GPU:lle, laskenta GPU:lla |
| `auto` | 8GB | ⚡ | Osa CPU:lla, osa GPU:lla, CPU-osat lasketaan CPU:lla |

**HUOM:** `sequential_cpu` on paras vaihtoehto kun GPU-muisti ei riitä, koska kaikki laskenta tapahtuu silti GPU:lla!

### Reasoning-testit (suomi)

```python
FINNISH_REASONING_TESTS = [
    # Transitiivinen päättely
    ("Matti on Pekan isä. Pekka on Jonin isä. Kuka on Jonin isoisä?",
     ["matti", "isoisä"]),

    # Aritmetiikka
    ("Milla on 5 vuotta vanhempi kuin Liisa. Liisa on 8. Kuinka vanha Milla on?",
     ["13", "kolmetoista"]),

    # Logiikka
    ("Isäni ainoan lapsen nimi on Ville. Kuka minä olen?",
     ["ville"]),
]
```

---

## Merge-metodit selitettynä

### SLERP (Spherical Linear Interpolation)

```
Malli A ----○----○----○----○---- Malli B
            ^
            |
        Interpoloitu malli (t=0.3 → 70% A, 30% B)
```

- Interpoloi painoja hyperpallolla
- Parempi kuin lineaarinen keskiarvo
- Sopii 2 mallin yhdistämiseen

### TIES (Trim, Elect Sign & Merge)

1. **TRIM** - Poista pienet muutokset (redundantit)
2. **ELECT SIGN** - Valitse etumerkki äänestämällä
3. **MERGE** - Yhdistä vain yhtenevät parametrit

### DARE (Drop And REscale)

1. **DROP** - Pudota X% parametreista satunnaisesti
2. **RESCALE** - Skaalaa loput ylös: `weight / (1 - drop_rate)`

DARE + TIES yhdessä on tehokas tapa yhdistää erilaisia kykyjä.

---

## Lähteet

- [huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated](https://huggingface.co/huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated)
- [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration) - Maxime Labonne
- [Norm-Preserving Biprojected Abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)
- [Merge Large Language Models with mergekit](https://huggingface.co/blog/mlabonne/merge-models)
- [NVIDIA: Introduction to Model Merging](https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/)
- [Model Merging Survey](https://cameronrwolfe.substack.com/p/model-merging)
- [Comparative Analysis of LLM Abliteration Methods](https://arxiv.org/pdf/2512.13655)

---

## Päivityshistoria

| Päivä | Muutos |
|-------|--------|
| 2025-01-29 | Ensimmäinen versio - Poro + DeepSeek-R1 merge suunnitelma |
