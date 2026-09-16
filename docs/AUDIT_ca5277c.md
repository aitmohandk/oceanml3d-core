# Audit — mise à jour sur `ca5277c`

Reprise de l'audit sur l'état courant de `main` (`ca5277c`), cinq commits après `06c9a4a` sur lequel
la première version portait. Ce document est un **delta** : il dit ce qui a changé, ce qui reste
valide, et ce qui est nouveau. Les constats non mentionnés ici sont inchangés.

Méthode : lecture du code à `ca5277c`, `ruff 0.16.7`, comparaison ligne à ligne des fichiers visés
par les correctifs prévus.

---

## 1. Ce qui a bougé

| PR | Effet |
|---|---|
| #2 | `docs/gridded_models.md` versé, puis étendu (707 l.) |
| #3 → `dc01af5` | **MONAI `DiffusionModelUNet` devient le tronc par défaut** des modèles grillés. `unet2d.py` → alias déprécié ; `unet_nosc.py` (l'ancien) et `unet_monai.py` (167 l.) ; option `trunk: monai\|nosc`, `ablation=trunk_nosc` |
| #3 → `5948ee5` | **La famille toy / Lorenz / QG passe en réserve** : `oceanml3d/legacy/` (51 modules), `legacy/` pour les 20 pilotes racine, `config/legacy/` pour 67 presets. 143 renommages, aucun code supprimé |

Diff global `06c9a4a..ca5277c` : 314 fichiers, +2 145 / −845.

Les deux changements sont bien faits. `unet_monai.py` documente en tête les trois divergences
traitées explicitement (dropout absent de `DiffusionUNetResnetBlock`, exigence de divisibilité
spatiale, absence de `bilinear`), et `NOSCUNet.on_load_checkpoint` refuse un checkpoint entraîné
avec l'autre tronc au lieu de le charger de travers. `tests/test_monai_unet2d.py` ajoute 13 tests,
dont l'initialisation à zéro héritée de `zero_module`, épinglée pour qu'elle ne change pas en
silence. C'est le niveau de soin qu'on veut voir.

---

## 2. Ce qui est réglé depuis la première passe

* **`config/experiment/` est trié** : 11 presets grillés à la racine, 67 dans `config/legacy/experiment/`.
  C'était un item du lot 4.
* **`attention_levels` est testé** (`test_attention_lands_on_the_requested_level`, plus le rejet d'un
  niveau hors bornes et d'un nombre de têtes qui ne divise pas la largeur). C'était un trou du lot 3.
* **`test_model_smoke.py` est paramétré sur `trunk`.**

Inchangé : `ruff check .` passe, et le jeu étendu `E,F,I,B,UP` moins les trois règles ignorées donne
toujours exactement **168** violations.

---

## 3. Ce qui survit intact

**Les quatre blocages de performance et les neuf bugs de justesse sont tous encore là.** Les deux PR
ont touché le tronc et déplacé la famille toy ; elles n'ont approché ni la couche de données ni
l'inférence. Vérifié à `ca5277c` :

| Réf. | Fichier, ligne | État |
|---|---|---|
| P0-1 `_has_target` par patch | `data/patches.py:82-87` | identique — `self.da.isel(...).values` dans une compréhension sur tous les patchs |
| P0-2 `torch.cat(preds)` en RAM | `inference/predict.py` | identique |
| P0-3 reconstruction fausse sous DDP | `inference/predict.py` | identique — `reconstruct` indexe toujours par `enumerate` |
| P1-4 ~110 `open_dataset` | `data/open.py` | identique |
| P1-1 `prepare-obs` sur tous les rangs | `cli.py:118,130` | identique — toujours dans `main()`, jamais dans `prepare_data()` |
| P1-2 `norm_stats.json` jamais relu | `training/callbacks.py:33` | identique — un seul point d'écriture, aucune lecture dans tout le paquet |
| P1-3 `depth_index: 0` falsy | `data/open.py:61` | identique — `elif spec.depth_index and ...` |
| P1-4 `reindex` sans tolérance | `data/open.py:84,90` | identique |
| P1-5 dates manquantes silencieuses | `data/open.py:91` | identique |
| P2-1..4 | — | identiques |

Également inchangés : `batch/` (166 scripts, 113 pointant sur l'ancien chemin, 4 avec imports plats)
et `reports/` (39 Mo).

---

## 4. Constats nouveaux, liés au basculement MONAI

### N1 (P0) — le tronc par défaut est passé de ~18 M à ~224 M paramètres, sans validation à l'échelle

`config/model/nosc_unet.yaml` le dit lui-même : aux largeurs par défaut
`[64, 128, 256, 512, 1024]`, le tronc MONAI fait **~224 M paramètres contre ~18 M** pour le tronc
NOSC. Un facteur 12 à 13, appliqué **par défaut**, sans que les tailles de patch aient changé.

Ordre de grandeur pour `surface_currents_15m` (patch 560 × 1440, `batch_size: 1`) : ~0,9 Go de poids,
~1,8 Go d'états Adam, ~0,9 Go de gradients, soit **~3,6 Go avant la moindre activation** ; les
activations à pleine résolution ajoutent plusieurs gigaoctets (un seul tenseur à largeur 64 pèse
64 × 560 × 1440 × 4 o ≈ 206 Mo, et `num_res_blocks: 2` en empile beaucoup).

Or la validation consignée au CHANGELOG est `experiment=smoke training=debug` — **données
synthétiques, CPU, patchs 16 × 16**. Aucun run n'a été fait sur les deux tâches réelles. Compte tenu
de P0-1 et P0-2, qui empêchent déjà ces tâches d'aboutir, personne ne peut savoir aujourd'hui si le
tronc par défaut tient sur une RTX 8000.

**Ce n'est pas un argument contre MONAI** — c'est un argument pour que le lot 1 passe avant, et pour
que `num_res_blocks: 1` ou une largeur de moins soit envisagé comme défaut.

### N2 (P0) — le seul test de bout en bout ne peut pas échouer sur un modèle mort

`test_train_predict_export` n'affirme que deux choses :

```python
assert manifest.exists()
assert len(list((tmp_path / "daily").glob("smoke_*.nc"))) == 10
```

Il passe sur un produit intégralement à zéro. Et c'est précisément le risque introduit par MONAI :
`test_zero_initialisation_is_inherited_from_monai` établit que **le tronc neuf est exactement
l'application nulle** — `zero_module` sur la convolution de sortie et sur le `conv2` de chaque bloc
résiduel.

Le CHANGELOG donne au passage un signal à vérifier : le smoke rend `test/loss` **1,4724** avec le
tronc MONAI et **1,4712** avec le tronc NOSC. Deux architectures séparées par un facteur 12 en taille
qui s'accordent à 0,08 % près, c'est compatible avec l'hypothèse que ni l'une ni l'autre n'apprend
quoi que ce soit sur ces données synthétiques — autrement dit que le test ne discrimine rien.

**Correctif :** ajouter au test de bout en bout une assertion de non-trivialité — produit fini, écart-type
non nul, et perte inférieure à celle de `passthrough` sur le même jeu. Sans quoi le filet de sécurité
ne couvre que la plomberie de fichiers.

### N3 (P1) — `torch>=2.0` est sous-contraint, et le vrai plafond est invisible

`dependencies` déclare `torch>=2.0` et `monai>=1.5,<1.6`. Or monai 1.5.x épingle
`torch>=2.4.1,<2.7.0` : la fenêtre réelle du projet est **2.4.1 à 2.6.0**, et rien dans le fichier ne
le dit côté torch. Conséquence pratique : dans un environnement portant déjà torch ≥ 2.7,
`pip install -e .` **rétrograde torch en silence**, depuis l'index PyPI par défaut, c'est-à-dire en
tirant les wheels CUDA. Déclarer `torch>=2.4.1,<2.7` rend la contrainte lisible et fait échouer tôt.

### N4 (P1) — la recette CI du lot 0 est cassée par la dépendance dure à MONAI

Le `ci.yml` que j'ai écrit installe `torch` depuis l'index CPU **sans borne haute**, puis le paquet.
pip verra ensuite `monai<1.6` exiger `torch<2.7` et rétrogradera depuis PyPI — exactement ce que
l'index CPU cherchait à éviter. Il faut `pip install 'torch>=2.4.1,<2.7' --index-url .../cpu`. Même
correction pour `environment.yml`, qui exclut encore MONAI au motif qu'il tire torch en avant : vrai
pour `>=1.6`, faux pour `1.5.x`, donc le fichier ne permet pas d'installer le dépôt.

### N5 (P2) — coût CPU du tronc par défaut en CI

`DiffusionModelUNet` construit toujours son MLP d'embedding temporel, même alimenté par un vecteur de
zéros, et porte de l'attention par niveau. Sur runner CPU, avec la suite complète désormais exécutée,
il faudra surveiller le temps de `test_monai_unet2d.py` et des smokes. Si ça déborde, marquer les cas
les plus lourds `slow` plutôt que de revenir à une liste de fichiers triée à la main.

### N6 (P2) — `config/` et les pilotes restent hors de la wheel

Documenté dans `pyproject.toml` : `include = ["oceanml3d*"]` ne prend ni `config/` ni `legacy/`, donc
**une distribution installée donne la bibliothèque, pas la CLI**. C'était acceptable tant que le seul
usage était un clone sur Odyssey. À partir du moment où le projet est destiné à d'autres personnes
sur d'autres plateformes, c'est un blocage : `pip install oceanml3d-core` suivi de
`oceanml3d experiment=...` ne peut pas fonctionner. Il faut soit déplacer `config/` sous
`oceanml3d/` avec le point d'entrée Hydra qui va avec, soit assumer par écrit que le mode
d'installation nominal est le clone (ou le conteneur).

---

## 5. Conséquence sur la feuille de route

L'ordre ne change pas, mais deux items montent.

**Lot 0 — à recompléter.** Les triggers CI sont toujours sur `master` à `ca5277c` : le lot 0 n'est pas
mergé. Il faut en plus y intégrer N3 et N4 (bornes torch dans `pyproject.toml`, `ci.yml`,
`environment.yml`) avant de pousser, sinon la CI qu'on vient de réparer échoue à l'installation.

**Lot 1 — inchangé sur le fond, renforcé par N1.** Les quatre blocages sont intacts. Le critère
d'acceptation devient plus explicite : un run complet des deux tâches **avec le tronc par défaut**,
relevé mémoire à l'appui, et l'arbitrage `num_res_blocks` / nombre de niveaux tranché sur cette base.

**Lot 3 — N2 y entre en tête.** Une assertion de non-trivialité dans le test de bout en bout est plus
urgente que les paramétrages de têtes restants : c'est ce qui distingue un pipeline qui marche d'un
pipeline qui écrit des fichiers.

**Nouveau lot — portabilité.** Absent de la première version, parce que la cible était Odyssey.
Maintenant que le projet vise plusieurs plateformes : définition Apptainer comme source de vérité,
couche de jobs indépendante de l'ordonnanceur (`jobs/run.sh` + enveloppes SLURM et PBS), fichiers
conda GPU et CPU, et arbitrage de N6. Ce lot absorbe et remplace l'item « nettoyer `batch/` » du
lot 4 : il ne s'agit plus de nettoyer 166 scripts SLURM mais de les remplacer par une couche qui
sert les deux ordonnanceurs.

---

## 6. En une phrase

Les deux PR d'agent sont de bonne facture et ne remettent rien en cause de l'audit précédent : les
treize défauts identifiés sont tous encore présents, au caractère près. Le basculement MONAI en
ajoute un de premier ordre — un modèle par défaut douze fois plus gros, validé uniquement sur
données synthétiques, derrière un test de bout en bout qui ne peut pas échouer sur un modèle mort.
