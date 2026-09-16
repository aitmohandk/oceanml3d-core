# `oceanml3d-core` — audit de cohérence et de qualité, et feuille de route

Audit réalisé sur `main` @ `06c9a4a`, par lecture du code, `ruff`, analyse d'imports et
quantification des volumes de données réels des deux tâches grillées livrées.

**Verdict en une ligne.** L'architecture répond au cahier des charges ; l'implémentation ne tient pas
encore à l'échelle réelle. Quatre blocages de performance ou de justesse empêchent aujourd'hui un run
complet sur `surface_currents_15m` ou `osse3d_gs21`, et le filet de sécurité (CI) est inopérant sur
ce dépôt.

Sévérités : **P0** bloquant · **P1** corrige un résultat faux ou un coût majeur · **P2** dette · **P3** cosmétique.

---

## 1. Le cahier des charges est-il rempli ?

| Exigence | État | Commentaire |
|---|---|---|
| Paquet installable, modèles enregistrables | **Oui** | registre + 7 entry points, contrat `BaseOceanModel` minimal et respecté par les 7 modèles |
| Chemins hors du code | **Oui** | catalogue par site ; sauf les 7 clés OSSE-3D absentes (§4.4) |
| Description unique des variables | **Oui** | `VariableSet` pilote datamodule, modèle, perte et export ; aucune arithmétique de canaux ailleurs |
| Portage NOSC fonctionnel | **Oui sur la forme** | `nosc_unet` couvre OSE et OSSE-3D via `head`/`attention_levels`/`time_mode` |
| Contrat de produit inter-dépôts | **Oui** | v2, SHA-256 épinglé des deux côtés, testé |
| **Pipeline performant sur données réelles** | **Non** | §2 : 4 blocages, dont un OOM et une erreur silencieuse sous DDP |
| Reproductibilité train → predict | **Non** | §3.2 : `norm_stats.json` écrit, jamais relu |
| Couverture CI de la voie grillée | **Non** | §5 : CI jamais déclenchée sur `main`, et sa liste de tests ne contient aucun test grillé |

---

## 2. Performance — les quatre blocages

Chiffrage sur les deux tâches livrées. `surface_currents_15m` : domaine mondial 1/4°, soit 560 × 1440
points ; `patch.lat = 560`, `patch.lon = 1440` ⇒ **une seule position spatiale**, et `stride.time = 1`
sur 8 ans d'entraînement ⇒ 2 912 patchs. `osse3d_gs21` : boîte Gulf Stream 1/12°, 144 × 144, même
structure temporelle, 64 cibles.

### P0-1 — `_has_target` relit tout le jeu d'entraînement à chaque `setup()`

`PatchArray.__init__` filtre les patchs sans cible via `_has_target`, qui fait un `.values` dask
**par patch** :

```python
self._valid = [i for i in self._valid if self._has_target(i, drop_all_nan_targets)]
```

| Tâche | Octets par patch | Patchs | Lecture totale |
|---|---|---|---|
| `surface_currents_15m` | 2 × 11 × 560 × 1440 × 4 B = 71 Mo | 2 912 | **≈ 207 Go** |
| `osse3d_gs21` | 64 × 11 × 144 × 144 × 4 B = 58 Mo | 2 912 | **≈ 170 Go** |

Avec `stride.time = 1`, chaque pas de temps est relu **11 fois**. `drop_empty_target_patches` vaut
`True` par défaut, donc ce coût est payé systématiquement, avant la première époque.

**Correctif.** Une seule réduction sur le temps, puis un test par fenêtre :

```python
finite_t = np.isfinite(self.da.isel(channel=target_idx)).any(dim=("lat", "lon")).any(dim="channel").compute().values
# patch i valide ⇔ finite_t[sl["time"]].any()
```

O(1) passe au lieu de O(n_patchs), et le résultat est identique dès lors que la fenêtre spatiale
couvre le domaine — ce qui est le cas des deux configs livrées. Pour le cas général multi-position,
réduire sur `time` seulement et tester `finite[t, lat_slice, lon_slice].any()` sur un tableau en
mémoire.

### P0-2 — `predict_field` matérialise toutes les prédictions en RAM

```python
preds = trainer.predict(model, dataloaders=loader)
items = torch.cat(preds).float().cpu().numpy()   # (n_patches, n_targets, T, H, W)
```

Split test de `surface_currents_15m` : 2018-12-20 → 2020-01-10 = 387 jours ⇒ 377 patchs.

| Poste | Taille |
|---|---|
| `items` (float32) | 377 × 2 × 11 × 560 × 1440 × 4 B = **26,8 Go** |
| `acc` + `cnt` dans `reconstruct` (**float64**) | 2 × (2 × 387 × 560 × 1440 × 8 B) = **10,0 Go** |
| Pic | **≈ 37 Go** |

`reconstruct` alloue en `float64` sans raison : les prédictions sont en float32 et la sortie est
recastée en float32 à la fin.

**Correctif.** Accumuler par lot au lieu de concaténer : un `reconstruct_streaming(acc, cnt, item, i)`
appelé dans une boucle sur le `DataLoader`, accumulateurs en float32. Pic ramené à la taille du champ
reconstruit (≈ 2,5 Go pour cette tâche), indépendant du nombre de patchs.

### P0-3 — `predict_field` produit un champ faux sous DDP, sans erreur

`training=ddp` est un preset livré (`devices: 4`, `strategy: ddp`). `trainer.predict` répartit le
dataloader entre les rangs ; chaque rang ne reçoit que sa part. Or `reconstruct` associe l'élément *i*
de `items` au patch `self.slices(i)`, c'est-à-dire aux indices **contigus** :

```python
for i, item in enumerate(items):
    sl = self.slices(i)
```

Sur 4 GPU, le rang 0 reconstruit donc un quart des prédictions placées aux mauvaises fenêtres, et
écrit un produit d'apparence normale. Aucune exception, aucun avertissement.

**Correctif.** Faire l'export sur un `Trainer(devices=1)` reconstruit après `fit`, sous
`trainer.is_global_zero`, avec une barrière. À défaut, faire porter à chaque patch son index et
`all_gather` avant reconstruction. **Ajouter un test** qui reconstruit à partir d'items volontairement
permutés et vérifie l'échec.

### P1-4 — ~110 ouvertures de fichier pour un batch OSSE-3D

`open_variable` appelle `xr.open_dataset` une fois par variable. Dans `osse3d_gs21`, 64 cibles
viennent de `glorys_gs_multidepth`, 21 entrées ARGO et 21 masques de `argo_virtual_thetao_gs21` :
**~110 ouvertures pour 3 fichiers**, puis un `xr.concat` de 110 tableaux réindexés paresseusement. Chaque
`__getitem__` traverse ce graphe dask, par worker, par échantillon.

`cache: true` (zarr) résout le problème — mais vaut `false` par défaut.

**Correctifs.** (a) grouper les variables par `source` et n'ouvrir chaque fichier qu'une fois ;
(b) passer `cache: true` par défaut dès que le nombre de variables dépasse ~10, ou au minimum le
mettre dans les deux configs `osse3d_*`.

### P1-5 — coûts secondaires

| | |
|---|---|
| `compute_norm_stats` fait **deux** passes dask complètes (moyenne puis écart-type) | une seule passe via les moments d'ordre 1 et 2 |
| `spec.mask` ouvert avec `xr.open_dataarray` sans `chunks` | chargement eager, en contradiction avec le docstring « everything stays lazy » |
| `dm.setup("fit")` appelé dans `main()` avant le Trainer | sous relance DDP, chaque rang recalcule les stats sur tout le split train |

---

## 3. Bugs de justesse

### P1-1 — `prepare-obs` s'exécute sur tous les rangs DDP

`prepare_observations(cfg, catalog)` est appelé depuis `main()`, avant l'existence du Trainer.
Lightning relance le script par rang en DDP : 4 processus écrivent concurremment les mêmes NetCDF.
L'idempotence (« skips existing outputs ») ne protège pas d'une écriture simultanée — elle l'aggrave,
puisque le test d'existence passe avant que le premier écrivain ait fini.

**Correctif.** Déplacer dans `OceanDataModule.prepare_data()`, que Lightning garantit d'exécuter une
seule fois, ou encadrer par `rank_zero_only` + `trainer.strategy.barrier()`.

### P1-2 — `norm_stats.json` est écrit et jamais relu

`VersioningCallback.on_fit_start` sauve les statistiques à côté des checkpoints. Aucun code ne les
relit. `command=predict` reconstruit un datamodule depuis la config courante et **recalcule** les
statistiques. Si les splits, le domaine ou la liste de variables ont bougé entre l'entraînement et la
prédiction, le produit est dénormalisé avec des statistiques différentes de celles de
l'entraînement — erreur scientifique silencieuse, dans le chemin même qui alimente `oceanml3d-eval`.

**Correctif.** `predict` charge `norm_stats.json` à côté du checkpoint et échoue s'il est absent ;
avertit si les valeurs recalculées en diffèrent au-delà d'une tolérance.

### P1-3 — `depth_index: 0` est falsy

`oceanml3d/data/open.py` :

```python
elif spec.depth_index and var_name == spec.var_name:
    raise ValueError(f"{spec.name}: depth_index given but {path} has no depth axis ...")
```

La garde ne se déclenche jamais pour l'indice 0. Une variable déclarée `depth_index: 0` sur un fichier
sans axe de profondeur renvoie silencieusement le champ 2D. C'est exactement le cas de `thetao_d00`,
`uo_d00`, `vo_d00` dans `osse3d_gs21` (`depth_indices` commence par `0`). Même confusion — sans
conséquence celle-là — à `idx = spec.depth_index or 0`.

**Correctif.** `is not None` dans les deux cas.

### P1-4 — `reindex(method="nearest")` sans tolérance

```python
da = da.reindex(lat=reference.lat, lon=reference.lon, method="nearest")
```

Une variable sur une grille différente (1/4° contre 1/12°) est rééchantillonnée au plus proche voisin
au lieu de lever. Un mauvais fichier dans le catalogue produit un run qui converge sur des données
fausses. Ajouter `tolerance` dérivée du pas de la grille de référence.

### P1-5 — `reindex(time=reference.time)` masque les jours manquants

Une date absente devient NaN, puis 0 via `BaseOceanModel.inputs()`. Un trou d'un mois dans ERA5 passe
inaperçu. Compter les pas insérés et avertir au-delà d'un seuil.

### P2-1 — le test multi-étapes n'utilise pas le meilleur checkpoint

```python
trainer.test(model, datamodule=dm, ckpt_path="best" if len(stages) == 1 else None)
```

Avec plus d'une étape, `test` tourne sur les poids en mémoire de la dernière étape, pas sur le
meilleur. Incohérent avec le cas mono-étape et non documenté.

### P2-2 — `command=predict` contourne `load_from_checkpoint`

```python
state = torch.load(cfg.ckpt, ...)["state_dict"]
model.load_state_dict(state)
```

Le modèle est reconstruit depuis la config courante ; les `hparams` du checkpoint sont ignorés. Un
`loss_combine: uncertainty` à l'entraînement ajoute `log_vars` et fait échouer le chargement strict ;
une divergence qui ne change pas les formes passe silencieusement.

### P2-3 — le jitter est unilatéral

```python
off = int(self._rng.integers(0, min(self.spec.stride[d], room) + 1)) if room > 0 else 0
out[d] = slice(s.start + off, s.stop + off)
```

Décalage toujours positif : les premières cellules du domaine sont sous-échantillonnées et les
dernières sur-échantillonnées. Un offset égal à `stride` reproduit exactement la fenêtre suivante.
Centrer sur `[-stride/2, +stride/2]` en bornant aux deux extrémités.

### P2-4 — `validate_config` ne vérifie pas recouvrement ≥ 2 × crop

Les bordures rognées par `rec_weight.crop` doivent être couvertes par le patch voisin, sinon le champ
exporté porte des bandes vides. Les deux configs livrées sont **exactement à l'égalité**
(`144 − 136 = 8 = 2 × 4`), donc toute modification de `patch`, `stride` ou `crop` casse l'export en
silence. Deux lignes à ajouter dans `validate_config`.

---

## 4. Code superflu, mort, ou à chemins codés en dur

### 4.1 `batch/` — 178 fichiers, inexécutables en l'état

**303 chemins absolus** dans le dépôt, dont la quasi-totalité ici.

| Constat | Nombre |
|---|---|
| Scripts de job (`.sbatch` / `.slurm`) | 166 |
| Référençant `/Odyssey/private/rfablet/Python/4dvarnet-fm-opencode` | **113** |
| Référençant l'environnement conda d'un autre utilisateur (`/Odyssey/private/rfablet/miniforge3/envs/fdv/bin`) | **81** |
| Mentionnant `oceanml3d` | **8** |
| Contenant du Python inline en imports plats (`from data.lorenz63 import …`) | 5 |

La réécriture des 455 imports a porté sur les `.py`, pas sur le Python embarqué dans les `.sbatch`.
Plusieurs appellent encore `evaluation/sweep_qg_baselines.py` à son chemin d'avant le renommage.

**Décision à prendre :** soit un gabarit unique paramétré (`$REPO_ROOT`, `$CONDA_ENV`,
`$OCEANML3D_DATA`) plus une poignée de scripts vivants, soit une sortie pure et simple du dépôt.

### 4.2 `reports/` — 39 Mo, 131 fichiers

`reports/qg/` pèse à lui seul 34 Mo (PNG, PDF, GIF). C'est l'essentiel des 86 Mo du clone. Ce sont des
artefacts de résultats, pas du code, et ils sont déjà exclus de `ruff` et de la wheel.

### 4.3 Modules jamais référencés dans le paquet

`evaluation/run_l96_sweep.py`, `evaluation/run_l96_sweep2.py`, `evaluation/tune_l96_weak4dvar.py`,
`evaluation/experiment.py` — aucune référence nulle part. `evaluation/sweep_qg_baselines.py` n'est
appelé que par des sbatch pointant sur son ancien chemin.

### 4.4 Autres

- `archive/` (3 fichiers), `PLAN_upstream.md`, `docs/MIGRATION_PLAN.md`, `docs/REORG_PLAN.md`,
  `tests/legacy/generate_golden.py` (imports plats).
- Trois U-Nets (`models/unet.py`, `models/direct_unet.py`, `models/ocean/nn/unet2d.py`).
- Deux abstractions de dynamiques (`models/dynamics.py` en `if/elif` contre le registre `dynamics/`).
- Deux schémas de configuration (`conf/schema.py` 412 l. contre `config_schema.py` 146 l.).
- `config/experiment/` : 78 presets à plat, dont 11 seulement concernent la voie grillée.
- **Les 7 clés de catalogue OSSE-3D manquantes** (`glorys_gs_surface`, `glorys_gs_multidepth`,
  `bathy_gs`, `pseudo_obs_ssh_gs`, `pseudo_obs_sst_gs`, `argo_profiles_gs`,
  `argo_virtual_thetao_gs21`) : `osse3d_gs21` ne peut être lancée sur aucun site livré.

---

## 5. Filet de sécurité — la CI ne tourne pas

Deux défauts indépendants, chacun suffisant à la neutraliser.

**La branche.** `.github/workflows/ci.yml` se déclenche sur `feat/qg-*`, `feat/l96-*` et **`master`**.
La branche par défaut de ce dépôt est **`main`**, et la PR #1 ciblait `main`. **La CI ne s'est donc
jamais exécutée sur ce dépôt**, y compris sur le commit qui a réécrit 455 imports et transplanté
120 fichiers.

**La liste de tests.** Même déclenchée, la CI lance 18 fichiers nommés à la main : L96, QG,
`test_direct_unet`, `test_vanilla_cfm`, `test_metrics`… **aucun ne touche la voie grillée.** Le fichier
`tests/test_hydra_config.py` y figure, mais la version grillée a été renommée
`test_hydra_config_ocean.py` lors du transplant et n'a jamais été ajoutée à la liste.

Bilan : 779 tests passent en local, dont **environ 52 répartis sur 15 fichiers** couvrent la moitié
grillée du dépôt — et zéro sous CI.

**Trous de couverture les plus coûteux :** aucun test de `head=grouped`, `head=vertical_modes`,
`attention_levels`, `time_mode=conv3d` — les quatre options dont dépendent toutes les ablations
OSSE-3D ; aucun test d'ordre sur `reconstruct` ; aucun test du chemin d'export sous multi-device.

---

## 6. Dérive documentaire et méta

| | |
|---|---|
| `docs/adding_a_model.md` | chemins d'avant le renommage (`oceanml3d/models/<n>/`, `models/__init__.py`) |
| `docs/product_format.md` | titré « v1 », le contrat est en v2 ; champs `ensemble_size`/`coords.member` non documentés |
| `docs/data_preparation.md` | 15 lignes, ne couvre ni GLORYS multi-profondeurs, ni ARGO, ni `prepare-obs` |
| Dette ruff | annoncée **148**, réelle **168** — le détail par règle dans le même commentaire (27+26+21+10+84) somme bien à 168 ; c'est le total qui est faux |
| `LICENSE` | ~~absent, MIT non étayé~~ — **réglé au lot 0** : EUPL-1.2 avec accord du détenteur des droits |
| Environnement | ni `environment.yaml`, ni lockfile, ni conteneur ; l'env conda `fdv` n'existe que dans les `PATH` des sbatch |

---

## 7. Feuille de route

Cinq lots. Les lots 1 et 2 sont séquentiels, les autres peuvent s'intercaler.

### Lot 0 — Débloquer la vérification — **fait**, sauf la vérification en ligne

> Mis à jour sur `ca5277c`. Voir `docs/AUDIT_ca5277c.md` pour le delta : les 13 défauts ci-dessous
> sont tous encore présents, et le basculement MONAI en ajoute deux (N1, N2).

Rien d'autre ne mérite d'être fait tant que la CI ne tourne pas.

- [x] `ci.yml` : déclencheurs sur `main`, `feat/**` et `agents/**` (les deux PR qui ont changé le tronc par défaut étaient sur `agents/*`).
- [x] `pr_llm_review.yml` supprimé : mêmes déclencheurs morts, et il dépend de secrets et d'un compte de revue (`rfablet-review`) qui appartiennent au dépôt amont.
- [x] Remplacer la liste de 18 fichiers par `pytest -m "not slow" -q` sur tout `tests/`, ou au minimum ajouter les 15 fichiers grillés.
- [ ] **À faire côté GitHub :** vérifier que la CI passe au vert sur une PR de test avant tout autre travail.
- [x] `LICENSE` (EUPL-1.2 verbatim), `NOTICE`, `LICENSING.md`, classifier. Accord du détenteur des droits (IMT Atlantique / OceaniX).
- [ ] Renseigner la référence de l'accord (qui, quand) dans `LICENSING.md`.
- [ ] En-têtes SPDX sur les fichiers Python ; même traitement sur `oceanml3d-eval`.
- [x] **N3/N4** : borner `torch>=2.4.1,<2.7` (fenêtre imposée par `monai<1.6`) dans `pyproject.toml`, `ci.yml` et les fichiers conda.
- [x] `environment.yml` (GPU) et `environment-cpu.yml` reproduisant l'env `fdv`, plus les extras.

### Lot 1 — Rendre le pipeline grillé exécutable à l'échelle (3–5 jours)

- [x] **P0-1** `_has_target` → `_valid_patches` : une réduction par *fenêtre spatiale* (une seule pour les deux tâches livrées), coût indépendant du pas temporel. *Test : équivalence de `_valid` avec l'implémentation actuelle sur un cas synthétique à trous.*
- [x] **P0-2** `predict_field` : accumulation en flux via `PatchAccumulator` ; `reconstruct` devient une enveloppe, résultat bit-à-bit identique. *Test : identité au bit près avec la version actuelle sur le cas synthétique ; mesure du pic RSS.*
- [x] **P0-3** `result()` refuse un champ partiel (le vrai garde-fou, testable sans DDP) ; `predict_field` prédit sur un seul device ; `_export` sur le rang 0 avec barrière. *Test : `reconstruct` sur items permutés doit lever.*
- [x] **P1-4** Une poignée de fichier par `source` pour tout le jeu de variables (~110 ouvertures → 3 sur `osse3d_gs21`).
- [ ] Activer `cache: true` dans les configs `osse3d_*` (décision à prendre au vu du run complet).
- [x] **P1-5** `compute_norm_stats` en une passe (changement d'ordonnancement, valeurs identiques) ; `mask` chunké et mis en cache.
- [ ] Un run complet de `nosc_15m_duacs` et un de `osse3d_gs21_multivar_unet`, avec relevé temps/mémoire consigné dans `CHANGELOG.md`. **C'est le critère d'acceptation du lot.**

### Lot 2 — Justesse (2–3 jours)

- [ ] **P1-1** `prepare-obs` dans `prepare_data()`. *Test : deux processus concurrents.*
- [ ] **P1-2** `predict` charge `norm_stats.json`, échoue si absent, avertit si divergent. *Test : entraîner sur un split, prédire sur un autre, vérifier l'avertissement.*
- [x] **P1-3** `is not None` sur `depth_index` (aux deux endroits). *Test : `depth_index: 0` sur un fichier 2D doit lever.*
- [x] **P1-4/5** `_align_space` refuse une grille différente (tolérance d'un demi-pas) ; `_align_time` annonce les pas inventés.
- [ ] **P2-1** `ckpt_path="best"` en multi-étapes, ou documenter le choix inverse.
- [ ] **P2-2** `predict` via `load_from_checkpoint`.
- [ ] **P2-3** Jitter centré.
- [x] **P2-4** Recouvrement ≥ 2 × crop vérifié, avec le stride correctif nommé dans le message. *Test : une config à recouvrement insuffisant doit être rejetée.*

### Lot 3 — Couverture (2–3 jours)

- [x] **N2** : assertion de non-trivialité. `test_the_trunk_can_actually_fit_a_batch` (les deux troncs, non `slow`) et écart-type spatial non nul sur le produit exporté dans `test_train_predict_export`.

- [ ] Paramétrer `test_model_smoke.py` sur `head ∈ {single, grouped, vertical_modes}`, `time_mode ∈ {channels, conv3d}`, `attention_levels ∈ {[], [2]}` — 12 combinaisons en forward seul, rapides.
- [ ] Test de bout en bout `prepare-obs → train (1 époque) → predict → validate_manifest` sur données synthétiques, marqué `slow`, exécuté en CI nocturne.
- [ ] Tests d'ordre et de bordure sur `reconstruct` (permutation, recouvrement insuffisant).
- [ ] Ajouter les 7 clés OSSE-3D à `config/paths/local.yaml` et un test qui valide **toutes** les expériences grillées avec `validate_config`.

### Lot 4 — Nettoyage (2 jours)

- [ ] `batch/` : gabarit unique paramétré (`$REPO_ROOT`, `$CONDA_ENV`, `$OCEANML3D_DATA`) + conserver au plus une dizaine de scripts vivants ; sortir le reste.
- [ ] `reports/` hors du dépôt (release GitHub, ou dépôt d'artefacts). Ramène le clone de 86 Mo à ~45 Mo.
- [ ] Supprimer les 4 modules jamais référencés, `archive/`, `tests/legacy/generate_golden.py`, `PLAN_upstream.md`, `docs/MIGRATION_PLAN.md`, `docs/REORG_PLAN.md`.
- [x] ~~Sous-répertoires `config/experiment/{toy,gridded}/`~~ — fait par le passage en `config/legacy/`.
- [ ] Corriger `docs/adding_a_model.md`, `docs/product_format.md`, `docs/data_preparation.md` ; corriger 148 → 168 dans le commentaire ruff.

### Lot 5 — Dette structurelle (à planifier, ~1 semaine)

- [ ] Fusionner `conf/schema.py` et `config_schema.py`, **ou** acter par écrit que les deux CLI sont permanentes.
- [ ] Converger `models/dynamics.py` sur le registre `dynamics/`.
- [ ] Ramener les trois U-Nets à un.
- [ ] Solder les 168 violations ruff par vagues : d'abord les 73 auto-corrigeables (UP006/UP045), puis **B023 × 27 en audit manuel — certaines sont de vrais bugs de capture de variable de boucle**, puis B905/B008/B007.
- [ ] Décider du sort des 27 commits amont non repris (backbone MONAI FDV1-Stier, `prior_hidden_channels`, truncated-BPTT, campagne QG S0/S1) : reprise manuelle chiffrée, ou abandon acté dans `PROVENANCE.md`.
- [ ] Profondeur comme dimension de patch (`PatchArray` est déjà générique sur `DIMS`).

---

## 8. Ordre de priorité, en une phrase

Réparer la CI (Lot 0) ; rendre les deux tâches grillées exécutables et mesurées (Lot 1) ; fermer les
erreurs silencieuses qui contaminent les produits envoyés à `oceanml3d-eval` (Lot 2) ; seulement
ensuite, nettoyer.

Les trois défauts les plus coûteux si on ne fait rien : la reconstruction fausse sous DDP (P0-3), qui
produit des scores plausibles et faux ; les statistiques de normalisation non rejouées (P1-2), même
effet ; et la CI inopérante (§5), qui garantit qu'aucun des deux ne sera détecté.
