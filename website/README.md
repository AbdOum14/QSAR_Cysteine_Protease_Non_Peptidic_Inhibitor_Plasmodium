# QSAR Falcipaína-2 — predicción actividad/inactividad

Predice si una molécula (dada como SMILES) sería **activa** o **inactiva**
frente a la Falcipaína-2 de *Plasmodium falciparum* (activo = IC50 < 5 µM),
replicando el flujo usado en el artículo.

## Idea general 
 **Inferencia** (`inference_pipeline.py`) — es lo que usa la web. Carga
   los dos ficheros joblib que poseen los modelos ya entrenados y, para cada molécula que envíe el
   usuario, tarda segundos en devolver la predicción.

## Forma de la respuesta JSON

```json
{
  "resultados": [
    {
      "id": "mol_1",
      "smiles": "CC(=O)Oc1ccccc1C(=O)O",
      "mordred": {
        "prediccion": "Activo",
        "probabilidad_activo": 0.69,
        "consenso_3_modelos": "mayoritario",
        "votos_a_favor_de_activo": 2,
        "dentro_del_dominio_de_aplicabilidad": true,
        "distancia_dominio_aplicabilidad": 1.17,
        "modelos_usados": ["LogisticRegression", "RandomForest", "SVM_RBF"],
        "explicacion_top_descriptores": [
          {"descriptor": "nAcid", "valor_molecula": 1.0,
           "contribucion": 0.135, "efecto": "aumenta probabilidad de actividad"}
        ]
      },
      "maccs": { "...": "mismo formato" }
    }
  ],
  "errores": [
    {"id": "mol_2", "smiles": "algo_mal_escrito", "motivo": "SMILES invalido..."}
  ]
}
```

Todo vive en `inference_pipeline.py`:
| Función | Qué hace |
|---|---|
| `read_input()` | Normaliza SMILES sueltos o un CSV subido a una tabla `id, smiles` |
| `validate_and_parse_smiles()` | Descarta SMILES mal escritos, avisa cuáles fallaron |
| `compute_mordred_descriptors()` | Calcula ~1600 descriptores fisicoquímicos con RDKit/Mordred |
| `compute_maccs_keys()` | Calcula el fingerprint estructural MACCS (166 bits) |
| `align_features()` | Deja solo las columnas que el modelo espera, en el orden correcto |
| `check_applicability_domain()` | Dice si la molécula es "conocida" para el modelo (fiabilidad) |
| `predict_with_top3()` | Ejecuta los 3 mejores modelos y hace un consenso |
| `explain_prediction()` | Explica, molécula a molécula, qué descriptores empujan hacia activo/inactivo |
| `predict_falcipain2()` | Junta todo lo anterior — normalmente es la única que se llama |

## Notas de rendimiento

- Mordred + SHAP por molécula es lo más lento (~1-3 s/molécula). Para CSV
  con muchas moléculas, considera poner `explain=False` en la llamada
  masiva y ofrecer la explicación solo cuando el usuario pincha en una
  molécula concreta (llamando de nuevo con esa única SMILES y `explain=True`).
- El dominio de aplicabilidad (`check_applicability_domain`) y la
  predicción (`predict_with_top3`) son prácticamente instantáneos.