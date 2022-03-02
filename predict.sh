#!/usr/bin/env bash

PORT=8080
echo "Port: $PORT"

# POST method predict
curl -d '
   [{"edad": 25,
  "montoSolicitado": 30000,
  "montoOtorgado": 23000,
  "genero": "Hombre",
  "quincenal": 1,
  "dependientesEconomicos": 3,
  "nivelEstudio": "Universidad",
  "fico": 569,
  "ingresosMensuales": 15000,
  "gastosMensuales": 4500,
  "emailScore": 0,
  "browser": "CHROME_MOBILE",
  "NUMTDC_AV": 0}]'\
     -H "Content-Type: application/json" \
     -X POST http://localhost:$PORT/predict