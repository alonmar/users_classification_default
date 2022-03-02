#!/usr/bin/env python

"""MLOps Library"""

import click
from mlib import predict
import json
import pandas as pd


@click.command()
@click.option(
    "--perfil",
    prompt="Users perfil",
    help="Pass the perfil of a user to predict if is a default",
)
def predictcli(perfil):
    """predict if is a default based on perfil"""
    data_to_predict = json.loads(perfil)
    data_to_predict = pd.DataFrame(data_to_predict)

    result = predict(data_to_predict)
    pred_class = result["Type_user"]
    if pred_class == "Moroso":
        click.echo(click.style(result, bg="red", fg="black"))
    else:
        click.echo(click.style(result, bg="green", fg="black"))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    predictcli()

# Example
# python cli.py --perfil '
# [{\"edad\": 25,
#  \"montoSolicitado\": 30000,
#  \"montoOtorgado\": 23000,
#  \"genero\": \"Hombre\",
#  \"quincenal\": 1,
#  \"dependientesEconomicos\": 3,
#  \"nivelEstudio\": \"Universidad\",
#  \"fico\": 569,
#  \"ingresosMensuales\": 15000,
#  \"gastosMensuales\": 4500,
#  \"emailScore\": 0,
#  \"browser\": \"CHROME_MOBILE\",
#  \"NUMTDC_AV\": 0}]'
