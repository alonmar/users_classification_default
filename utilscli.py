#!/usr/bin/env python
import click
import mlib
import requests
import json


@click.group()
@click.version_option("1.0")
def cli():
    """Machine Learning Utility Belt"""


@cli.command("retrain")
@click.option("--tsize", default=0.2, help="Test Size")
def retrain(tsize):
    """Retrain Model
    You may want to extend this with more options, such as setting model_name
    """

    click.echo(click.style(f"Retraining Model, tsize: {tsize}", bg="green", fg="black"))
    f1_value, model_name = mlib.retrain(tsize=tsize)
    click.echo(
        click.style(f"Retrained Model F1 score: {f1_value}", bg="blue", fg="black")
    )
    click.echo(click.style(f"Retrained Model Name: {model_name}", bg="red", fg="black"))


payload_default = """
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
  "NUMTDC_AV": 0}]
  """


@cli.command("predict")
@click.option("--perfil", default=payload_default, help="Perfil to Pass In")
@click.option("--host", default="http://localhost:8080/predict", help="Host to query")
def mkrequest(perfil, host):
    """Sends prediction to ML Endpoint"""

    click.echo(
        click.style(
            f"Querying host {host} \n with Perfil:\n {perfil}", bg="green", fg="black"
        )
    )

    payload = json.loads(perfil)
    result = requests.post(url=host, json=payload)
    click.echo(click.style(f"result: {result.text}", bg="red", fg="black"))


if __name__ == "__main__":
    cli()
