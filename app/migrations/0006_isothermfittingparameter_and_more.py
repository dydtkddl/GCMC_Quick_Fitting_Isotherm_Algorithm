# Generated by Django 5.1.1 on 2024-10-06 15:25

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0005_alter_gcmcresults_run_time"),
    ]

    operations = [
        migrations.CreateModel(
            name="IsothermFittingParameter",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("MOF", models.CharField(max_length=100)),
                ("MoleculeName", models.CharField(max_length=100)),
                ("Temperature", models.FloatField()),
                ("K1", models.FloatField(blank=True, null=True)),
                ("K2", models.FloatField(blank=True, null=True)),
                ("K3", models.FloatField(blank=True, null=True)),
                ("K4", models.FloatField(blank=True, null=True)),
                ("Isotherm_model", models.CharField(max_length=100)),
            ],
        ),
        migrations.AddField(
            model_name="gcmcresults",
            name="Isotherm_Fitting_Parameter",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to="app.isothermfittingparameter",
            ),
        ),
    ]
