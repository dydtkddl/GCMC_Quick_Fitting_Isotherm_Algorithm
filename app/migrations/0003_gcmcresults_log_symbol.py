# Generated by Django 5.1.1 on 2024-10-05 17:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0002_gcmcresults_description"),
    ]

    operations = [
        migrations.AddField(
            model_name="gcmcresults",
            name="log_symbol",
            field=models.CharField(default="log_symbol", max_length=200),
        ),
    ]
