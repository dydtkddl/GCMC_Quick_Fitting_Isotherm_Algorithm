# Generated by Django 5.1.1 on 2024-10-06 06:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0003_gcmcresults_log_symbol"),
    ]

    operations = [
        migrations.AddField(
            model_name="gcmcresults",
            name="run_time",
            field=models.CharField(default=None, max_length=200),
        ),
    ]
