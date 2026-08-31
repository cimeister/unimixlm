# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import carbon.compounds.core.models
import lesleyloraine.s3utils


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='csspackage',
            name='file_minified',
            field=models.FileField(storage=lesleyloraine.s3utils._MediaS3BotoStorage(bucket=b'lesleyloraine-dev', custom_domain=b'lesleyloraine-dev.s3.amazonaws.com', location=b'media'), null=True, upload_to=carbon.compounds.core.models.title_file_name, blank=True),
        ),
        migrations.AlterField(
            model_name='csspackage',
            name='file_source',
            field=models.FileField(storage=lesleyloraine.s3utils._MediaS3BotoStorage(bucket=b'lesleyloraine-dev', custom_domain=b'lesleyloraine-dev.s3.amazonaws.com', location=b'media'), null=True, upload_to=carbon.compounds.core.models.title_file_name, blank=True),
        ),
        migrations.AlterField(
            model_name='jspackage',
            name='file_minified',
            field=models.FileField(storage=lesleyloraine.s3utils._MediaS3BotoStorage(bucket=b'lesleyloraine-dev', custom_domain=b'lesleyloraine-dev.s3.amazonaws.com', location=b'media'), null=True, upload_to=carbon.compounds.core.models.title_file_name, blank=True),
        ),
        migrations.AlterField(
            model_name='jspackage',
            name='file_source',
            field=models.FileField(storage=lesleyloraine.s3utils._MediaS3BotoStorage(bucket=b'lesleyloraine-dev', custom_domain=b'lesleyloraine-dev.s3.amazonaws.com', location=b'media'), null=True, upload_to=carbon.compounds.core.models.title_file_name, blank=True),
        ),
    ]